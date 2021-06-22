#!/usr/bin/env python3

# coding=utf-8
# Lint as: python2, python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import tf_agents.drivers.dynamic_episode_driver
from absl import app
from absl import flags
from absl import logging

import gin
import gym
from six.moves import range
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments.examples import masked_cartpole  # pylint: disable=unused-import
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
# from nets import dqn_q_network as q_network
from tf_agents.networks import q_rnn_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

from agents.util import get_dirs
from gyms.hhh.actionset import LargeDiscreteActionSet
from gyms.hhh.env import register_hhh_gym
from gyms.hhh.state import BaseObservations
from lib.datastore import Datastore

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string('timestamp', None,
                    'Restore from specified timestamp.')
flags.DEFINE_integer('num_iterations', 1000,
                     'Total number train/eval iterations to perform.')
FLAGS = flags.FLAGS


@gin.configurable
def train_eval(
        root_dir,
        timestamp=None,
        env_name='HHHGym-v0',
        num_iterations=10000,
        train_sequence_length=1,
        # Params for QNetwork
        fc_layer_params=(100, 50),
        # Params for collect
        initial_collect_steps=1000,
        collect_steps_per_iteration=1,
        epsilon_greedy=0.1,
        replay_buffer_capacity=100000,
        # Params for target update
        target_update_tau=0.05,
        target_update_period=5,
        # Params for train
        train_steps_per_iteration=1,
        batch_size=64,
        learning_rate=1e-4,
        n_step_update=1,
        gamma=0.99,
        reward_scale_factor=1.0,
        gradient_clipping=None,
        use_tf_functions=True,
        # Params for eval
        num_eval_episodes=5,
        eval_interval=100,
        # Params for summaries and logging
        log_interval=20,
        summaries_flush_secs=10,
        debug_summaries=False,
        summarize_grads_and_vars=False,
        state_selection=[BaseObservations()],
        actionset_selection=LargeDiscreteActionSet(),
        use_prev_action_as_obs=False,
        trace_length=50000):
    """A simple train and eval for DQN."""

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    if timestamp is None:
        timestamp = Datastore.get_timestamp()

    # setup writers
    dirs = get_dirs(root_dir, timestamp, 'dqn-play')
    train_summary_writer = tf.compat.v2.summary.create_file_writer(
        dirs['tf_train'], flush_millis=summaries_flush_secs * 1000)
    eval_summary_writer = tf.compat.v2.summary.create_file_writer(
        dirs['tf_eval'], flush_millis=summaries_flush_secs * 1000)

    # setup metrics
    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(),
        tf_metrics.AverageEpisodeLengthMetric(),
        tf_metrics.ChosenActionHistogram()
    ]
    eval_metrics = [
        tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
        tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes),
        #		tf_metrics.ChosenActionHistogram()
    ]

    # setup global step counter
    global_step = tf.compat.v1.train.get_or_create_global_step()

    # setup datastore
    ds_train = Datastore(dirs['root'], 'train')
    ds_eval = Datastore(dirs['root'], 'eval')

    # setup envs (train, eval)
    gym_kwargs = {
        'state_obs_selection': state_selection,
        'actionset': actionset_selection,
        'trace_length': trace_length,
        'use_prev_action_as_obs': use_prev_action_as_obs
    }
    train_gym = suite_gym.load(
        env_name, gym_kwargs={'data_store': ds_train, **gym_kwargs},
        spec_dtype_map={gym.spaces.Discrete: np.int32})
    tf_env = tf_py_environment.TFPyEnvironment(train_gym)
    eval_tf_env = tf_py_environment.TFPyEnvironment(suite_gym.load(
        env_name, gym_kwargs={'data_store': ds_eval, **gym_kwargs},
        spec_dtype_map={gym.spaces.Discrete: np.int32}))

    # setup DQN
    q_net = q_network.QNetwork(
        tf_env.observation_spec(),
        tf_env.action_spec(),
        fc_layer_params=fc_layer_params)
    train_sequence_length = n_step_update

    # setup agent
    # TODO(b/127301657): Decay epsilon based on global step, cf. cl/188907839
    tf_agent = dqn_agent.DqnAgent(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        q_network=q_net,
        epsilon_greedy=epsilon_greedy,
        n_step_update=n_step_update,
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        #			optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = learning_rate),
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        td_errors_loss_fn=common.element_wise_squared_loss,
        gamma=gamma,
        reward_scale_factor=reward_scale_factor,
        gradient_clipping=gradient_clipping,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=global_step)
    tf_agent.initialize()

    eval_policy = tf_agent.policy
    collect_policy = tf_agent.collect_policy

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=tf_env.batch_size,
        max_length=replay_buffer_capacity)
    collect_driver = dynamic_step_driver.DynamicStepDriver(
        tf_env,
        collect_policy,
        observers=[replay_buffer.add_batch] + train_metrics,
        num_steps=collect_steps_per_iteration)

    if use_tf_functions:
        collect_driver.run = common.function(collect_driver.run)
        tf_agent.train = common.function(tf_agent.train)

    ds_eval.commit_config(gin.operative_config_str())

    logging.info('Initializing replay buffer')
    dynamic_step_driver.DynamicStepDriver(
        tf_env,
        random_tf_policy.RandomTFPolicy(
            tf_env.time_step_spec(), tf_env.action_spec()),
        observers=[replay_buffer.add_batch] + train_metrics,
        num_steps=initial_collect_steps).run()

    time_step = None
    policy_state = collect_policy.get_initial_state(tf_env.batch_size)
    timed_at_step = global_step.numpy()
    time_acc = 0

    # Dataset generates trajectories with shape [Bx2x...]
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=train_sequence_length + 1).prefetch(3)
    iterator = iter(dataset)

    def train_step():
        experience, _ = next(iterator)
        return tf_agent.train(experience)

    if use_tf_functions:
        train_step = common.function(train_step)

    logging.info('Training')

    for _ in range(num_iterations):
        start_time = time.time()
        time_step, policy_state = collect_driver.run(
            time_step=time_step,
            policy_state=policy_state,
        )
        for _ in range(train_steps_per_iteration):
            train_loss = train_step()
        time_acc += time.time() - start_time

        if global_step.numpy() % log_interval == 0:
            logging.info('step = %d, loss = %f', global_step.numpy(),
                         train_loss.loss)
            steps_per_sec = (global_step.numpy() - timed_at_step) / time_acc
            logging.info('%.3f steps/sec', steps_per_sec)
            with train_summary_writer.as_default():
                tf.compat.v2.summary.scalar(name='global_steps_per_sec',
                                            data=steps_per_sec, step=global_step)
            timed_at_step = global_step.numpy()
            time_acc = 0

        if global_step.numpy() % eval_interval == 0:
            metric_utils.eager_compute(
                eval_metrics,
                eval_tf_env,
                eval_policy,
                num_episodes=num_eval_episodes,
                train_step=global_step,
                summary_writer=eval_summary_writer,
                summary_prefix='Metrics',
                use_function=use_tf_functions
            )
            metric_utils.log_metrics(eval_metrics)


def main(_):
    env_name = register_hhh_gym()
    logging.set_verbosity(logging.INFO)
    tf.get_logger().setLevel('INFO')
    train_eval(FLAGS.root_dir, FLAGS.timestamp, num_iterations=FLAGS.num_iterations, env_name=env_name)
    return 0  # exit code


if __name__ == '__main__':
    try:
        flags.mark_flag_as_required('root_dir')
        app.run(main)
    except KeyboardInterrupt:
        pass
