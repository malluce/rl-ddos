#!/usr/bin/env python3

# coding=utf-8
# Lint as: python2, python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

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
from lib.datastore import Datastore
from gyms.hhh.state import *

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string('gin_file', None, 'Paths to the gin-config files.')

FLAGS = flags.FLAGS


@gin.configurable
def train_eval(
        root_dir,
        timestamp=None,
        env_name='HHHGym-v0',
        num_iterations=100000,
        train_sequence_length=1,
        # Params for QNetwork
        fc_layer_params=(100, 50),
        # Params for QRnnNetwork
        batch_normalization=False,
        input_fc_layer_params=(200, 200),
        lstm_size=(100,),
        output_fc_layer_params=(20,),
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
        batch_size=1024,  # 64
        learning_rate=1e-4,
        learning_rate_decay_rate=None,
        learning_rate_decay_step=None,
        n_step_update=1,
        gamma=0.99,
        reward_scale_factor=1.0,
        gradient_clipping=None,
        use_tf_functions=True,
        # Params for eval
        num_eval_episodes=5,
        eval_interval=10000,
        # Params for checkpoints
        train_checkpoint_interval=10000,
        policy_checkpoint_interval=5000,
        rb_checkpoint_interval=20000,
        # Params for summaries and logging
        log_interval=1000,
        summary_interval=1000,
        summaries_flush_secs=10,
        debug_summaries=False,
        summarize_grads_and_vars=False,
        state_obs_selection=[BaseObservations(), FalsePositiveRate(), DistVol(), MinMaxBlockedAddress(), DistVolStd(),
                             BlocklistDistribution()],
        use_prev_action_as_obs=True,
        actionset_selection=LargeDiscreteActionSet(),
        trace_length=100):
    """A simple train and eval for DQN."""
    dirs = get_dirs(root_dir, timestamp, 'dqn')
    eval_summary_writer, train_summary_writer = get_summary_writers(dirs, summaries_flush_secs)
    global_step = tf.compat.v1.train.get_or_create_global_step()
    eval_metrics, train_metrics = get_metrics(num_eval_episodes)

    with train_summary_writer.as_default():
        ds_train = Datastore(dirs['root'], 'train')
        ds_eval = Datastore(dirs['root'], 'eval')
        ds_eval.commit_config(gin.operative_config_str())

        eval_tf_env, tf_env = get_envs(actionset_selection, ds_eval, ds_train, env_name, state_obs_selection,
                                       trace_length, use_prev_action_as_obs)

        if train_sequence_length != 1 and n_step_update != 1:
            raise NotImplementedError(
                'train_eval does not currently support n-step updates with stateful '
                'networks (i.e., RNNs)')

        if train_sequence_length > 1:
            logging.info('Using QRnnNetwork')
            q_net = q_rnn_network.QRnnNetwork(
                tf_env.observation_spec(),
                tf_env.action_spec(),
                input_fc_layer_params=input_fc_layer_params,
                lstm_size=lstm_size,
                output_fc_layer_params=output_fc_layer_params)
        else:
            logging.info('Using QNetwork')
            q_net = q_network.QNetwork(
                tf_env.observation_spec(),
                tf_env.action_spec(),
                #				batch_normalization=batch_normalization,
                fc_layer_params=fc_layer_params)
            train_sequence_length = n_step_update

        # TODO decay lr?

        # TODO decay epsilon?

        tf_agent = dqn_agent.DqnAgent(
            tf_env.time_step_spec(),
            tf_env.action_spec(),
            q_network=q_net,
            epsilon_greedy=epsilon_greedy,
            n_step_update=n_step_update,
            target_update_tau=target_update_tau,
            target_update_period=target_update_period,
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
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

        policy_checkpointer, rb_checkpointer, train_checkpointer = get_checkpointers(dirs, eval_policy, global_step,
                                                                                     replay_buffer, tf_agent,
                                                                                     train_metrics)

        if use_tf_functions:
            collect_driver.run = common.function(collect_driver.run)
            tf_agent.train = common.function(tf_agent.train)

        initial_collect_policy = random_tf_policy.RandomTFPolicy(
            tf_env.time_step_spec(), tf_env.action_spec())

        if not rb_checkpointer.checkpoint_exists:
            logging.info('Initializing replay buffer')

            dynamic_step_driver.DynamicStepDriver(
                tf_env,
                initial_collect_policy,
                observers=[replay_buffer.add_batch] + train_metrics,
                num_steps=initial_collect_steps).run()

            results = metric_utils.eager_compute(
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

        # Dataset generates trajectories with shape [batch_size,num_steps,obs_spec]
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
        policy_state = collect_policy.get_initial_state(tf_env.batch_size)
        time_step = None
        timed_at_step = global_step.numpy()
        time_acc = 0
        for _ in range(num_iterations):
            start_time = time.time()
            time_step, policy_state = collect_driver.run(
                time_step=time_step,
                policy_state=policy_state,
            )
            for _ in range(train_steps_per_iteration):
                train_loss = train_step()

            log_and_eval(eval_interval, eval_metrics, eval_policy, eval_summary_writer, eval_tf_env, global_step,
                         log_interval, num_eval_episodes, policy_checkpoint_interval, policy_checkpointer,
                         rb_checkpoint_interval, rb_checkpointer, start_time, time_acc, timed_at_step,
                         train_checkpoint_interval, train_checkpointer, train_loss, train_metrics, use_tf_functions)


def log_and_eval(eval_interval, eval_metrics, eval_policy, eval_summary_writer, eval_tf_env, global_step, log_interval,
                 num_eval_episodes, policy_checkpoint_interval, policy_checkpointer, rb_checkpoint_interval,
                 rb_checkpointer, start_time, time_acc, timed_at_step, train_checkpoint_interval, train_checkpointer,
                 train_loss, train_metrics, use_tf_functions):
    time_acc += time.time() - start_time
    if global_step.numpy() % log_interval == 0:
        logging.info('step = %d, loss = %f', global_step.numpy(),
                     train_loss.loss)
        steps_per_sec = (global_step.numpy() - timed_at_step) / time_acc
        logging.info('%.3f steps/sec', steps_per_sec)
        tf.compat.v2.summary.scalar(name='global_steps_per_sec',
                                    data=steps_per_sec, step=global_step)
        timed_at_step = global_step.numpy()
        time_acc = 0
    for train_metric in train_metrics:
        train_metric.tf_summaries(train_step=global_step,
                                  step_metrics=train_metrics[:2])
    if global_step.numpy() % train_checkpoint_interval == 0:
        train_checkpointer.save(global_step=global_step.numpy())
    if global_step.numpy() % policy_checkpoint_interval == 0:
        policy_checkpointer.save(global_step=global_step.numpy())
    if global_step.numpy() % rb_checkpoint_interval == 0:
        rb_checkpointer.save(global_step=global_step.numpy())
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


def get_metrics(num_eval_episodes):
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
    return eval_metrics, train_metrics


def get_summary_writers(dirs, summaries_flush_secs):
    train_summary_writer = tf.compat.v2.summary.create_file_writer(
        dirs['tf_train'], flush_millis=summaries_flush_secs * 1000)
    train_summary_writer.set_as_default()
    eval_summary_writer = tf.compat.v2.summary.create_file_writer(
        dirs['tf_eval'], flush_millis=summaries_flush_secs * 1000)
    return eval_summary_writer, train_summary_writer


def get_envs(actionset_selection, ds_eval, ds_train, env_name, state_obs_selection, trace_length,
             use_prev_action_as_obs):
    gym_kwargs = {
        'state_obs_selection': state_obs_selection,
        'use_prev_action_as_obs': use_prev_action_as_obs,
        'actionset': actionset_selection,
        'trace_length': trace_length
    }
    train_gym = suite_gym.load(
        env_name, gym_kwargs={'data_store': ds_train, **gym_kwargs},
        spec_dtype_map={gym.spaces.Discrete: np.int32})
    tf_env = tf_py_environment.TFPyEnvironment(train_gym)
    eval_tf_env = tf_py_environment.TFPyEnvironment(suite_gym.load(
        env_name, gym_kwargs={'data_store': ds_eval, **gym_kwargs},
        spec_dtype_map={gym.spaces.Discrete: np.int32}))
    return eval_tf_env, tf_env


def get_checkpointers(dirs, eval_policy, global_step, replay_buffer, tf_agent, train_metrics):
    train_checkpointer = common.Checkpointer(
        ckpt_dir=dirs['chkpt'],
        agent=tf_agent,
        global_step=global_step,
        metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'))
    policy_checkpointer = common.Checkpointer(
        ckpt_dir=dirs['policy_chkpt'],
        policy=eval_policy,
        global_step=global_step)
    rb_checkpointer = common.Checkpointer(
        ckpt_dir=dirs['replay_buf_chkpt'],
        max_to_keep=1,
        replay_buffer=replay_buffer)

    train_checkpointer.initialize_or_restore()
    rb_checkpointer.initialize_or_restore()

    return policy_checkpointer, rb_checkpointer, train_checkpointer


def main(_):
    register_hhh_gym()
    logging.set_verbosity(logging.INFO)
    tf.get_logger().setLevel('INFO')
    # gin.parse_config_file(FLAGS.gin_file)
    train_eval(FLAGS.root_dir, timestamp=Datastore.get_timestamp())
    return 0


if __name__ == '__main__':
    try:
        flags.mark_flag_as_required('root_dir')
        app.run(main)
    except KeyboardInterrupt:
        pass
