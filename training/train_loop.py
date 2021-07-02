import time
from abc import ABC, abstractmethod

import gin
import tensorflow as tf
from typing import Tuple

from absl import logging

from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.environments import suite_gym
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.utils import common

from agents.util import get_dirs
from gyms.hhh.actionset import ActionSet
from gyms.hhh.obs import Observation
from lib.datastore import Datastore
from training.td3.td3_wrap_agent import TD3WrapAgent


@gin.configurable
class TrainLoop(ABC):
    def __init__(self,
                 root_dir: str,
                 # env params
                 env_name: str, actionset_selection: ActionSet,
                 state_obs_selection: Tuple[Observation], trace_length: int,
                 use_prev_action_as_obs: bool,
                 # training params
                 num_iterations: int = 200000, eval_interval: int = 2400, num_eval_episodes: int = 10,
                 batch_size: int = 64,
                 replay_buf_size: int = 100000,
                 initial_collect_steps: int = 1200, log_interval: int = 600,
                 checkpoint_interval: int = 10000
                 ):

        self.checkpoint_interval = checkpoint_interval
        self.num_iterations = num_iterations
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes
        self.batch_size = batch_size,
        self.replay_buf_size = replay_buf_size
        self.initial_collect_steps = initial_collect_steps
        self.log_interval = log_interval

        (self.train_env, self.eval_env) = (None, None)  # set in _init_envs
        self.dirs = get_dirs(root_dir, Datastore.get_timestamp(), self._get_alg_name())
        self._init_envs(actionset_selection, self.dirs, env_name, state_obs_selection, trace_length,
                        use_prev_action_as_obs)

        (self.train_summary_writer, self.eval_summary_writer) = (None, None)  # set in _init_summary_writers
        self._init_summary_writers(self.dirs)

        (self.train_metrics, self.eval_metrics) = (None, None)  # set in _init_metrics
        (self.replay_buffer, self.dataset_iterator) = (None, None)  # set in _init_replay_buffer
        (self.collect_driver, self.initial_collect_driver) = (None, None)  # set in _init_drivers

        self.agent = self._get_agent()
        self.ds_eval.commit_config(gin.operative_config_str())

        # set in _init_checkpointers
        (self.policy_checkpointer, self.train_checkpointer, self.rb_checkpointer) = (None, None, None)

    @abstractmethod
    def _get_agent(self):
        pass

    @abstractmethod
    def _get_alg_name(self):
        pass

    def _init_envs(self, actionset_selection, dirs, env_name, state_obs_selection, trace_length,
                   use_prev_action_as_obs):
        ds_train = Datastore(dirs['root'], 'train')
        self.ds_eval = Datastore(dirs['root'], 'eval')
        gym_kwargs = {
            'state_obs_selection': state_obs_selection,
            'use_prev_action_as_obs': use_prev_action_as_obs,
            'actionset': actionset_selection,
            'trace_length': trace_length
        }
        self.train_env = TFPyEnvironment(suite_gym.load(env_name, gym_kwargs={'data_store': ds_train, **gym_kwargs}))
        self.eval_env = TFPyEnvironment(suite_gym.load(env_name, gym_kwargs={'data_store': self.ds_eval, **gym_kwargs}))

    def _init_summary_writers(self, dirs):
        flush_seconds = 10 * 1000
        self.train_summary_writer = tf.compat.v2.summary.create_file_writer(dirs['tf_train'],
                                                                            flush_millis=flush_seconds)
        self.train_summary_writer.set_as_default()
        self.eval_summary_writer = tf.compat.v2.summary.create_file_writer(dirs['tf_eval'], flush_millis=flush_seconds)

    def _init_metrics(self):
        self.eval_metrics = [
            tf_metrics.AverageReturnMetric(buffer_size=self.num_eval_episodes),
            tf_metrics.AverageEpisodeLengthMetric(buffer_size=self.num_eval_episodes),
            tf_metrics.ChosenActionHistogram(buffer_size=self.num_eval_episodes)
        ]
        self.train_metrics = [
            tf_metrics.NumberOfEpisodes(),
            tf_metrics.EnvironmentSteps(),
            tf_metrics.AverageReturnMetric(),
            tf_metrics.AverageEpisodeLengthMetric(),
            tf_metrics.ChosenActionHistogram()
        ]

    def _init_replay_buffer(self):
        self.replay_buffer = TFUniformReplayBuffer(
            self.agent.collect_data_spec,
            batch_size=self.train_env.batch_size,
            max_length=self.replay_buf_size)
        # Dataset generates trajectories with shape [Bx2x...]
        logging.info(f'Initializing dataset with sample_batch_size={self.batch_size[0]}')
        dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=self.batch_size[0],
            num_steps=2).prefetch(3)
        self.dataset_iterator = iter(dataset)

    def _init_drivers(self, collect_policy):
        self.initial_collect_driver = DynamicStepDriver(
            self.train_env,
            RandomTFPolicy(self.train_env.time_step_spec(), self.train_env.action_spec()),
            observers=[self.replay_buffer.add_batch],
            num_steps=self.initial_collect_steps)
        self.collect_driver = DynamicStepDriver(
            self.train_env,
            collect_policy,
            observers=[self.replay_buffer.add_batch] + self.train_metrics,
            num_steps=1)

        self.initial_collect_driver.run = common.function(self.initial_collect_driver.run)
        self.collect_driver.run = common.function(self.collect_driver.run)

    def _init_checkpointers(self, dirs):
        global_step = tf.compat.v1.train.get_or_create_global_step()
        self.train_checkpointer = common.Checkpointer(
            ckpt_dir=dirs['chkpt'],
            agent=self.agent,
            global_step=global_step,
            metrics=metric_utils.MetricsGroup(self.train_metrics, 'train_metrics'))
        self.policy_checkpointer = common.Checkpointer(
            ckpt_dir=dirs['policy_chkpt'],
            policy=self.agent.policy,
            global_step=global_step)
        self.rb_checkpointer = common.Checkpointer(
            ckpt_dir=dirs['replay_buf_chkpt'],
            max_to_keep=1,
            replay_buffer=self.replay_buffer)
        self.train_checkpointer.initialize_or_restore()
        self.rb_checkpointer.initialize_or_restore()
        self.policy_checkpointer.initialize_or_restore()

    def _eval(self, eval_policy, global_step):
        logging.info(f'eval for {self.num_eval_episodes} episodes')
        metric_utils.eager_compute(
            self.eval_metrics,
            self.eval_env,
            eval_policy,
            num_episodes=self.num_eval_episodes,
            train_step=global_step,
            summary_writer=self.eval_summary_writer,
            summary_prefix='Metrics',
        )
        metric_utils.log_metrics(self.eval_metrics)

    def _do_one_train_step(self):
        experience, _ = next(self.dataset_iterator)

        if tf.compat.v1.train.get_or_create_global_step() == 0:
            tf.summary.trace_on()  # trace the first execution of train to obtain model graph for tensorboard

        loss = self.agent.train(experience)

        if tf.compat.v1.train.get_or_create_global_step() == 1:  # export graph
            with self.train_summary_writer.as_default():
                tf.summary.trace_export(name='graph export', step=0)
        return loss

    @gin.configurable
    def train(self):
        self.agent.initialize()
        eval_policy = self.agent.policy
        collect_policy = self.agent.collect_policy
        self.agent.train = common.function(self.agent.train)

        self._init_metrics()
        self._init_replay_buffer()
        self._init_drivers(collect_policy)
        self._init_checkpointers(self.dirs)

        global_step = tf.compat.v1.train.get_or_create_global_step()

        with tf.compat.v2.summary.record_if(global_step.numpy() % 1000 == 0):
            logging.info(f'Initializing replay for {self.initial_collect_steps} steps.')
            self.initial_collect_driver.run()
            self._eval(eval_policy, global_step)

            self._run_loop(collect_policy, eval_policy, global_step)

    def _run_loop(self, collect_policy, eval_policy, global_step):
        policy_state = collect_policy.get_initial_state(self.train_env.batch_size)
        timed_at_step = global_step.numpy()
        time_acc = 0
        time_step = None
        logging.info(f'Training for {self.num_iterations} iterations with batch size {self.batch_size}.')
        for _ in range(self.num_iterations):
            start_time = time.time()
            time_step, policy_state = self.collect_driver.run(
                time_step=time_step,
                policy_state=policy_state,
            )
            train_loss = self._do_one_train_step()
            global_step = tf.compat.v1.train.get_or_create_global_step()
            time_acc += time.time() - start_time

            self._maybe_save_checkpoints(global_step)

            if global_step.numpy() % self.log_interval == 0:  # log to cmd every log_interval steps
                logging.info('step = %d, loss = %f', global_step.numpy(), train_loss.loss)
                steps_per_sec = (global_step.numpy() - timed_at_step) / time_acc
                logging.info('%.3f steps/sec', steps_per_sec)
                tf.compat.v2.summary.scalar(name='global_steps_per_sec', data=steps_per_sec, step=global_step)
                timed_at_step = global_step.numpy()
                time_acc = 0

            for train_metric in self.train_metrics:  # update train metrics every step
                train_metric.tf_summaries(train_step=global_step, step_metrics=self.train_metrics[:2])

            for name, extra_loss in train_loss.extra._asdict().items():  # log _all_ available loss info to tensorboard
                with tf.name_scope('Losses/'):
                    tf.compat.v2.summary.scalar(name=name, data=extra_loss, step=global_step)

            if global_step.numpy() % self.eval_interval == 0:  # update eval metrics every eval_interval steps
                self._eval(eval_policy, global_step)

    def _maybe_save_checkpoints(self, global_step):
        if global_step.numpy() % self.checkpoint_interval == 0:  # checkpoints
            self.rb_checkpointer.save(global_step=global_step.numpy())
            self.train_checkpointer.save(global_step=global_step.numpy())
            self.policy_checkpointer.save(global_step=global_step.numpy())


@gin.configurable
class Td3TrainLoop(TrainLoop):

    def __init__(self, env_name: str):
        super(Td3TrainLoop, self).__init__(env_name=env_name)  # remainder of parameters are set via gin

    def _get_alg_name(self):
        return 'td3'

    def _get_agent(self):
        return TD3WrapAgent(self.train_env.time_step_spec(), self.train_env.action_spec())


LOOPS = {
    'td3': Td3TrainLoop
}


@gin.configurable
def get_train_loop(alg_name: str, env_name: str) -> TrainLoop:
    if alg_name not in LOOPS:
        raise ValueError(f'{alg_name} is not a known algorithm (no known TrainLoop subclass)')
    else:
        return LOOPS[alg_name](env_name=env_name)
