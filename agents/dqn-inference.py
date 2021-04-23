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
import tensorflow as tf	# pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments.examples import masked_cartpole	# pylint: disable=unused-import
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
#from nets import dqn_q_network as q_network
from tf_agents.networks import q_rnn_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

import gyms.hhh
from lib.datastore import Datastore

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
	'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string('timestamp', None,
	'Restore from specified timestamp.')
flags.DEFINE_integer('num_iterations', 200000,
	'Total number train/eval iterations to perform.')
flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_param', None, 'Gin binding parameters.')

FLAGS = flags.FLAGS

@gin.configurable
def train_eval(
		root_dir,
		timestamp = None,
		env_name = 'HHHGym-v0',
		num_iterations = 100000,
		train_sequence_length = 1,
		# Params for QNetwork
		fc_layer_params = (100,50),
		# Params for QRnnNetwork
		batch_normalization = False,
		input_fc_layer_params = (200,200),
		lstm_size = (100,),
		output_fc_layer_params = (20,),
		# Params for collect
		initial_collect_steps = 1000,
		collect_steps_per_iteration = 1,
		epsilon_greedy = 0.1,
		replay_buffer_capacity = 100000,
		# Params for target update
		target_update_tau = 0.05,
		target_update_period = 5,
		# Params for train
		train_steps_per_iteration = 1,
		batch_size = 64,
		learning_rate = 1e-4,
		learning_rate_decay_rate = None,
		learning_rate_decay_step = None,
		n_step_update = 1,
		gamma = 0.99,
		reward_scale_factor = 1.0,
		gradient_clipping = None,
		use_tf_functions = True,
		# Params for eval
		num_eval_episodes = 10,
		eval_interval = 1000,
		# Params for checkpoints
		train_checkpoint_interval = 10000,
		policy_checkpoint_interval = 5000,
		rb_checkpoint_interval = 20000,
		# Params for summaries and logging
		log_interval = 1000,
		summary_interval = 1000,
		summaries_flush_secs = 10,
		debug_summaries = False,
		summarize_grads_and_vars = False,
		eval_metrics_callback = None,
		state_selection = ['base', 'actions', 'distvol', 'fpr', 'distvolstd', 'bldist'],
		actionset_selection = 'LargeDiscreteActionSet',
		trace_length = 50000):
	"""A simple train and eval for DQN."""
	if timestamp is None:
		timestamp = Datastore.get_timestamp()
	root_dir = os.path.expanduser(root_dir)
	train_dir = os.path.join(root_dir, 'train')
	eval_dir = os.path.join(root_dir, 'eval')

	eval_summary_writer = tf.compat.v2.summary.create_file_writer(
		eval_dir, flush_millis = summaries_flush_secs * 1000)

	train_metrics = [
		tf_metrics.NumberOfEpisodes(),
		tf_metrics.EnvironmentSteps(),
		tf_metrics.AverageReturnMetric(),
		tf_metrics.AverageEpisodeLengthMetric(),
		tf_metrics.ChosenActionHistogram()
	]
	eval_metrics = [
		tf_metrics.AverageReturnMetric(buffer_size = num_eval_episodes),
		tf_metrics.AverageEpisodeLengthMetric(buffer_size = num_eval_episodes),
	]

	global_step = tf.compat.v1.train.get_or_create_global_step()

	with eval_summary_writer.as_default():
		ds_eval = Datastore('eval', timestamp)

		gym_kwargs = {
			'state_selection' : state_selection,
			'actionset_selection' : actionset_selection,
			'trace_length' : trace_length
		}

		tf_env = tf_py_environment.TFPyEnvironment(suite_gym.load(
			env_name, gym_kwargs = {'data_store' : ds_eval, **gym_kwargs},
			spec_dtype_map = {gym.spaces.Discrete : np.int32}))

		if train_sequence_length != 1 and n_step_update != 1:
			raise NotImplementedError(
				'train_eval does not currently support n-step updates with stateful '
				'networks (i.e., RNNs)')

		if train_sequence_length > 1:
			logging.info('Using QRnnNetwork')
			q_net = q_rnn_network.QRnnNetwork(
				tf_env.observation_spec(),
				tf_env.action_spec(),
				input_fc_layer_params = input_fc_layer_params,
				lstm_size = lstm_size,
				output_fc_layer_params = output_fc_layer_params)
		else:
			logging.info('Using QNetwork')
			q_net = q_network.QNetwork(
				tf_env.observation_spec(),
				tf_env.action_spec(),
#				batch_normalization=batch_normalization,
				fc_layer_params = fc_layer_params)
			train_sequence_length = n_step_update

		tf_agent = dqn_agent.DqnAgent(
			tf_env.time_step_spec(),
			tf_env.action_spec(),
			q_network = q_net,
			epsilon_greedy = epsilon_greedy,
			n_step_update = n_step_update,
			target_update_tau = target_update_tau,
			target_update_period = target_update_period,
			optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate),
			td_errors_loss_fn = common.element_wise_squared_loss,
			gamma = gamma,
			reward_scale_factor = reward_scale_factor,
			gradient_clipping = gradient_clipping,
			debug_summaries = debug_summaries,
			summarize_grads_and_vars = summarize_grads_and_vars,
			train_step_counter = global_step)
		tf_agent.initialize()

		eval_policy = tf_agent.policy

		train_checkpointer = common.Checkpointer(
			ckpt_dir = train_dir,
			agent = tf_agent,
			global_step = global_step,
			metrics = metric_utils.MetricsGroup(train_metrics, 'train_metrics'))

		train_checkpointer.initialize_or_restore()

		results = metric_utils.eager_compute(
			eval_metrics,
			tf_env,
			eval_policy,
			num_episodes = num_eval_episodes,
			train_step = global_step,
			summary_writer = eval_summary_writer,
			summary_prefix = 'Metrics',
			use_function = use_tf_functions
		)

		if eval_metrics_callback is not None:
			eval_metrics_callback(results, global_step.numpy())
		metric_utils.log_metrics(eval_metrics)


def main(_):
	logging.set_verbosity(logging.INFO)
	tf.compat.v1.enable_v2_behavior()
	tf.get_logger().setLevel('ERROR')
	gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
	train_eval(FLAGS.root_dir, FLAGS.timestamp, num_iterations = FLAGS.num_iterations)

if __name__ == '__main__':
	try:
		flags.mark_flag_as_required('root_dir')
		app.run(main)
	except KeyboardInterrupt:
		pass
