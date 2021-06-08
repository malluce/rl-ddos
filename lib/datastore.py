#!/usr/bin/env python

import yaml

from datetime import datetime
from multiprocessing.managers import BaseManager
from pathlib import Path

class Datastore(object):

	__instance = None

	BASE_DIR = Path('data/stats')
	TENSORBOARD_DIR = Path('data/tensorboard')
	EPISODE_FILE = 'episodes.csv'
	ENVIRONMENT_FILE = 'environment.csv'
	CONFIG_FILE = 'config.gin'

	EPISODE_HEADER = 'Episode, Split, MeanRules, MeanPrec, MeanRecall, MeanFPR, MeanHHHDistSum, MeanReward'
#	STEP_HEADER = 'Episode, Split, Reward, Phi, MinPrefix, BlackSize, Precision, EstPrecision, Recall, EstRecall, FPR, EstFPR, HHHDistanceAvg, HHHDistanceSum, HHHDistanceMin, HHHDistanceMax'
	STEP_HEADER = 'Episode, Step, Reward, Phi, MinPrefix, BlackSize, Precision, EstPrecision, Recall, EstRecall, FPR, EstFPR, HHHDistanceAvg, HHHDistanceSum, HHHDistanceMin, HHHDistanceMax'

	@staticmethod
	def get_timestamp():
		return datetime.now().strftime('%Y%m%d-%H%M%S')

	@staticmethod
	def _create_entry(subdir, timestamp):
		experiment_dir = Datastore.BASE_DIR/timestamp

		if subdir is not None:
			experiment_dir = experiment_dir/subdir

		experiment_dir.mkdir(parents = True, exist_ok = True)

		return experiment_dir

	@staticmethod
	def _add(file, line):
		file.write(line + '\n')

	@staticmethod
	def _print(line):
		print(line.replace(',', ''))

	@staticmethod
	def _format_episode(episode, split, rules, precision, recall, fpr, hhh_distance_sum, reward):
		return '{:7d}, {:7d}, {:8.3f}, {:7.5f}, {:9.5f}, {:9.5f}, {:9.5f}, {:9.5f}'.format(
			episode, split, rules, precision, recall, fpr, hhh_distance_sum, reward)

	@staticmethod
	def _format_step(episode, split, reward, state):
		return '{:5d}, {:5.1f}, {:7.3f}, {:7.5f}, {:3d}, {:5d}, {:5.3f}, {:5.3f}, {:5.3f}, {:5.3f}, {:5.3f}, {:5.3f}, {:9.7f}, {:9.7f}, {:7.5f}, {:7.5f}'.format(
			episode, split, reward, state.phi,
			state.min_prefix, state.blacklist_size, state.precision,
			state.estimated_precision, state.recall, state.estimated_recall,
			state.fpr, state.estimated_fpr,
			state.hhh_distance_avg, state.hhh_distance_sum, state.hhh_min,
			state.hhh_max)

	def __init__(self, subdir = None, timestamp = None):
		if timestamp is None:
			timestamp = Datastore.get_timestamp()

		self.experiment_dir = Datastore._create_entry(subdir, timestamp)

		episode_file_path = self.experiment_dir/Datastore.EPISODE_FILE
		environment_file_path = self.experiment_dir/Datastore.ENVIRONMENT_FILE

		self.episode_file = episode_file_path.open('a')
		self.environment_file = environment_file_path.open('a')

		self.config = {}

		Datastore._add(self.episode_file, Datastore.EPISODE_HEADER)
		Datastore._add(self.environment_file, Datastore.STEP_HEADER)

	def __del__(self):
		self.episode_file.close()
		self.environment_file.close()

	def add_episode_header(self):
		Datastore._add(self.episode_file, Datastore.EPISODE_HEADER)

	def add_step_header(self):
		Datastore._add(self.environment_file, Datastore.STEP_HEADER)

	def add_episode(self, episode, split, rules, precision, recall, fpr, hhh_distance_sum, reward):
		Datastore._add(self.episode_file, Datastore._format_episode(episode,
			split, rules, precision, recall, fpr, hhh_distance_sum, reward))

	def add_step(self, episode, split, reward, state):
		Datastore._add(self.environment_file, Datastore._format_step(episode,
			split, reward, state))

	def add_step_line(self, line):
		Datastore._add(self.environment_file, line)

	def print_episode_header(self):
		Datastore._print(Datastore.EPISODE_HEADER)

	def print_step_header(self):
		Datastore._print(Datastore.STEP_HEADER)

	def print_episode(self, episode, split, rules, precision, recall, epsilon, reward):
		Datastore._print(Datastore._format_episode(episode, split, rules,
			precision, recall, epsilon, reward))

	def print_step(self, episode, trace, state):
		Datastore._print(Datastore._format_step(episode, trace, state))

	def set_config(self, parameter, value):
		self.config[parameter] = value

	def commit_config(self, str):
		config_file_path = self.experiment_dir/Datastore.CONFIG_FILE

		with config_file_path.open('w') as f:
			f.write(str)


class DatastoreManager(BaseManager):
	pass

DatastoreManager.register('Datastore', Datastore,
	exposed = ['add_step', 'add_episode', 'set_config', 'commit_config'])
