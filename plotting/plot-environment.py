#!/usr/bin/env python

import argparse
import glob
import json
import os

from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib import ticker
from matplotlib.gridspec import GridSpec
from statistics import mean
from threading import Thread
from time import sleep
from _tkinter import TclError

terminated = False

def cmdline():
	argp = argparse.ArgumentParser(description = 'Plotter')
	argp.add_argument(metavar = 'data dir', type = str, dest = 'datadir',
		help = 'directory containing the data files')
	argp.add_argument(metavar = 'image', type = str, dest = 'image',
		nargs = '?', default = None, help = 'output image')
	argp.add_argument('--max-episodes', default = None,
		type = int, help = 'Maximum number of episodes to plot')
	argp.add_argument('--episode-length', default = 10,
		type = int, help = 'Steps per episode')
	argp.add_argument('--dpi', default = 800,
		type = int, help = 'Image DPI')

	return argp.parse_args()


class DataPoint(object):

	@staticmethod
	def from_line(line):
		values = [_.strip() for _ in line.split(',')]

		p = DataPoint()
		p.episode = int(values[0])
		p.split = int(float(values[1]))
		p.reward = float(values[2])
		p.phi = float(values[3])
		p.min_prefix = int(values[4])
		p.blacklist_size = int(values[5])
		p.precision = float(values[6])
		p.estimated_precision = float(values[7])
		p.recall = float(values[8])
		p.estimated_recall = float(values[9])
		p.fpr = float(values[10])
		p.estimated_fpr = float(values[11])
		p.hhh_distance_avg = float(values[12])
		p.hhh_distance_sum = float(values[13])
		p.hhh_min = float(values[14])
		p.hhh_max = float(values[15])

		return p

	# FIXME: Add missing fields
	def __init__(self, episode = 0, reward = 0, split = 0, phi = 0,
			min_prefix = 0, blacklist_size = 0, precision = 0,
			estimated_precision = 0, recall = 0, estimated_recall = 0,
			fpr = 0, estimated_fpr = 0):
		self.episode = episode 
		self.split = split 
		self.reward = reward 
		self.phi = phi 
		self.min_prefix = min_prefix 
		self.blacklist_size = blacklist_size 
		self.precision = precision 
		self.estimated_precision = estimated_precision 
		self.recall = recall 
		self.estimated_recall = estimated_recall 
		self.fpr = fpr 
		self.estimated_fpr = estimated_fpr 

	# FIXME: Add missing fields
	def __str__(self):
		return ', '.join(str(_) for _ in [self.episode, self.split, self.reward,
			self.phi, self.min_prefix, self.blacklist_size, self.precision,
			self.recall, self.estimated_precision, self.estimated_recall])


class Projection(object):

	def __init__(self, data, member, length = None):
		self.data = data
		self.member = member
		self.length = len(data) if length is None else length

	def __len__(self):
		return self.length

	def __getitem__(self, index):
		return getattr(self.data[index], self.member)


class Subplot(object):

	def __init__(self, fig, grid_entry, field, xlabel = None,
			ylabel = None, ylim = None, yscale = None, yticks = None,
			yformatter = None , color = 'b'):
		self.plot = fig.add_subplot(grid_entry)
		self.field = field
		self.xlabel = xlabel
		self.ylabel = ylabel
		self.ylim = ylim
		self.yscale = yscale
		self.yticks = yticks
		self.yformatter = yformatter
		self.color = color

	def draw(self, data, length):
		self.plot.cla()

		self.plot.spines['top'].set_visible(False)
		self.plot.spines['right'].set_visible(False)
		self.plot.get_xaxis().tick_bottom()
		self.plot.get_yaxis().tick_left()
		self.plot.grid(True, axis = 'y', which = 'major', linestyle = '--')

		if self.xlabel is not None:
			self.plot.set_xlabel(self.xlabel)

		if self.ylabel is not None:
			self.plot.set_ylabel(self.ylabel)

		if self.ylim is not None:
			self.plot.set_ylim(*self.ylim)

		if self.yscale is not None:
			self.plot.set_yscale(self.yscale)

		if self.yticks is not None:
			self.plot.set_yticks(self.yticks)

		if self.yformatter is not None:
			self.plot.get_yaxis().set_major_formatter(self.yformatter)

		self.plot.scatter(range(length), s = 1, marker = '.', c = self.color,
			y = Projection(data, self.field, length))


def create_subplots(title):

	def on_close(ev):
		global terminated

		terminated = True

	fig = plt.figure(figsize = (18, 9))
	fig.canvas.mpl_connect('close_event', on_close)

	plt.gcf().canvas.set_window_title(title)

	gs = GridSpec(6, 2, height_ratios = [1] * 6)
	gs.update(wspace = 0.2, hspace = 0.8)

	subplots = [
		Subplot(fig, gs[0, 0], 'precision', ylabel = 'Precision',
			ylim = (0, 1.09), yticks = [0, 0.5, 1.0], color = 'g'),
		Subplot(fig, gs[1, 0], 'recall', ylabel = 'Recall',
			ylim = (0, 1.09), yticks = [0, 0.5, 1.0], color = 'g'),
		Subplot(fig, gs[2, 0], 'fpr', ylabel = 'FPR',
			ylim = (1e-3, 1e0), yticks = [1e-3, 1e-2, 1e-1, 1e0],
			yscale = 'log', color = 'g'),
		Subplot(fig, gs[3, 0], 'rules_by_split2', ylabel = 'Rules 2',
			ylim = (1, 40), yticks = [1, 8, 32], yscale = 'log',
			yformatter = ticker.ScalarFormatter()),
		Subplot(fig, gs[4, 0], 'rules_by_split4', ylabel = 'Rules 8',
			ylim = (1, 40), yticks = [1, 8, 32], yscale = 'log',
			yformatter = ticker.ScalarFormatter()),
		Subplot(fig, gs[5, 0], 'rules_by_split6', ylabel = 'Rules 32',
			ylim = (1, 40), yticks = [1, 8, 32], yscale = 'log',
			yformatter = ticker.ScalarFormatter()),
		Subplot(fig, gs[0, 1], 'reward', ylabel = 'Reward',
			ylim = (0, 1.08), yticks = [0, 0.5, 1.0], color = 'r'),
		Subplot(fig, gs[1, 1], 'ewma_reward', ylabel = 'EwmaReward',
			ylim = (0, 1.08), yticks = [0, 0.5, 1.0], color = 'r'),
		Subplot(fig, gs[3, 1], 'hhh_distance_sum_by_split2', ylabel = 'DistSum 2',
			ylim = (0, 1)),
		Subplot(fig, gs[4, 1], 'hhh_distance_sum_by_split4', ylabel = 'DistSum 8',
			ylim = (0, 1)),
		Subplot(fig, gs[5, 1], 'hhh_distance_sum_by_split6', ylabel = 'DistSum 32',
			ylim = (0, 1)),
	]

	return subplots


class DataWatchdog(Thread):

	@staticmethod
	def infinity():
		while True:
			yield

	def __init__(self, path, data, episode_length, max_episodes):
		super(DataWatchdog, self).__init__()
		self.path = path
		self.data = data
		self.episode_length = episode_length
		self.ewma_reward = 0

		if max_episodes is None:
			self.episodes = DataWatchdog.infinity()
		else:
			self.episodes = range(0, max_episodes)

	def ewma(self, x, y, alpha = 0.4):
		return (1 - alpha) * x + alpha * y

	def aggregate(self, episode):
		rules_by_split = [[] for _ in range(7)]
		hhh_distance_avg_by_split = [[] for _ in range(7)]
		hhh_distance_sum_by_split = [[] for _ in range(7)]
#		hhh_distance_std_by_split = [[] for _ in range(7)]

		for _ in episode:
			rules_by_split[_.split].append(_.blacklist_size)
			hhh_distance_avg_by_split[_.split].append(_.hhh_distance_avg)
			hhh_distance_sum_by_split[_.split].append(_.hhh_distance_sum)
#			hhh_distance_std_by_split[_.split].append(_.hhh_distance_std)

		p = DataPoint()
		p.precision = mean(Projection(episode, 'precision'))
		p.recall = mean(Projection(episode, 'recall'))
		p.fpr = max(1e-5, mean(Projection(episode, 'fpr')))
		p.reward = sum(Projection(episode, 'reward')) / self.episode_length

		self.ewma_reward = self.ewma(self.ewma_reward, p.reward)

		p.ewma_reward = max(1e-3, self.ewma_reward)

		for s in range(len(rules_by_split)):
			if len(rules_by_split[s]) == 0:
				m = 0
			else:
				m = mean(rules_by_split[s])

			setattr(p, 'rules_by_split{}'.format(s), m)

			if len(hhh_distance_avg_by_split[s]) == 0:
				m = -10000
			else:
				m = mean(hhh_distance_avg_by_split[s])

			setattr(p, 'hhh_distance_avg_by_split{}'.format(s), m)

			if len(hhh_distance_sum_by_split[s]) == 0:
				m = -10000
			else:
				m = mean(hhh_distance_sum_by_split[s])

			setattr(p, 'hhh_distance_sum_by_split{}'.format(s), m)

#			if len(hhh_distance_std_by_split[s]) == 0:
#				m = -10000
#			else:
#				m = mean(hhh_distance_std_by_split[s])
#
#			setattr(p, 'hhh_distance_std_by_split{}'.format(s), m)

		return p

	def read_episode(self, file, continuous = True):
		episode = []

		while len(episode) != self.episode_length:
			if terminated:
				return False

			line = file.readline()

			if not line:
				if continuous:
					sleep(1)
					continue
				else:
					return False

			try:
				episode.append(DataPoint.from_line(line))
			except IndexError:
				pass
			except ValueError:
				pass

		self.data.append(self.aggregate(episode))

		return True

	def read_all(self):
		with open(self.path, 'r') as f:
			f.readline()

			for _ in self.episodes:
				if not self.read_episode(f, False):
					break

	def run(self):
		with open(self.path, 'r') as f:
			f.readline()

			for _ in self.episodes:
				if terminated:
					break

				self.read_episode(f)

def plot_animated(subplots, data):

	while not terminated:
		try:
			length = len(data)

			plt.ion()
			plt.show()

			for subplot in subplots:
				subplot.draw(data, length)

		except ValueError:
			pass

		try:
			plt.draw()
			plt.pause(5.0)
		except TclError:
			return

def plot_image(subplots, data, image, dpi):
	length = len(data)

	for subplot in subplots:
		subplot.draw(data, length)

	plt.savefig(image, bbox_inches = 'tight', dpi = dpi)

def main():
	args = cmdline()

	path   = args.datadir.strip().rstrip('/')
	title  = path.split('/')[-1]

	data = []

	subplots = create_subplots(title)

	if args.image is not None:
		DataWatchdog(path, data, args.episode_length, args.max_episodes).read_all()
		plot_image(subplots, data, args.image, args.dpi)
	else:
		wd = DataWatchdog(path, data, args.episode_length, args.max_episodes)
		wd.start()

		plot_animated(subplots, data)

if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		pass
