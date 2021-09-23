#!/usr/bin/env python
import json

import queue
import threading

import gzip

import numpy as np
import yaml

from datetime import datetime
from multiprocessing.managers import BaseManager
from pathlib import Path


class Datastore(object):
    __instance = None

    SUB_BASE_DIR = Path('datastore')  # datastore writes to subdir in root dir (rootdir/datastore/...)
    RAW_DATA_DIR = 'rawdata'
    EPISODE_FILE = 'episodes.csv'
    ENVIRONMENT_FILE = 'environment.csv'
    CONFIG_FILE = 'config.gin'

    EPISODE_HEADER = 'Episode, Steps, SrcPatternSeq, RatePatternSeq, ChangePattern, MeanRules, MeanPrec, MeanRecall, MeanFPR, MeanHHHDistSum, MeanReward, ' \
                     'ReturnDiscounted, ReturnUndiscounted '

    STEP_HEADER = 'Episode, Step, Reward, SrcPattern, RatePattern, ChangePattern, DiscountedReturnSoFar, UndiscountedReturnSoFar, Phi, MinPrefix, BlackSize, ' \
                  'Precision, EstPrecision, ' \
                  'Recall, EstRecall, FPR, EstFPR, HHHDistanceAvg, HHHDistanceSum, HHHDistanceMin, HHHDistanceMax '

    @staticmethod
    def get_timestamp():
        return datetime.now().strftime('%Y%m%d-%H%M%S')

    @staticmethod
    def _create_entry(rootdir, subdir):
        experiment_dir = rootdir / Datastore.SUB_BASE_DIR

        if subdir is not None:
            experiment_dir = experiment_dir / subdir

        experiment_dir.mkdir(parents=True, exist_ok=True)

        return experiment_dir

    @staticmethod
    def _add(file, line):
        file.write(line + '\n')

    @staticmethod
    def _print(line):
        print(line.replace(',', ''))

    @staticmethod
    def _format_episode(episode, split, source_seq, rate_seq, change_pattern, rules, precision, recall, fpr,
                        hhh_distance_sum, reward, return_discounted,
                        return_undiscounted):
        return '{:7d}, {:7d}, {}, {}, {}, {:8.3f}, {:7.5f}, {:9.5f}, {:9.5f}, {:9.5f}, {:9.5f}, {:9.5f}, {:9.5f}'.format(
            episode, split, source_seq, rate_seq, change_pattern,
            rules, precision, recall, fpr, hhh_distance_sum, reward, return_discounted,
            return_undiscounted)

    @staticmethod
    def _format_step(episode, split, reward, source_pattern, rate_pattern, change_pattern, discounted_return_so_far,
                     undiscounted_return_so_far, state):
        return '{:5d}, {:5.1f}, {:7.3f}, {}, {}, {}, {:7.3f},{:7.3f},{:7.5f}, {:3d}, {:5d}, {:5.3f}, {:5.3f}, {:5.3f}, ' \
               '{:5.3f}, {:5.3f}, {:5.3f}, {:9.7f}, {:9.7f}, {:7.5f}, {:7.5f}' \
            .format(
            episode, split, reward,
            source_pattern, rate_pattern, change_pattern,
            discounted_return_so_far, undiscounted_return_so_far, state.phi,
            state.min_prefix, state.blacklist_size, state.precision,
            state.estimated_precision, state.recall, state.estimated_recall,
            state.fpr, state.estimated_fpr,
            state.hhh_distance_avg, state.hhh_distance_sum, state.hhh_min,
            state.hhh_max)

    class NumpyWriter(object):
        """ An asynchronous writer to avoid blocking the main env thread
            while writing out numpy random-generated training data.
        """

        def __init__(self, rawdatadir, compresslevel=1, queuesize=24):
            self.rawdatadir = rawdatadir
            self.compresslevel = compresslevel
            self.queue = queue.Queue(queuesize)
            self.thread = threading.Thread(target=self.write_loop)

        def start(self):
            self.thread.start()

        def stop(self):
            # Send finishing signal to writer thread
            self.queue.put(None)
            self.queue.join()
            self.thread.join()

        def write_loop(self):
            while True:
                item = self.queue.get()
                if item is None:
                    self.queue.task_done()
                    break
                trace, suffix = item
                file = self.rawdatadir / 'trace_{}.npy.gz'.format(suffix)
                with gzip.GzipFile(file, 'w', compresslevel=self.compresslevel) as f:
                    np.save(f, trace)
                self.queue.task_done()

        def enqueue(self, item):
            self.queue.put(item)

    def __init__(self, rootdir: str, subdir: str, collect_raw=False):
        self.rootdir = rootdir
        self.subdir = subdir  # e.g. ppo-{timestamp}
        self.experiment_dir = Datastore._create_entry(Path(rootdir), subdir)
        self.rawdatadir = self.experiment_dir / Datastore.RAW_DATA_DIR
        self.collect_raw = collect_raw
        self.episode_file = None
        self.environment_file = None
        self.numpy_writer = None

    def __del__(self):
        if self.episode_file is not None:
            self.episode_file.close()
        if self.environment_file is not None:
            self.environment_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.numpy_writer is not None:
            self.numpy_writer.stop()

    def enter_scope(self, scopename):
        # Construct new object with updated subdir
        return Datastore(self.experiment_dir, scopename, self.collect_raw)

    def _initialize_basedir(self):
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

    def _initialize_episode_file(self):
        if self.episode_file is not None:
            return
        self._initialize_basedir()
        self.episode_file = (self.experiment_dir / Datastore.EPISODE_FILE).open('a')
        self.add_episode_header()

    def add_episode_header(self):
        Datastore._add(self.episode_file, Datastore.EPISODE_HEADER)

    def _initialize_environment_file(self):
        if self.environment_file is not None:
            return
        self._initialize_basedir()
        self.environment_file = (self.experiment_dir / Datastore.ENVIRONMENT_FILE).open('a')
        self.add_environment_header()

    def add_environment_header(self):
        self._add(self.environment_file, Datastore.STEP_HEADER)

    def add_episode(self, episode, split, source_seq, rate_seq, change_pattern, rules, precision, recall, fpr,
                    hhh_distance_sum, reward,
                    return_discounted,
                    return_undiscounted):
        self._initialize_episode_file()
        Datastore._add(self.episode_file, Datastore._format_episode(episode,
                                                                    split, source_seq, rate_seq, change_pattern, rules,
                                                                    precision, recall, fpr,
                                                                    hhh_distance_sum, reward, return_discounted,
                                                                    return_undiscounted))

    def add_step(self, episode, split, reward, source_pattern, rate_pattern, change_pattern,
                 discounted_return_so_far,
                 undiscounted_return_so_far, state):
        self._initialize_environment_file()
        Datastore._add(self.environment_file, Datastore._format_step(episode,
                                                                     split, reward, source_pattern, rate_pattern,
                                                                     change_pattern, discounted_return_so_far,
                                                                     undiscounted_return_so_far, state))

    def flush(self):
        self.environment_file.flush()
        self.episode_file.flush()

    def _initialize_rawdata(self):
        if self.numpy_writer is not None:
            return
        self.rawdatadir.mkdir(parents=True, exist_ok=True)
        self.numpy_writer = Datastore.NumpyWriter(self.rawdatadir)
        self.numpy_writer.start()

    def add_numpy_data(self, trace, suffix):
        if not self.collect_raw:
            return
        self._initialize_rawdata()
        self.numpy_writer.enqueue((trace, suffix))

    def add_blacklist(self, blacklist, suffix):
        if not self.collect_raw:
            return
        self._initialize_rawdata()
        file = self.rawdatadir / 'trace_{}.json'.format(suffix)
        with file.open('w') as f:
            f.write(json.dumps(blacklist))

    def commit_config(self, str):
        config_file_path = self.experiment_dir / Datastore.CONFIG_FILE

        with config_file_path.open('w') as f:
            f.write(str)


class DatastoreManager(BaseManager):
    pass


DatastoreManager.register('Datastore', Datastore,
                          exposed=['add_step', 'add_episode', 'add_numpy_data', 'add_blacklist', 'commit_config'])
