#!/usr/bin/env python

import numpy as np

from numpy.random import random, randint

from gyms.hhh.actionset import LargeDiscreteActionSet


class State(object):

    @staticmethod
    def get_onehot_action(state, action):
        a = np.zeros(state.actionset.shape)

        a[action if isinstance(action, int) else tuple(action)] = 1.0

        return a.flatten()

    GROUPS = {
        'base': {
            'obs': lambda s, a: np.array([s.trace_start, s.min_prefix,
                                          s.estimated_precision, s.estimated_recall,
                                          s.blacklist_size, s.episode_progress]),
            'lower': lambda s: np.array([0.0, 16.0, 0.0, 0.0, 0.0, 0.0]),
            'upper': lambda s: np.array([1.0, 32.0, 1.2, 1.2, 128.0, 1.0]),
            'init': lambda s: np.array([1.0, 32.0, 0.2 * random(),
                                        0.2 * random(), 1.0 * randint(16, 32), 0.0])
        },
        'actions': {
            'obs': get_onehot_action.__func__,
            'lower': lambda s: np.zeros(s.actionset.shape).flatten(),
            'upper': lambda s: np.ones(s.actionset.shape).flatten(),
            'init': lambda s: np.zeros(s.actionset.shape).flatten()
        },
        'cont_actions': {
            #			'obs'   : lambda s, a: np.array([a[0], a[1]]),
            #			'lower' : lambda s : np.array([0.01, 17]),
            #			'upper' : lambda s : np.array([0.25, 21]),
            #			'init'  : lambda s : np.array([0.12, 19])
            'obs': lambda s, a: np.array([a[0]]),
            'lower': lambda s: np.array([0.01]),
            'upper': lambda s: np.array([0.25]),
            'init': lambda s: np.array([0.12])
        },
        'distvol': {
            'obs': lambda s, a: np.array([s.hhh_distance_avg,
                                          s.hhh_distance_sum, s.hhh_distance_min, s.hhh_distance_max]),
            'lower': lambda s: np.zeros(4),
            'upper': lambda s: np.ones(4),
            'init': lambda s: np.zeros(4)
        },
        'minmax': {
            'obs': lambda s, a: np.array([s.hhh_min, s.hhh_max]),
            'lower': lambda s: np.zeros(2),
            'upper': lambda s: np.ones(2),
            'init': lambda s: np.zeros(2)
        },
        'fpr': {
            'obs': lambda s, a: np.array([s.estimated_fpr]),
            'lower': lambda s: np.array([0.0]),
            'upper': lambda s: np.array([1.2]),
            'init': lambda s: np.array([0.2 * random()])
        },
        'distvolstd': {
            'obs': lambda s, a: np.array([s.hhh_distance_std]),
            'lower': lambda s: np.zeros(1),
            'upper': lambda s: np.ones(1),
            'init': lambda s: np.zeros(1)
        },
        'bldist': {
            'obs': lambda s, a: np.array(s.bl_dist),
            'lower': lambda s: np.zeros(16),
            'upper': lambda s: np.ones(16),
            'init': lambda s: np.zeros(16) * 64.0
        }
    }

    def __init__(self, selection, actionset):
        self.selection = selection
        self.actionset = actionset
        self.trace_start = 1.0
        self.phi = 0.5
        self.min_prefix = 20
        self.total = 0  # total packets
        self.rewind()

    def rewind(self):
        # keep total items and current phi
        self.samples = 0  # sampled packets ("behind" upstream filter)
        self.blocked = 0  # blocked packets
        self.malicious = 0  # malicious packets
        self.malicious_passed = 0  # non-blocked malicious packets
        self.malicious_blocked = 0  # blocked malicious packets
        self.benign_passed = 0  # non-blocked benign packets
        self.estimated_malicious = 0
        self.estimated_malicious_blocked = 0  # sampled packets that turned out to be malicious
        self.estimated_benign = 0
        self.estimated_benign_blocked = 0
        self.blacklist_size = 0
        self.episode_progress = 0.0
        self.hhh_distance_mean = 0
        self.hhh_distance_avg = 0
        self.hhh_distance_sum = 0
        self.hhh_distance_min = 0
        self.hhh_distance_max = 0
        self.hhh_distance_std = 0
        self.hhh_min = 0
        self.hhh_max = 0
        self.bl_dist = np.zeros(16)

    def complete(self):
        self._estimate_packet_counters()
        self._calc_precision()
        self._calc_recall()
        self._calc_false_positive_rate()

    def _estimate_packet_counters(self):
        # upsampling
        if self.samples != 0:
            blocked_pkts_per_sample = self.blocked / self.samples
            self.estimated_malicious_blocked *= blocked_pkts_per_sample
            self.estimated_benign_blocked *= blocked_pkts_per_sample
        else:
            self.estimated_malicious_blocked = 0.0
            self.estimated_benign_blocked = 0.0
        self.estimated_malicious = (self.malicious_passed
                                    + self.estimated_malicious_blocked)
        self.estimated_benign = (self.benign_passed
                                 + self.estimated_benign_blocked)

    def _calc_precision(self):
        # calculation of precision
        try:
            self.precision = 1.0 * self.malicious_blocked / self.blocked
            self.estimated_precision = (self.estimated_malicious_blocked
                                        / self.blocked)
        except ZeroDivisionError:
            self.precision = 0.0
            self.estimated_precision = 0.0

    def _calc_false_positive_rate(self):
        # calculation of false positive rate
        benign = self.total - self.malicious
        benign_blocked = self.blocked - self.malicious_blocked
        estimated_benign = self.total - self.estimated_malicious
        estimated_benign_blocked = self.blocked - self.estimated_malicious_blocked
        try:
            self.fpr = 1.0 * benign_blocked / benign
            self.estimated_fpr = (estimated_benign_blocked / estimated_benign)
        except ZeroDivisionError:
            self.fpr = 0.0
            self.estimated_fpr = 0.0

    def _calc_recall(self):
        try:
            self.recall = 1.0 * self.malicious_blocked / self.malicious
            self.estimated_recall = (self.estimated_malicious_blocked
                                     / self.estimated_malicious)
        except ZeroDivisionError:
            self.recall = 0.0
            self.estimated_recall = 0.0

    def _build_state(self, what, args) -> np.ndarray:
        return np.concatenate([
            State.GROUPS[s][what](*args) for s in self.selection
        ])

    def get_features(self, action):
        return self._build_state('obs', [self, action])

    # used to set observation_space in env
    def get_lower_bounds(self):
        return self._build_state('lower', [self])

    # used to set observation_space in env
    def get_upper_bounds(self):
        return self._build_state('upper', [self])

    # used to reset env
    def get_initialization(self):
        return self._build_state('init', [self])
