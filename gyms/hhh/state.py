#!/usr/bin/env python
import gin
import numpy as np

from numpy.random import default_rng, random, randint

from .obs import Observation
from .util import maybe_cast_to_arr


@gin.register
class FalsePositiveRate(Observation):

    def get_observation(self, state):
        return np.array([state.estimated_fpr])

    def get_lower_bound(self):
        return np.array([0.0])

    def get_upper_bound(self):
        return np.array([1.2])

    def get_initialization(self):
        return np.array([0.2 * random()])


@gin.register
class DistVol(Observation):

    def get_observation(self, state):
        return np.array([state.hhh_distance_avg,
                         state.hhh_distance_sum,
                         state.hhh_distance_min,
                         state.hhh_distance_max])

    def get_lower_bound(self):
        return np.zeros(4)

    def get_upper_bound(self):
        return np.ones(4)

    def get_initialization(self):
        return np.zeros(4)


@gin.register
class MinMaxBlockedAddress(Observation):

    def get_observation(self, state):
        return np.array([state.hhh_min, state.hhh_max])

    def get_lower_bound(self):
        return np.zeros(2)

    def get_upper_bound(self):
        return np.ones(2)

    def get_initialization(self):
        return np.zeros(2)


@gin.register
class DistVolStd(Observation):

    def get_observation(self, state):
        return np.array([state.hhh_distance_std])

    def get_lower_bound(self):
        return np.zeros(1)

    def get_upper_bound(self):
        return np.ones(1)

    def get_initialization(self):
        return np.zeros(1)


@gin.register
class BaseObservations(Observation):

    def get_observation(self, state):
        return np.array([state.trace_start,
                         # state.estimated_precision, state.estimated_recall,
                         state.blacklist_size
                         ])

    def get_lower_bound(self):
        # return np.array([0.0, 16.0, 0.0, 0.0, 0.0])
        return np.array([0.0, 0.0])

    def get_upper_bound(self):
        # return np.array([1.0, 32.0, 1.2, 1.2, 128.0])
        return np.array([1.0, 128.0])

    def get_initialization(self):
        # return np.array([1.0, 32.0, 0.2 * random(),
        #                 0.2 * random(), 1.0 * randint(16, 32)])
        return np.array([1.0, 1.0 * randint(16, 32)])


@gin.register
class HafnerObservations(Observation):
    TCAM_CAP = 0

    def __init__(self, tcam_cap=TCAM_CAP):
        self.tcam_cap = tcam_cap

    def get_observation(self, state):
        # all metrics per step
        return np.array([
            state.packets_per_step, state.blocked,  # n, n_blocked
            state.estimated_malicious_blocked, state.estimated_benign_blocked,  # {m,b}_blocked^estimated
            state.estimated_malicious, state.estimated_benign,  # {m,b}^estimated
            state.estimated_precision, state.estimated_recall,  # p, r
            state.blacklist_size / self.tcam_cap,  # ratio of used TCAM capacity
            state.blacklist_coverage  # ratio of address space covered by blacklist rules
        ])

    def get_lower_bound(self):
        return np.zeros(10)

    def get_upper_bound(self):
        inf = np.finfo(np.float32).max
        return np.array([inf, inf, inf, inf, inf, inf, 1.0, 1.0, 1.0, 1.0])

    def get_initialization(self):
        raise NotImplementedError('Not expected to use initialization for HafnerObservation!')


# TODO image observation? those are different from vector observations, since it requires modified NN...

class State(object):

    def __init__(self, selection):
        self.selection: [Observation] = selection
        self.trace_start = 1.0
        self.phi = 0.5
        self.min_prefix = -1
        self.thresh = -1
        self.total = 0  # total packets

        # Hafner Observation: for address space covered by rules
        self.lowest_ip = 2 ** 32
        self.highest_ip = 0

        self.rewind()

    def rewind(self):
        # keep total items and current phi
        self.packets_per_step = 0  # packets per step
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
        self.hhh_distance_mean = 0
        self.hhh_distance_avg = 0
        self.hhh_distance_sum = 0
        self.hhh_distance_min = 0
        self.hhh_distance_max = 0
        self.hhh_distance_std = 0
        self.hhh_min = 0
        self.hhh_max = 0
        self.bl_dist = np.zeros(16)
        self.image = None  # 2-channel image of last steps traffic and filter
        self.hhh_image = None  # 1-channel image of current traffic
        self.blacklist_coverage = 0  # Hafner observation: fraction of blocked address space

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
        except ZeroDivisionError:
            # if no packet was blocked, but there was no mal packet, then precision=1
            self.precision = 1.0 if self.malicious == 0 else 0.0

        try:
            self.estimated_precision = (self.estimated_malicious_blocked / self.blocked)
        except ZeroDivisionError:
            self.estimated_precision = 1.0 if int(self.estimated_malicious) == 0 else 0.0

    def _calc_false_positive_rate(self):
        # calculation of false positive rate
        benign = self.packets_per_step - self.malicious
        benign_blocked = self.blocked - self.malicious_blocked
        estimated_benign = self.packets_per_step - self.estimated_malicious
        estimated_benign_blocked = self.blocked - self.estimated_malicious_blocked
        try:
            self.fpr = 1.0 * benign_blocked / benign
        except ZeroDivisionError:
            self.fpr = 0.0

        try:
            self.estimated_fpr = (estimated_benign_blocked / estimated_benign)
        except ZeroDivisionError:
            self.estimated_fpr = 0.0

    def _calc_recall(self):
        try:
            self.recall = 1.0 * self.malicious_blocked / self.malicious
        except ZeroDivisionError:
            self.recall = 1.0  # if no malicious packets -> captured all! -> recall=1

        try:
            self.estimated_recall = (self.estimated_malicious_blocked / self.estimated_malicious)
        except ZeroDivisionError:
            self.estimated_recall = 1.0

    def get_features(self) -> np.ndarray:
        return np.concatenate([
            maybe_cast_to_arr(s.get_observation(self)) for s in self.selection
        ])

    # used to set observation_space in env
    def get_lower_bounds(self) -> np.ndarray:
        return np.concatenate([
            maybe_cast_to_arr(s.get_lower_bound()) for s in self.selection
        ])

    # used to set observation_space in env
    def get_upper_bounds(self) -> np.ndarray:
        return np.concatenate([
            maybe_cast_to_arr(s.get_upper_bound()) for s in self.selection
        ])

    # used to reset env
    def get_initialization(self) -> np.ndarray:
        return np.concatenate([
            maybe_cast_to_arr(s.get_initialization()) for s in self.selection
        ])
