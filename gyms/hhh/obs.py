#!/usr/bin/env python

from abc import ABC, abstractmethod

import gin
import numpy as np


class Observation(ABC):
    @abstractmethod
    def get_observation(self, observed_object):
        pass

    @abstractmethod
    def get_lower_bound(self):
        pass

    @abstractmethod
    def get_upper_bound(self):
        pass


@gin.register
class FalsePositiveRate(Observation):

    def get_observation(self, state):
        return np.array([state.estimated_fpr])

    def get_lower_bound(self):
        return np.array([0.0])

    def get_upper_bound(self):
        return np.array([1.2])


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


@gin.register
class MinMaxBlockedAddress(Observation):

    def get_observation(self, state):
        return np.array([state.hhh_min, state.hhh_max])

    def get_lower_bound(self):
        return np.zeros(2)

    def get_upper_bound(self):
        return np.ones(2)


@gin.register
class DistVolStd(Observation):

    def get_observation(self, state):
        return np.array([state.hhh_distance_std])

    def get_lower_bound(self):
        return np.zeros(1)

    def get_upper_bound(self):
        return np.ones(1)


@gin.register
class BaseObservations(Observation):

    def get_observation(self, state):
        return np.array([
            # state.estimated_precision, state.estimated_recall,
            state.blacklist_size
        ])

    def get_lower_bound(self):
        return np.array([0.0])

    def get_upper_bound(self):
        inf = np.finfo(np.float32).max
        return np.array([inf])


@gin.configurable
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

# TODO image observation? those are different from vector observations, since it requires modified NN...
