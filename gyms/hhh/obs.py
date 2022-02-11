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
class TrafficSituation(Observation):

    def get_observation(self, state):
        try:
            malicious_fraction = state.estimated_malicious / state.packets_per_step
        except ZeroDivisionError:
            malicious_fraction = 0

        return np.array([
            state.packets_per_step, malicious_fraction
        ])

    def get_lower_bound(self):
        return np.array([0.0, 0.0])

    def get_upper_bound(self):
        inf = np.finfo(np.float32).max
        return np.array([inf, 2.0])
