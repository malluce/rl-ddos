#!/usr/bin/env python

from abc import ABC, abstractmethod


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

    @abstractmethod
    def get_initialization(self):
        pass
