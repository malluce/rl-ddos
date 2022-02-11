from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
from gym import Space

from gyms.hhh.action import ActionSpace


class BasicAgent(ABC):
    @abstractmethod
    def action(self, observation):
        pass


class RandomAgent(BasicAgent):
    """Randomly selects actions from the action space."""

    def __init__(self, action_space: Space, action_set: ActionSpace):
        self.action_space = action_space
        self.action_set = action_set

    def action(self, observation):
        return self.action_space.sample()


class FixedAgent(BasicAgent):
    """Applies the same action at every adaptation step."""

    def __init__(self, phi: float, min_prefix_length: int, thresh: float = None, action_space=None):
        self.phi = phi
        self.min_prefix_length = min_prefix_length
        self.thresh = thresh
        self.action_space = action_space

    def action(self, observation):
        action = [x for x in filter(lambda y: y is not None, [self.phi, self.thresh, self.min_prefix_length])]
        return self.action_space.inverse_resolve(np.array(action))


class FixedSequenceAgent(BasicAgent):
    """Applies a fixed sequence of actions (sequences indicate the parameter values for each adaptation step)."""

    def __init__(self, episode_length, phis: List[float], min_prefix_lengths: Optional[List[int]],
                 pthreshs: Optional[List[float]] = None,
                 action_space=None):
        assert phis is not None
        if min_prefix_lengths is None:
            # action space phi pthresh
            assert pthreshs is not None
            assert len(phis) == len(pthreshs)
        elif pthreshs is None:
            # action space phi L
            assert min_prefix_lengths is not None
            assert len(phis) == len(min_prefix_lengths)

        self.phis = phis
        self.min_prefix_lengths = min_prefix_lengths
        self.pthreshs = pthreshs
        self.action_space = action_space
        self.action_counter = 0
        self.episode_len = episode_length
        self.not_none_action_lists = [x for x in
                                      filter(lambda y: y is not None,
                                             [self.phis, self.pthreshs, self.min_prefix_lengths])]

    def action(self, observation):
        action = [x[self.action_counter] for x in self.not_none_action_lists]

        self.action_counter = (self.action_counter + 1) % self.episode_len

        return self.action_space.inverse_resolve(np.array(action))
