from abc import ABC, abstractmethod
from typing import List

import numpy as np
from gym import Space

from gyms.hhh.actionset import ActionSet


class EvalAgent(ABC):
    @abstractmethod
    def action(self, observation):
        pass


class RandomEvalAgent(EvalAgent):

    def __init__(self, action_space: Space, action_set: ActionSet):
        self.action_space = action_space
        self.action_set = action_set

    def action(self, observation):
        return self.action_space.sample()


class FixedEvalAgent(EvalAgent):

    def __init__(self, phi: float, min_prefix_length: int, thresh: float = None, actionset=None):
        self.phi = phi
        self.min_prefix_length = min_prefix_length
        self.thresh = thresh
        self.actionset = actionset

    def action(self, observation):

        # action() has to return unresolved actions, but API should take resolved actions, hence require inverse resolve
        # e.g. phi=0.002 will be "inverse resolved" to approx 0
        def inverse_resolve(x):
            lb = self.actionset.get_lower_bound()
            ub = self.actionset.get_upper_bound()

            return 1 / (ub - (ub + lb) / 2) * x - ((ub + lb) / 2) / (ub - (ub + lb) / 2)

        if self.thresh is not None:
            if self.min_prefix_length is not None:
                return inverse_resolve(np.array([self.phi, self.thresh, self.min_prefix_length]))
            else:
                return inverse_resolve(np.array([self.phi, self.thresh]))
        else:
            return inverse_resolve(np.array([self.phi, self.min_prefix_length]))
