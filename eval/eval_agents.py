from abc import ABC, abstractmethod
from typing import List

import numpy as np
from gym import Space

from gyms.hhh.action import ActionSpace


class EvalAgent(ABC):
    @abstractmethod
    def action(self, observation):
        pass


class RandomEvalAgent(EvalAgent):

    def __init__(self, action_space: Space, action_set: ActionSpace):
        self.action_space = action_space
        self.action_set = action_set

    def action(self, observation):
        return self.action_space.sample()


class FixedEvalAgent(EvalAgent):

    def __init__(self, phi: float, min_prefix_length: int, thresh: float = None, action_space=None):
        self.phi = phi
        self.min_prefix_length = min_prefix_length
        self.thresh = thresh
        self.action_space = action_space

    def action(self, observation):
        action = [x for x in filter(lambda y: y is not None, [self.phi, self.thresh, self.min_prefix_length])]
        return self.action_space.inverse_resolve(np.array(action))
