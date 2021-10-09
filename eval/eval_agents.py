from abc import ABC, abstractmethod
from typing import List

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

    def __init__(self, phi: float, min_prefix_length: int, thresh: float = None):
        self.phi = phi
        self.min_prefix_length = min_prefix_length
        self.thresh = thresh

    def action(self, observation):
        if self.thresh is not None:
            return self.phi, self.thresh, self.min_prefix_length
        else:
            return self.phi, self.min_prefix_length


class ProgressDependentEvalAgent(EvalAgent):

    def __init__(self, phis: List[float], min_prefix_lengths: List[int], step_bounds: List[int]):
        self.phis = phis
        self.ls = min_prefix_lengths
        self.step_bounds = step_bounds
        self.steps = 0
        self.bounds_progress = 0
        assert len(phis) == len(min_prefix_lengths) == len(step_bounds) + 1

    def action(self, obs, step, new_episode):
        if new_episode:
            self.steps = 0
            self.bounds_progress = 0

        try:
            if self.step_bounds[self.bounds_progress] <= self.steps:
                self.bounds_progress += 1
        except IndexError:
            pass

        self.steps += 1
        return self.phis[self.bounds_progress], self.ls[self.bounds_progress]
