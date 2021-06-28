from abc import ABC, abstractmethod

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

    def __init__(self, phi: float, min_prefix_length: int):
        self.phi = phi
        self.min_prefix_length = min_prefix_length

    def action(self, observation):
        return self.phi, self.min_prefix_length
