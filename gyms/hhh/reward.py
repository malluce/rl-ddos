from abc import ABC, abstractmethod
from math import log2, sqrt

import gin

from gyms.hhh.state import State


class RewardCalc(ABC):
    @abstractmethod
    def calc_reward(self, state: State):
        pass


@gin.configurable
class DefaultRewardCalc(RewardCalc):
    def calc_reward(self, state: State):
        if state.blacklist_size == 0:
            # fpr is 0 (nothing blocked, so there are no false positives)
            # precision and recall are 1 if there is no malicious packet, 0 otherwise
            reward = state.precision * state.recall
        else:
            reward = (
                    (state.precision ** 6) * sqrt(state.recall) * (1 - sqrt(state.fpr))
                    * (1.0 - 0.2 * sqrt(log2(state.blacklist_size)))
            )
        return reward
