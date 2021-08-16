from abc import ABC, abstractmethod

import gin
import numpy as np

from gyms.hhh.state import State


class RewardCalc(ABC):
    @abstractmethod
    def calc_reward(self, state: State):
        pass

    @abstractmethod
    def weighted_precision(self, precision: float):
        pass

    @abstractmethod
    def weighted_recall(self, recall: float):
        pass

    @abstractmethod
    def weighted_fpr(self, fpr: float):
        pass

    @abstractmethod
    def weighted_bl_size(self, bl_size: int):
        pass


@gin.configurable
class DefaultRewardCalc(RewardCalc):
    def weighted_precision(self, precision):
        return precision ** 6

    def weighted_recall(self, recall):
        return np.sqrt(recall)

    def weighted_fpr(self, fpr):
        return 1 - np.sqrt(fpr)

    def weighted_bl_size(self, bl_size):
        return 1.0 - 0.2 * np.sqrt(np.log2(bl_size))

    def calc_reward(self, state: State):
        if state.blacklist_size == 0:
            # fpr is 0 (nothing blocked, so there are no false positives)
            # precision and recall are 1 if there is no malicious packet, 0 otherwise
            reward = state.precision * state.recall
        else:
            reward = self.weighted_recall(state.recall) * self.weighted_precision(state.precision) * \
                     self.weighted_fpr(state.fpr) * self.weighted_bl_size(state.blacklist_size)
        return reward
