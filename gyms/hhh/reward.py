from abc import ABC, abstractmethod

import gin
import numpy as np

from gyms.hhh.state import State
from gyms.hhh.obs import HafnerObservations


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
class AdditiveRewardExponentialWeights(RewardCalc):

    def __init__(self, precision_weight, fpr_weight, recall_weight, bl_weight):
        self.precision_weight = precision_weight
        self.fpr_weight = fpr_weight
        self.recall_weight = recall_weight
        self.bl_weight = bl_weight

    def calc_reward(self, state: State):
        return self.weighted_precision(state.precision) \
               + self.weighted_bl_size(state.blacklist_size) \
               + self.weighted_recall(state.recall) \
               + self.weighted_fpr(state.fpr)

    def weighted_precision(self, precision: float):
        return precision ** self.precision_weight

    def weighted_recall(self, recall: float):
        return recall ** self.recall_weight

    def weighted_fpr(self, fpr: float):
        return (1 - fpr) ** self.fpr_weight
 
    def weighted_bl_size(self, bl_size: int):
        return 1 - self.bl_weight * np.log2(np.maximum(1, bl_size))


@gin.configurable
class AdditiveRewardCalc(RewardCalc):

    def __init__(self, precision_weight, fpr_weight, recall_weight, bl_weight):
        self.precision_weight = precision_weight
        self.fpr_weight = fpr_weight
        self.recall_weight = recall_weight
        self.bl_weight = bl_weight

    def calc_reward(self, state: State):
        return self.weighted_precision(state.precision) \
               + self.weighted_bl_size(state.blacklist_size) \
               + self.weighted_recall(state.recall) \
               + self.weighted_fpr(state.fpr)

    def weighted_precision(self, precision: float):
        return self.precision_weight * precision

    def weighted_recall(self, recall: float):
        return self.recall_weight * recall

    def weighted_fpr(self, fpr: float):
        return self.fpr_weight * (1 - fpr)

    def weighted_bl_size(self, bl_size: int):
        return 1 - self.bl_weight * np.log2(np.maximum(1, bl_size))


@gin.register
class HafnerRewardCalc(RewardCalc):
    TCAM_CAP = 0

    def __init__(self, tcam_cap=TCAM_CAP):
        self.tcam_cap = tcam_cap

    def calc_reward(self, state: State):
        reward = self.weighted_precision(state.precision) \
                 * self.weighted_recall(state.recall) \
                 * self.weighted_fpr(state.fpr) \
                 * 300 \
                 + self.weighted_bl_size(state.blacklist_size)

        return reward

    def weighted_precision(self, precision: float):
        return precision

    def weighted_recall(self, recall: float):
        return recall

    def weighted_fpr(self, fpr: float):
        return 1 - fpr

    def weighted_bl_size(self, bl_size: int):
        return (1 - bl_size / self.tcam_cap) * 100


@gin.configurable
class DefaultRewardCalc(RewardCalc):
    def weighted_precision(self, precision):
        return precision ** 6

    def weighted_recall(self, recall):
        return np.sqrt(recall)

    def weighted_fpr(self, fpr):
        return 1 - np.sqrt(fpr)

    def weighted_bl_size(self, bl_size):
        if bl_size < 1:
            return 1
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


@gin.configurable
class MultiplicativeReward(DefaultRewardCalc):

    def __init__(self, precision_weight, fpr_weight, recall_weight, bl_weight):
        self.precision_weight = precision_weight
        self.fpr_weight = fpr_weight
        self.recall_weight = recall_weight
        self.bl_weight = bl_weight

    def weighted_precision(self, precision):
        return precision ** self.precision_weight

    def weighted_fpr(self, fpr):
        return 1 - fpr ** self.fpr_weight

    def weighted_recall(self, recall):
        return recall ** self.recall_weight

    def weighted_bl_size(self, bl_size):
        # max to avoid nan, avg bl_size can be < 1
        return 1.0 - self.bl_weight * np.sqrt(np.log2(np.maximum(1, bl_size)))


@gin.configurable
class MultiplicativeRewardSpecificity(MultiplicativeReward):
    def weighted_fpr(self, fpr):
        return (1 - fpr) ** self.fpr_weight
