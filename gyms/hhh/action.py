#!/usr/bin/env python
import abc
import math

import gin
import numpy as np

from gym.spaces import Box, Discrete, Tuple
from abc import ABC

from gyms.hhh.obs import Observation


class ActionSpace(Observation, ABC):

    def __init__(self):
        self.actionspace = None
        self.shape = None

    @abc.abstractmethod
    def resolve(self, action):
        """ transform selected action into phi and min-prefix values """
        pass

    @abc.abstractmethod
    def inverse_resolve(self, chosen_action):
        pass

    def get_initialization(self):
        """
        Randomly samples action for use in env.reset(), to build first observation on some action.
        """
        return self.actionspace.sample()


class RejectionActionSpace(ActionSpace, ABC):  # marker interface
    def __init__(self):
        super(RejectionActionSpace, self).__init__()


@gin.register
class ExponentialContinuousRejectionActionSpace(RejectionActionSpace):
    """DDPG and PPO (phi, pthresh) action space"""

    def __init__(self):
        super().__init__()
        self.actionspace = Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )
        self.shape = self.actionspace.shape

    def resolve(self, action):
        resolved_pthresh = agent_action_to_resolved(action[1], lower_bound=self.get_lower_bound()[1],
                                                    upper_bound=self.get_upper_bound()[1])
        resolved_phi = agent_action_to_resolved_phi(action[0], lower_bound=self.get_lower_bound()[0],
                                                    upper_bound=self.get_upper_bound()[0])
        return resolved_phi, resolved_pthresh

    def inverse_resolve(self, chosen_action):
        raise NotImplementedError('need to implement invers_resolve for new phi scaling')
        # return inverse_resolve(chosen_action, self.get_lower_bound(), self.get_upper_bound())

    def get_observation(self, action):
        return np.array(self.resolve(action))

    def get_lower_bound(self):
        return np.array([0.001, 0.85])

    def get_upper_bound(self):
        return np.array([0.3, 1.0])


@gin.configurable
class PpoMinPrefLenActionSpace(ActionSpace):
    """PPO (phi,L) action space"""
    NUMBER_OF_PREFIXES = 17  # number of different prefixes, e.g. 3 means /32, /31, /30 and 17 means /32.../16

    def __init__(self):
        super(PpoMinPrefLenActionSpace, self).__init__()
        phi_space = Box(-1.0, 1.0, shape=(), dtype=np.float32)
        prefix_space = Discrete(self.NUMBER_OF_PREFIXES)  # values 0..NUMBER_OF_PREFIXES-1
        self.actionspace = Tuple((phi_space, prefix_space))

    def resolve(self, action):
        phi = agent_action_to_resolved(action[0], self.get_lower_bound()[0], self.get_upper_bound()[0])
        prefix_len = int(32 - action[1])  # values 32..NUMBER_OF_PREFIXES-1
        return phi, prefix_len

    def get_observation(self, action):
        return np.array(self.resolve(action))

    def get_lower_bound(self):
        return np.array([0.001, 16])

    def get_upper_bound(self):
        return np.array([0.5, 32])

    def inverse_resolve(self, chosen_action):
        assert len(chosen_action) == 2
        inverse_phi = inverse_resolve(chosen_action[0], self.get_lower_bound()[0], self.get_upper_bound()[0])
        inverse_l = 32 - chosen_action[1]
        return np.array([inverse_phi, inverse_l])


@gin.register
class DdpgMinPrefLenActionSpace(ActionSpace):
    """DDPG (phi, L) action space"""

    def __init__(self):
        super().__init__()
        self.actionspace = Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )
        self.shape = self.actionspace.shape

    def resolve(self, action):
        resolved_l = agent_action_to_resolved_l(action[1], lower_bound=self.get_lower_bound()[1],
                                                upper_bound=self.get_upper_bound()[1])
        resolved_phi = agent_action_to_resolved_phi(action[0], lower_bound=self.get_lower_bound()[0],
                                                    upper_bound=self.get_upper_bound()[0])
        return resolved_phi, resolved_l

    def inverse_resolve(self, chosen_action):
        raise NotImplementedError('need to implement invers_resolve for new phi scaling')
        # return inverse_resolve(chosen_action, self.get_lower_bound(), self.get_upper_bound())

    def get_observation(self, action):
        return np.array(self.resolve(action))

    def get_lower_bound(self):
        return np.array([0.001, 16])

    def get_upper_bound(self):
        return np.array([0.3, 32])


class DiscreteActionSpace(ActionSpace, ABC):
    def __init__(self):
        super().__init__()
        self.actions = None

    def get_observation(self, action):
        return np.array(self.resolve(action))
        # one_hot_obs = np.zeros(self.shape)
        # one_hot_obs[action if isinstance(action, (int, np.integer)) else tuple(action)] = 1.0
        # return one_hot_obs.flatten()

    def get_lower_bound(self):
        return np.array([0.001, 16])

    def get_upper_bound(self):
        return np.array([1.0, 32])

    def __repr__(self):
        return self.__class__.__name__ + str(self.actions)

    def resolve(self, action):
        return self.actions[action]

    def inverse_resolve(self, chosen_action):
        for idx, action in enumerate(self.actions):
            if len(action) == len(chosen_action):
                all_equal = True
                for idx2, act in enumerate(chosen_action):
                    if not math.isclose(action[idx2], act):
                        all_equal = False
                if all_equal:
                    return idx
        raise ValueError(f'Chosen action {chosen_action} not included in actions.')


@gin.register
class DqnRejectionActionSpace(DiscreteActionSpace, RejectionActionSpace):
    """DQN (phi, pthresh) action space"""

    def __init__(self):
        super().__init__()
        self.actions = [(x, y)
                        for x in  # phi
                        [(_ + 1) * 1e-3 for _ in range(10)] +  # 0.001, 0.002, ..., 0.01
                        [_ * 1e-2 for _ in range(2, 11)] +  # 0.02, 0.03, ..., 0.1
                        [0.2, 0.3]
                        for y in  # thresh
                        [0.85, 0.9, 0.925, 0.95, 0.975, 1.0]
                        ]
        self.actionspace = Discrete(len(self.actions))

    def get_lower_bound(self):
        return np.array([0.001, 0.85])

    def get_upper_bound(self):
        return np.array([0.3, 1.0])


@gin.configurable
class DqnMinPrefLenActionSpace(DiscreteActionSpace):
    """DQN (phi, L) action space"""

    def __init__(self):
        super().__init__()
        self.actions = [(x, y)
                        # high resolution for low phi values, low resolution for high values
                        for x in  # phi
                        [(_ + 1) * 1e-3 for _ in range(10)] +  # 0.001, 0.002, ..., 0.01
                        [_ * 1e-2 for _ in range(2, 11)] +  # 0.02, 0.03, ..., 0.1
                        [0.2, 0.3]
                        for y in [_ + 16 for _ in range(17)]]  # l
        self.actionspace = Discrete(len(self.actions))


def agent_action_to_resolved(agent_action, lower_bound, upper_bound):
    """
    Transforms an action chosen by the agent in [-1.0, 1.0] to a valid value in [lower_bound, upper_bound].
    Uses linear resolving.
    :param upper_bound: upper bound on the output
    :param lower_bound: lower bound on the output
    :param agent_action: action in [-1.0, 1.0]
    """
    middle = (upper_bound + lower_bound) / 2
    middle_to_bound = upper_bound - middle
    return np.clip(middle + middle_to_bound * agent_action, lower_bound, upper_bound)


def agent_action_to_resolved_l(agent_action, lower_bound, upper_bound):
    """
    Resolves agent action to L parameter.
    :param agent_action: action in [-1.0, 1.0]
    :param lower_bound: lower bound on L (output)
    :param upper_bound: upper bound on L (output)
    :return:
    """
    bin_size = 2 / 17  # range[-1,1], 17 L values (16,17,...,32)
    return int(min((agent_action + 1) / bin_size + lower_bound, upper_bound))


def agent_action_to_resolved_phi(agent_action, lower_bound, upper_bound):
    """
    Transforms an action chosen by the agent in [-1.0, 1.0] to a valid value in [lower_bound, upper_bound] by applying sigmoid scaling.
    :param upper_bound: upper bound on the output
    :param lower_bound: lower bound on the output
    :param agent_action: action in [-1.0, 1.0]
    """
    return np.maximum((1 / (1 + np.exp(-3.25 * (agent_action - 1)))) * 2 * upper_bound, lower_bound)


def inverse_resolve(chosen_action, lower_bound, upper_bound):
    """
    Inverse of agent_action_to_resolved.
    :param chosen_action: the chosen action e.g. phi=0.2
    :param lower_bound: lower bound of action space
    :param upper_bound: upper bound of action space
    """
    ub, lb = upper_bound, lower_bound
    return 1 / (ub - (ub + lb) / 2) * chosen_action - ((ub + lb) / 2) / (ub - (ub + lb) / 2)
