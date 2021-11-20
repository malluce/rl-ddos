#!/usr/bin/env python
import abc

import gin
import numpy as np

from gym.spaces import Box, Discrete, MultiDiscrete, Tuple
from abc import ABC

from numpy.random import default_rng

from gyms.hhh.obs import Observation

from absl import logging


class ActionSet(Observation, ABC):

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


class RejectionActionSet(Observation, ABC):
    def __init__(self):
        self.actionspace = None

    @abc.abstractmethod
    def resolve(self, action):
        """ transform selected action into phi and rejection threshold """
        pass

    @abc.abstractmethod
    def inverse_resolve(self, chosen_action):
        pass

    def get_initialization(self):
        """
        Randomly samples action for use in env.reset(), to build first observation on some action.
        """
        return self.actionspace.sample()


@gin.register
class ContinuousRejectionActionSet(RejectionActionSet):

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
        resolved = agent_action_to_resolved(action, lower_bound=self.get_lower_bound(),
                                            upper_bound=self.get_upper_bound())
        phi = resolved[0]
        thresh = resolved[1]
        return phi, thresh

    def inverse_resolve(self, chosen_action):
        return inverse_resolve(chosen_action, self.get_lower_bound(), self.get_upper_bound())

    def get_observation(self, action):
        return np.array(self.resolve(action))

    def get_lower_bound(self):
        return np.array([0.001, 0.75])

    def get_upper_bound(self):
        return np.array([0.5, 1.0])


class DiscreteActionSet(ActionSet, ABC):
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
                    if action[idx2] != act:
                        all_equal = False
                if all_equal:
                    return idx
        raise ValueError('Chosen action not included in actions.')


@gin.register
class DiscreteRejectionActionSet(DiscreteActionSet, RejectionActionSet):
    def __init__(self):
        super().__init__()
        self.actions = [(x, y)
                        for x in  # phi
                        [(_ + 1) * 1e-3 for _ in range(100)] +  # 0.001, 0.002, ..., 0.1
                        [_ * 1e-2 for _ in range(11, 26)] +  # 0.11, 0.12, ..., 0.25
                        [0.3, 0.4, 0.5]
                        for y in  # thresh
                        [0.75, 0.8, 0.85, 0.9]
                        + [_ * 1e-2 for _ in range(91, 101)]  # 0.91,...,0.99,1.0
                        ]
        self.actionspace = Discrete(len(self.actions))

    def get_lower_bound(self):
        return np.array([0.001, 0.75])

    def get_upper_bound(self):
        return np.array([0.5, 1.0])


@gin.register
class DiscreteRejectionActionSetWithL(DiscreteActionSet, RejectionActionSet):
    def __init__(self):
        super().__init__()
        self.actions = [(x, y, z)
                        for x in
                        [(_ + 1) * 1e-3 for _ in range(100)] +
                        [(_ + 1) * 1e-2 for _ in range(10, 29)] +
                        [_ * 1e-1 for _ in range(3, 11)]
                        for y in [_ * 1e-1 for _ in range(5, 10)]  # 0.5,...,0.9
                        + [_ * 1e-2 for _ in range(91, 101)]  # 0.9,...,0.99,1.0
                        for z in range(16, 23)
                        ]
        self.actionspace = Discrete(len(self.actions))

    def get_lower_bound(self):
        return np.array([0.001, 0.0, 16])

    def get_upper_bound(self):
        return np.array([1.0, 1.0, 32])


@gin.configurable
class SmallDiscreteActionSet(DiscreteActionSet):

    def __init__(self):
        super().__init__()
        self.actions = [(0.25, 17), (0.25, 20), (0.07, 18), (0.02, 17), (0.02, 20)]
        self.actionspace = Discrete(len(self.actions))


@gin.configurable
class MediumDiscreteActionSet(DiscreteActionSet):

    def __init__(self):
        super().__init__()
        self.actions = [(x, y) for x in [0.25, 0.07, 0.02] for y in [17, 18, 19, 20]]
        self.actionspace = Discrete(len(self.actions))


@gin.configurable
class LargeDiscreteActionSet(DiscreteActionSet):

    def __init__(self):
        super().__init__()
        self.actions = [(x, y)
                        for x in [(_ + 1) * 1e-2 for _ in range(24)]
                        for y in [_ + 16 for _ in range(7)]]
        self.actionspace = Discrete(len(self.actions))


@gin.configurable
class VeryLargeDiscreteActionSet(DiscreteActionSet):

    def __init__(self):
        super().__init__()
        self.actions = [(x, y)
                        # high resolution for low phi values, low resolution for high values
                        for x in [(_ + 1) * 1e-2 for _ in range(25)] + [_ * 1e-1 for _ in range(3, 11)]
                        for y in [_ + 16 for _ in range(17)]]
        self.actionspace = Discrete(len(self.actions))


@gin.configurable
class HugeDiscreteActionSet(DiscreteActionSet):

    def __init__(self):
        super().__init__()
        self.actions = [(x, y)
                        # high resolution for low phi values, low resolution for high values
                        # for x in [(_ + 1) * 1e-3 for _ in range(250)] + [_ * 1e-1 for _ in range(3, 11)]
                        for x in
                        [(_ + 1) * 1e-3 for _ in range(100)] +
                        [(_ + 1) * 1e-2 for _ in range(10, 29)] +
                        [_ * 1e-1 for _ in range(3, 11)]
                        for y in [_ + 16 for _ in range(17)]]
        self.actionspace = Discrete(len(self.actions))

    def get_initialization(self):
        return self.actionspace.sample()


def agent_action_to_resolved(agent_action, lower_bound, upper_bound):
    """
    Transforms an action chosen by the agent in [-1.0, 1.0] to a valid value in [lower_bound, upper_bound].
    :param upper_bound: upper bound on the output
    :param lower_bound: lower bound on the output
    :param agent_action: action in [-1.0, 1.0]
    """
    middle = (upper_bound + lower_bound) / 2
    middle_to_bound = np.abs(middle - upper_bound)
    return np.clip(middle + middle_to_bound * agent_action, lower_bound, upper_bound)


def inverse_resolve(chosen_action, lower_bound, upper_bound):
    """
    Inverse of agent_action_to_resolved.
    :param chosen_action: the chosen action e.g. phi=0.2
    :param lower_bound: lower bound of action space
    :param upper_bound: upper bound of action space
    """
    ub, lb = upper_bound, lower_bound
    return 1 / (ub - (ub + lb) / 2) * chosen_action - ((ub + lb) / 2) / (ub - (ub + lb) / 2)


@gin.configurable
class TupleActionSet(ActionSet):
    NUMBER_OF_PREFIXES = 17  # number of different prefixes, e.g. 3 means /32, /31, /30 and 17 means /32.../16

    def __init__(self):
        super(TupleActionSet, self).__init__()
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

    def get_initialization(self):
        return self.actionspace.sample()

    def inverse_resolve(self, chosen_action):
        assert len(chosen_action) == 2
        inverse_phi = inverse_resolve(chosen_action[0], self.get_lower_bound()[0], self.get_upper_bound()[0])
        inverse_l = 32 - chosen_action[1]
        return np.array([inverse_phi, inverse_l])


@gin.configurable
class RejectionTupleActionSet(RejectionActionSet):
    NUMBER_OF_PREFIXES = 17  # number of different prefixes, e.g. 3 means /32, /31, /30 and 17 means /32.../16

    def __init__(self):
        super(RejectionTupleActionSet, self).__init__()
        phi_thresh_space = Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        prefix_space = Discrete(self.NUMBER_OF_PREFIXES)  # values 0..NUMBER_OF_PREFIXES-1
        self.actionspace = Tuple((phi_thresh_space, prefix_space))

    def resolve(self, action):
        phi_thresh = agent_action_to_resolved(action[0], self.get_lower_bound()[:2], self.get_upper_bound()[:2])
        phi = phi_thresh[0]
        thresh = phi_thresh[1]
        prefix_len = 32 - action[1]  # values 32..NUMBER_OF_PREFIXES-1
        return phi, thresh, prefix_len

    def get_observation(self, action):
        return np.array(self.resolve(action))

    def get_lower_bound(self):
        return np.array([0.001, 0.0, 16])

    def get_upper_bound(self):
        return np.array([1.0, 1.0, 32])

    def inverse_resolve(self, chosen_action):
        assert len(chosen_action) == 3
        inverse_phi_thresh = inverse_resolve(chosen_action[:2], self.get_lower_bound()[0], self.get_upper_bound()[1])
        inverse_l = 32 - chosen_action[2]
        return np.array([inverse_phi_thresh, inverse_l])


@gin.register
class HafnerActionSet(ActionSet):

    def __init__(self):
        super().__init__()
        self.re_roll_phi()
        self.possible_actions = {
            0: lambda phi: phi,  # keep last phi
            # assignment actions
            1: lambda phi: 0.5,
            2: lambda phi: 0.3,
            3: lambda phi: 0.1,
            4: lambda phi: 0.01,
            5: lambda phi: 0.001,
            # multiplicative actions
            6: lambda phi: 2 * phi,
            7: lambda phi: 0.5 * phi,
            8: lambda phi: 1.1 * phi,
            9: lambda phi: 0.9 * phi,
            10: lambda phi: 10 * phi,
            11: lambda phi: 0.1 * phi
        }
        self.actionspace = Discrete(len(self.possible_actions))

    def re_roll_phi(self):
        self.current_phi = default_rng().uniform(0.0001, 1.0)
        logging.debug(f're-rolled phi, new: {self.current_phi}')

    def resolve(self, action):
        self.current_phi = self.possible_actions[action](self.current_phi)
        self.current_phi = np.clip(self.current_phi, 0.0001, 1.0)
        min_prefix = 16  # allow unbounded propagation at query time, use pre-processing for L heuristic
        logging.debug(f'action={action}')
        logging.debug(f'resolved actions=({self.current_phi}, {min_prefix})')
        return self.current_phi, min_prefix

    def get_observation(self, action):
        return np.array(self.current_phi)

    def get_lower_bound(self):
        return 0.0001

    def get_upper_bound(self):
        return 1.0

    def get_initialization(self):
        raise NotImplementedError('Initialization not fitting for Hafner')

    def inverse_resolve(self, chosen_action):
        raise NotImplementedError('Inverse resolve not implemented for Hafner')
