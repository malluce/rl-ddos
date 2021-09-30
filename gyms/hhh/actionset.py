#!/usr/bin/env python
import abc

import gin
import numpy as np

from gym.spaces import Box, Discrete, MultiDiscrete, Tuple
from abc import ABC

from numpy.random import default_rng

from gyms.hhh.obs import Observation


class ActionSet(Observation, ABC):

    def __init__(self):
        self.actionspace = None
        self.shape = None

    @abc.abstractmethod
    def resolve(self, action):
        """ transform selected action into phi and min-prefix values """
        pass

    @abc.abstractmethod
    def get_min_prefixlen(self):
        pass

    def __repr__(self):
        return 'ActionSet'


class DiscreteActionSet(ActionSet):
    def get_observation(self, action):
        one_hot_obs = np.zeros(self.shape)
        one_hot_obs[action if isinstance(action, (int, np.integer)) else tuple(action)] = 1.0
        return one_hot_obs.flatten()

    def get_lower_bound(self):
        return np.zeros(self.shape)

    def get_upper_bound(self):
        return np.ones(self.shape)

    def get_initialization(self):
        return np.zeros(self.shape)


@gin.configurable
class SmallDiscreteActionSet(DiscreteActionSet):

    def __init__(self):
        super().__init__()
        self.actions = [(0.25, 17), (0.25, 20), (0.07, 18), (0.02, 17), (0.02, 20)]
        self.actionspace = Discrete(len(self.actions))
        self.shape = tuple((len(self.actions),))

    def resolve(self, action):
        return self.actions[action]

    def get_min_prefixlen(self):
        return 17

    def __repr__(self):
        return self.__class__.__name__ + str(self.actions)


@gin.configurable
class MediumDiscreteActionSet(DiscreteActionSet):

    def __init__(self):
        super().__init__()
        self.actions = [(x, y) for x in [0.25, 0.07, 0.02] for y in [17, 18, 19, 20]]
        self.actionspace = Discrete(len(self.actions))
        self.shape = tuple((len(self.actions),))

    def resolve(self, action):
        return self.actions[action]

    def get_min_prefixlen(self):
        return 17

    def __repr__(self):
        return self.__class__.__name__ + str(self.actions)


@gin.configurable
class LargeDiscreteActionSet(DiscreteActionSet):

    def __init__(self):
        super().__init__()
        self.actions = [(x, y)
                        for x in [(_ + 1) * 1e-2 for _ in range(24)]
                        for y in [_ + 16 for _ in range(7)]]
        self.actionspace = Discrete(len(self.actions))
        self.shape = tuple((len(self.actions),))

    def resolve(self, action):
        return self.actions[action]

    def get_min_prefixlen(self):
        return 16

    def __repr__(self):
        return self.__class__.__name__ + str(self.actions)


@gin.configurable
class VeryLargeDiscreteActionSet(DiscreteActionSet):

    def __init__(self):
        super().__init__()
        self.actions = [(x, y)
                        # high resolution for low phi values, low resolution for high values
                        for x in [(_ + 1) * 1e-2 for _ in range(25)] + [_ * 1e-1 for _ in range(3, 11)]
                        for y in [_ + 16 for _ in range(17)]]
        self.actionspace = Discrete(len(self.actions))
        self.shape = tuple((len(self.actions),))

    def resolve(self, action):
        return self.actions[action]

    def get_min_prefixlen(self):
        return 16

    def __repr__(self):
        return self.__class__.__name__ + str(self.actions)


class DirectResolveActionSet(ActionSet):
    """
    Used for eval purposes, where actions are not used as observations (e.g., actions are chosen randomly or fixed)
    """

    def __init__(self):
        self.actionspace = Tuple((Box(-1.0, 1.0, shape=(), dtype=np.float32), Discrete(17)))

    def resolve(self, action):
        return action[0], action[1]

    def get_min_prefixlen(self):
        pass

    def get_observation(self, observed_object):
        pass

    def get_lower_bound(self):
        pass

    def get_upper_bound(self):
        pass

    def get_initialization(self):
        pass


@gin.configurable
class MultiDiscreteActionSet(DiscreteActionSet):

    def __init__(self):
        super().__init__()
        self.actionspace = MultiDiscrete((48, 7))
        self.shape = self.actionspace.nvec

    def resolve(self, action):
        return (action[0] + 1) * 0.5 * 1e-2, action[1] + 16

    def get_min_prefixlen(self):
        return 16

    def __repr__(self):
        return str(self.actionspace)


def agent_action_to_phi(agent_action, lower_bound, upper_bound):
    """
    Transforms an action chosen by the agent in [-1.0, 1.0] to a valid phi value.
    :param upper_bound: upper bound on the output
    :param lower_bound: lower bound on the output
    :param agent_action: action in [-1.0, 1.0]
    """
    return np.clip(0.5 + 0.5 * agent_action, lower_bound, upper_bound)


@gin.configurable
class TupleActionSet(ActionSet):
    NUMBER_OF_PREFIXES = 17  # number of different prefixes, e.g. 3 means /32, /31, /30 and 17 means /32.../16

    def __init__(self):
        super(TupleActionSet, self).__init__()
        phi_space = Box(-1.0, 1.0, shape=(), dtype=np.float32)
        prefix_space = Discrete(self.NUMBER_OF_PREFIXES)  # values 0..NUMBER_OF_PREFIXES-1
        self.actionspace = Tuple((phi_space, prefix_space))

    def resolve(self, action):
        phi = agent_action_to_phi(action[0], self.get_lower_bound(), self.get_upper_bound())
        prefix_len = 32 - action[1]  # values 32..NUMBER_OF_PREFIXES+1
        return phi, prefix_len

    def get_min_prefixlen(self):
        return 32 - self.NUMBER_OF_PREFIXES + 1

    def get_observation(self, action):
        # TODO also min_prefix here? (action[1])
        return agent_action_to_phi(action[0], self.get_lower_bound(), self.get_upper_bound())

    def get_lower_bound(self):
        return 0.001

    def get_upper_bound(self):
        return 1.0

    def get_initialization(self):
        return 0.12


@gin.configurable
class ContinuousActionSet(ActionSet):

    def __init__(self):
        super().__init__()
        self.actionspace = Box(low=-1.0,
                               high=1.0,
                               shape=(),
                               dtype=np.float32)

    def get_observation(self, action):
        # return action
        return agent_action_to_phi(action, self.get_lower_bound(), self.get_upper_bound())

    def get_lower_bound(self):
        return 0.01

    def get_upper_bound(self):
        return 1.0

    def get_initialization(self):
        return 0.12

    def resolve(self, action):
        phi = agent_action_to_phi(action, self.get_lower_bound(), self.get_upper_bound())
        min_prefixlen = self._phi_to_prefixlen(phi)

        return phi, min_prefixlen

    def _phi_to_prefixlen(self, phi):
        min_prefixlen = 0
        if phi <= 0.06:
            min_prefixlen = 21
        if 0.06 < phi <= 0.2:
            min_prefixlen = 20
        if 0.2 < phi <= 0.48:
            min_prefixlen = 19
        if 0.48 < phi <= 0.8:
            min_prefixlen = 18
        if phi > 0.8:
            min_prefixlen = 17
        return min_prefixlen

    def get_min_prefixlen(self):
        return 17

    def __repr__(self):
        return str(self.actionspace)


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
        print(f're-rolled phi, new: {self.current_phi}')

    def resolve(self, action):
        self.current_phi = self.possible_actions[action](self.current_phi)
        self.current_phi = np.clip(self.current_phi, 0.0001, 1.0)
        min_prefix = 16  # allow unbounded propagation at query time, use pre-processing for L heuristic
        print(f'action={action}')
        print(f'resolved actions=({self.current_phi}, {min_prefix})')
        return self.current_phi, min_prefix

    def get_min_prefixlen(self):
        return 16

    def get_observation(self, action):
        return self.current_phi

    def get_lower_bound(self):
        return 0.0001

    def get_upper_bound(self):
        return 1.0

    def get_initialization(self):
        raise NotImplementedError('Initialization not fitting for Hafner')
