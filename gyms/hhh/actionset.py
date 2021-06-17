#!/usr/bin/env python

import numpy as np

from gym.spaces import Box, Discrete, MultiDiscrete


class ActionSet(object):

    def __init__(self):
        self.actionspace = None
        self.shape = None

    def resolve(self, action):
        """ transform selected action into phi and min-prefix values """
        pass

    def get_min_prefixlen(self):
        pass

    def __repr__(self):
        return 'ActionSet'


class SmallDiscreteActionSet(ActionSet):

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


class MediumDiscreteActionSet(ActionSet):

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


class LargeDiscreteActionSet(ActionSet):

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


class MultiDiscreteActionSet(ActionSet):

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


class ContinuousActionSet(ActionSet):

    def __init__(self):
        super().__init__()
        self.actionspace = Box(low=np.array([-1.0]),
                               high=np.array([1.0]), dtype=np.float32)

    def resolve(self, action):
        phi = (action[0] + 1.0) * 0.5 * 0.24 + 0.01

        if phi <= 0.015:
            min_prefixlen = 21
        if phi > 0.015 and phi <= 0.05:
            min_prefixlen = 20
        if phi > 0.05 and phi <= 0.12:
            min_prefixlen = 19
        if phi > 0.12 and phi <= 0.2:
            min_prefixlen = 18
        if phi > 0.2:
            min_prefixlen = 17

        return phi, min_prefixlen

    def get_min_prefixlen(self):
        return 17

    def __repr__(self):
        return str(self.actionspace)
