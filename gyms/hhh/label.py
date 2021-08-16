#!/usr/bin/env python

from .cpp import hhhmodule


class Label:
    PREFIXMASK = [(1 << hhhmodule.HIERARCHY_SIZE) - (1 << l)
                  for l in reversed(range(hhhmodule.HIERARCHY_SIZE + 1))]
    SIZE_BY_LENGTH = [1 << (hhhmodule.HIERARCHY_SIZE - l) for l in range(33)]

    @staticmethod
    def generalize(id, length):
        assert length > 0
        return id & Label.PREFIXMASK[length - 1], length - 1

    @staticmethod
    def subnet_size(length):
        return Label.SIZE_BY_LENGTH[length]

    def __init__(self, id, length=hhhmodule.HIERARCHY_SIZE):
        self.id = id & Label.PREFIXMASK[length]
        self.length = length

    def is_root(self):
        return self.length == 0

    def __str__(self):
        return '{}/{}'.format(hex(self.id), self.length)

    def __eq__(self, other):
        return (self.id, self.length) == (other.id, other.length)
