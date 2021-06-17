#!/usr/bin/env python

class Label:
    MAXLEN = 32

    # PREFIXMASK[prefix_len] returns mask for given prefix_len
    PREFIXMASK = [
        0x00000000,
        0x80000000, 0xc0000000, 0xe0000000, 0xf0000000,
        0xf8000000, 0xfc000000, 0xfe000000, 0xff000000,
        0xff800000, 0xffc00000, 0xffe00000, 0xfff00000,
        0xfff80000, 0xfffc0000, 0xfffe0000, 0xffff0000,
        0xffff8000, 0xffffc000, 0xffffe000, 0xfffff000,
        0xfffff800, 0xfffffc00, 0xfffffe00, 0xffffff00,
        0xffffff80, 0xffffffc0, 0xffffffe0, 0xfffffff0,
        0xfffffff8, 0xfffffffc, 0xfffffffe, 0xffffffff
    ]

    @staticmethod
    def generalize(id, length):
        """
        Generalizes id with length by 1 bit.
        """
        return id & Label.PREFIXMASK[length - 1], length - 1

    def __init__(self, id, length=MAXLEN):
        self.id = id & Label.PREFIXMASK[length]
        self.length = length

    def is_root(self):
        return self.length == 0

    def __str__(self):
        return '{}/{}'.format(hex(self.id), self.length)

    def __eq__(self, other):
        return (self.id, self.length) == (other.id, other.length)
