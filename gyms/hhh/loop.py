#!/usr/bin/env python

import gin
import numpy as np

from collections import Counter
from gym.spaces import Box, Discrete, MultiDiscrete
from math import log2, log10, sqrt, exp

from gyms.hhh.cpp.hhhmodule import SketchHHH as HHHAlgo
from gyms.hhh.label import Label
from gyms.hhh.state import State


class Blacklist(object):

    def __init__(self, hhhs):
        self.hhhs = hhhs

    def covers(self, ip):
        for h in self.hhhs:
            if ip & Label.PREFIXMASK[h.len] == h.id:
                return True

        return False

    def __len__(self):
        return len(self.hhhs)


@gin.configurable
class Loop(object):
    ACTION_INTERVAL = 500
    SAMPLING_RATE = 0.3
    HHH_EPSILON = 0.0001

    def __init__(self, trace, create_state_fn, actionset, epsilon=HHH_EPSILON, sampling_rate=SAMPLING_RATE,
                 action_interval=ACTION_INTERVAL):
        self.create_state_fn = create_state_fn
        self.state = create_state_fn()
        self.trace = trace
        self.actionset = actionset
        self.epsilon = epsilon
        self.sampling_rate = sampling_rate
        self.weight = 1.0 / sampling_rate
        self.action_interval = action_interval
        self.blacklist = Blacklist([])
        self.hhh = HHHAlgo(epsilon)
        self.trace_ended = False
        self.packets = []

    def reset(self):
        self.blacklist = Blacklist([])
        self.hhh = HHHAlgo(self.epsilon)
        self.state = self.create_state_fn()
        self.trace_ended = False
        self.packets = []

    @staticmethod
    def gauss(n):
        return (n ** 2 + n) // 2

    @staticmethod
    def hhh_voldist(lo, hi):
        n = lo.len
        m = hi.len

        end = lambda h: h.id + 2 ** (32 - h.len)

        if end(lo) < end(hi):
            d = hi.id - end(lo)
        else:
            d = lo.id - end(hi)

        return n * m * d + n * Loop.gauss(m - 1) + m * Loop.gauss(n - 1)

    def step(self, action):
        s = self.state

        if s.trace_start == 1.0:
            s.trace_start = 0.0
            # step_finished = False ## TODO revert commit

            for i in range(self.action_interval):  ## TODO revert commit
                # while not step_finished: ## TODO revert commit
                # pre-sample without incrementing the
                # trace counter when initiating a new
                # episode
                p = self.trace.sample()
                # p, step_finished = self.trace.next() ## TODO revert commit
                self.hhh.update(p.ip)
        else:
            s.rewind()

        s.phi, s.min_prefix = self.actionset.resolve(action)

        # Sorting avoids double-checking the same
        # IP range when determining hhh coverage
        b = sorted(self.hhh.query(s.phi, s.min_prefix), key=lambda _: _.len)

        self._calc_blocklist_distr(b, s)

        self.blacklist = Blacklist(b)
        s.blacklist_size = len(self.blacklist)

        self._calc_hhh_distance_metrics(b, s)

        s.samples = 0
        # step_finished = False ## TODO revert commit

        for i in range(self.action_interval):  ## TODO revert commit
            # while not step_finished: ## TODO revert commit
            try:
                # p, step_finished = self.trace.next()## TODO revert commit
                p = self.trace.next()
                self.packets.append(p)
            except StopIteration:
                self.trace_ended = True
                break

            s.total += 1

            if p.malicious:
                s.malicious += 1

            if self.blacklist.covers(p.ip):
                s.blocked += 1

                if p.malicious:
                    s.malicious_blocked += 1

                if np.random.random() < self.sampling_rate:
                    s.samples += 1

                    self.hhh.update(p.ip, int(self.weight))

                    # Estimate the number of mal packets
                    # filtered by the blacklist by sampling
                    if p.malicious:
                        s.estimated_malicious_blocked += 1
                    else:
                        s.estimated_benign_blocked += 1
            else:
                self.hhh.update(p.ip)

                if p.malicious:
                    s.malicious_passed += 1
                else:
                    s.benign_passed += 1

        s.episode_progress = 1.0 * s.total / self.trace.N

        s.complete()

        return self.trace_ended, self.state

    def _calc_hhh_distance_metrics(self, b, s):
        H = list(sorted(b, key=lambda _: _.id))
        if len(H) > 1:
            vd = [Loop.hhh_voldist(H[i], H[j])
                  for i in range(len(H))
                  for j in range(i + 1, len(H))]

            # Remove (negative) voldist of any two overlapping HHHs
            vd = [_ for _ in vd if _ > 0]

            if vd:
                scale = lambda x, n: log10(x) / n
                sigmoid = lambda x: 1.0 / (1.0 + exp(-x))
                squash = lambda x, n: 0 if x == 0 else sigmoid(10.0 * (scale(x, n) - 0.5))
                #				crop = lambda x : 1.0 * (x - 0xffff0000) / 0xffff
                crop = lambda x: 1.0 * x / 0xffffffff

                s.hhh_distance_avg = squash(np.mean(vd), 14)
                s.hhh_distance_sum = squash(sum(vd), 14)
                s.hhh_distance_min = squash(min(vd), 14)
                s.hhh_distance_max = squash(max(vd), 14)
                s.hhh_distance_std = squash(np.std(vd), 9)
                s.hhh_min = crop(H[0].id)
                s.hhh_max = crop(max([_.id + 2 ** (32 - _.len) for _ in H]))

    def _calc_blocklist_distr(self, b, s):
        s.bl_dist = np.zeros(16)
        blcnt = Counter([_.len for _ in b])
        for k, v in blcnt.items():
            s.bl_dist[k - 17] = v
