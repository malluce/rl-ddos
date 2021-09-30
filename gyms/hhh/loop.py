#!/usr/bin/env python
from ipaddress import IPv4Address

import gin
import numpy as np

from collections import Counter
from gym.spaces import Box, Discrete, MultiDiscrete
from math import log2, log10, sqrt, exp

from gyms.hhh.cpp.hhhmodule import SketchHHH as HHHAlgo

from gyms.hhh.actionset import HafnerActionSet
from gyms.hhh.images import ImageGenerator
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

    def to_serializable(self):
        return [{'id': h.id, 'len': h.len, 'hi': h.hi, 'lo': h.lo}
                for h in self.hhhs]


def remove_overlapping_rules(hhhs):  # remove specific rules that are covered by more general rules
    if len(hhhs) > 0:
        filtered_hhhs = []
        for hhh in sorted(hhhs, key=lambda hhh: hhh.len):
            is_already_covered = False
            for included in filtered_hhhs:
                if hhh.id >= included.id and hhh.id + Label.subnet_size(hhh.len) <= included.id + Label.subnet_size(
                        included.len):
                    is_already_covered = True
            if not is_already_covered:
                filtered_hhhs.append(hhh)
        return filtered_hhhs
    else:
        return hhhs


def apply_hafner_heuristic(hhhs):
    if len(hhhs) > 0:
        longest_hhh_prefix = sorted(hhhs, key=lambda hhh: hhh.len, reverse=True)[0].len
        filtered_hhhs = []
        for hhh in hhhs:
            if hhh.len >= longest_hhh_prefix - 1:
                filtered_hhhs.append(hhh)
        return filtered_hhhs
    else:
        return hhhs


@gin.configurable
class Loop(object):
    ADDRESS_SPACE = 16
    ACTION_INTERVAL = 10
    SAMPLING_RATE = 0.3
    HHH_EPSILON = 0.0001

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

    def __init__(self, trace, create_state_fn, actionset, image_gen: ImageGenerator = None, epsilon=HHH_EPSILON,
                 sampling_rate=SAMPLING_RATE,
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
        self.blacklist_history = []
        self.hhh = HHHAlgo(epsilon)
        self.trace_ended = False
        self.image_gen = image_gen
        self.time_index = 0

    def reset(self):
        self.blacklist = Blacklist([])
        self.hhh.clear()
        self.state = self.create_state_fn()
        self.trace_ended = False
        self.time_index = 0

        if isinstance(self.actionset, HafnerActionSet):
            self.actionset.re_roll_phi()
            self.step(0)  # execute one step with randomly chosen phi

    def step(self, action):
        s = self.state
        self.blacklist_history = []
        if s.trace_start == 1.0:
            s.trace_start = 0.0
            time_index_finished = False
            interval = 0

            # Initialize the HHH instance with pre-sampled items
            while not (time_index_finished and interval == self.action_interval):
                p, time_index_finished = self.trace.next()
                self.hhh.update(p.ip)

                s.lowest_ip = min(s.lowest_ip, p.ip)
                s.highest_ip = max(s.highest_ip, p.ip)

                if time_index_finished:
                    interval += 1
                    self.blacklist_history.append(self.blacklist)
                    self.time_index += 1
        else:
            s.rewind()

        s.phi, s.min_prefix = self.actionset.resolve(action)

        # Reverse order to sort by HHH size in descending order
        # Avoids double checking IP coverage
        hhhs = self.hhh.query(s.phi, s.min_prefix)[::-1]

        if isinstance(self.actionset, HafnerActionSet):
            hhhs = apply_hafner_heuristic(hhhs)

        hhhs = remove_overlapping_rules(hhhs)

        self._calc_blocklist_distr(hhhs, s)

        self.blacklist = Blacklist(hhhs)
        s.blacklist_size = len(self.blacklist)
        s.blacklist_coverage = self._calc_blacklist_coverage(hhhs)
        self._calc_hhh_distance_metrics(hhhs, s)

        if self.image_gen is not None:
            s.image = self.image_gen.generate_image(hhh_algo=self.hhh, hhh_query_result=hhhs)

        # All necessary monitoring information has been extracted from
        # the HHH instance in this step. Reset the HHH algorithm to
        # get rid of stale monitoring information.
        self.hhh.clear()
        s.samples = 0
        time_index_finished = False
        interval = 0
        while not (time_index_finished and interval == self.action_interval):
            try:
                p, time_index_finished = self.trace.next()

                s.lowest_ip = min(s.lowest_ip, p.ip)
                s.highest_ip = max(s.highest_ip, p.ip)

                if time_index_finished:
                    interval += 1
                    self.blacklist_history.append(self.blacklist)
                    self.time_index += 1
                if self.time_index == self.trace.trace_sampler.maxtime + 1:  # rate grid ended
                    raise StopIteration()
            except StopIteration:
                self.trace_ended = True
                break

            s.total += 1
            s.packets_per_step += 1

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

        if self.image_gen is not None:
            s.hhh_image = self.image_gen.generate_image(hhh_algo=self.hhh, hhh_query_result=None)

        return self.trace_ended, self.state, self.blacklist_history

    def _calc_blacklist_coverage(self, hhhs):
        """
        Calculates the fraction of address space that is covered by filter rules.
        :param hhhs:  the hhhs/filter rules
        :return: coverage of address space
        """
        observed_address_space = max(0, self.state.highest_ip - self.state.lowest_ip + 1)
        # count number of covered addresses (ok to count, since no overlapping rules exist at this point)
        covered_addresses = 0
        for hhh in hhhs:
            covered_addresses += Label.subnet_size(hhh.len)

        blacklist_coverage = min(1.0, covered_addresses / observed_address_space)
        return blacklist_coverage

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
