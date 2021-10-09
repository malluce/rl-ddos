#!/usr/bin/env python
import time
from ipaddress import IPv4Address

import gin
import numpy as np

from collections import Counter, defaultdict, namedtuple
from gym.spaces import Box, Discrete, MultiDiscrete
from math import log2, log10, sqrt, exp

from gyms.hhh.cpp.hhhmodule import SketchHHH as HHHAlgo

from gyms.hhh.actionset import HafnerActionSet, RejectionActionSet
from gyms.hhh.images import ImageGenerator
from gyms.hhh.label import Label
from gyms.hhh.state import DistVol, DistVolStd, State


class RulePerformanceTable:
    def __init__(self):
        self.RulePerformance = namedtuple('RulePerformance',
                                          ('num_pkt', 'num_mal_pkt', 'num_ben_pkt', 'rule_perf'))
        # stores (start IP, end IP, len) -> ()
        # both IPs are inclusive
        self.table = {}

    def print_rules(self):
        print('all rules: ')
        for start_ip, end_ip, hhh_len in self.table.keys():
            print(f' {str(IPv4Address(start_ip))}/{hhh_len} (perf={self.table[(start_ip, end_ip, hhh_len)]})')

    def set_rules(self, hhhs):
        self.table = {}
        for hhh in hhhs:
            start_ip = hhh.id
            end_ip = start_ip + Label.subnet_size(hhh.len) - 1
            self.table[(start_ip, end_ip, hhh.len)] = self.RulePerformance(0, 0, 0, 1.0)

    def update(self, id, is_malicious):
        for start_ip, end_ip, hhh_len in self.table.keys():
            if start_ip <= id <= end_ip:
                self._update_entry(start_ip, end_ip, hhh_len, is_malicious)

    def _update_entry(self, start_ip, end_ip, hhh_len, is_malicious):
        # current perf
        n, nm, nb, _ = self.table[(start_ip, end_ip, hhh_len)]

        # update rule performance entries
        n += 1

        if is_malicious:
            nm += 1
        else:
            nb += 1

        prec = nm / n

        # set new perf
        self.table[(start_ip, end_ip, hhh_len)] = self.RulePerformance(n, nm, nb, prec)

    def get_rejected_rules(self, performance_threshold):
        for start_ip, end_ip, hhh_len in list(self.table):
            if self.table[(start_ip, end_ip, hhh_len)].rule_perf < performance_threshold:
                # remove rule from table and return it
                self.table.pop((start_ip, end_ip, hhh_len))
                yield start_ip, end_ip, hhh_len


class Blacklist(object):

    def __init__(self, hhhs):
        self.hhhs = hhhs

        # set up bitmap that indicates whether each IP gets blocked or not
        self.filter_bitmap = np.full((2 ** Loop.ADDRESS_SPACE), False)
        for h in self.hhhs:
            start = h.id
            end = start + Label.subnet_size(h.len)
            self.filter_bitmap[start:end] = True

    def covers(self, ip):
        return self.filter_bitmap[ip]

    def __len__(self):
        return len(self.hhhs)

    def to_serializable(self):
        return [{'id': h.id, 'len': h.len, 'hi': h.hi, 'lo': h.lo}
                for h in self.hhhs]

    def remove_rule(self, ip_start, hhh_len):
        for idx, h in enumerate(self.hhhs):
            if h.id == ip_start and h.len == hhh_len:
                self.hhhs.pop(idx)


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

    @staticmethod
    def _calc_hhh_distance_metrics(b, s):
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

        self.use_hhh_distvol = max(
            [isinstance(feature, DistVol) or isinstance(feature, DistVolStd) for feature in self.state.selection]) > 0

        self.is_hafner = isinstance(self.actionset, HafnerActionSet)
        self.is_rejection = isinstance(self.actionset, RejectionActionSet)
        if self.is_rejection:
            self.rule_perf_table = RulePerformanceTable()

    def reset(self):
        self.blacklist = Blacklist([])
        self.hhh.clear()
        self.state = self.create_state_fn()
        self.trace_ended = False
        self.time_index = 0

        # TODO handle start for our traces in a similar way (first action is otherwise based on random initialized obs)
        if self.is_hafner:
            self.actionset.re_roll_phi()
            self.step(0)  # execute one step with randomly chosen phi

    def step(self, action):
        s = self.state
        self.blacklist_history = []
        if s.trace_start == 1.0:
            self.pre_sample()
        else:
            s.rewind()

        self.resolve_action(action, s)

        # Reverse order to sort by HHH size in descending order
        # Avoids double checking IP coverage
        hhhs = self.hhh.query(s.phi, s.min_prefix)[::-1]

        if self.is_hafner:
            hhhs = apply_hafner_heuristic(hhhs)

        if self.is_rejection:
            # don't remove overlaps (overlapped rules might be useful after rejecting more coarse-grained rules)
            self.rule_perf_table.set_rules(hhhs)
        else:
            hhhs = remove_overlapping_rules(hhhs)

        self.blacklist = Blacklist(hhhs)
        s.blacklist_size = len(self.blacklist)
        s.blacklist_coverage = self._calc_blacklist_coverage(hhhs)
        if self.use_hhh_distvol:
            self._calc_hhh_distance_metrics(hhhs, s)
        # print(f'initial number of rules={len(self.blacklist.hhhs)}')

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
            p, time_index_finished = self.trace.next()

            s.lowest_ip = min(s.lowest_ip, p.ip)
            s.highest_ip = max(s.highest_ip, p.ip)

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
                    if self.is_rejection:
                        self.rule_perf_table.update(p.ip, p.malicious)

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

            if time_index_finished:
                interval += 1
                self.blacklist_history.append(self.blacklist)
                self.time_index += 1
                if self.is_rejection and interval != self.action_interval:
                    # delete rejected rules
                    for rejected_rule in self.rule_perf_table.get_rejected_rules(s.thresh):
                        start_ip, end_ip, hhh_len = rejected_rule
                        # print(f'removing {hhh_len}')
                        self.blacklist.remove_rule(start_ip, hhh_len)

                    # adapt average bl size for state and reward
                    s.blacklist_size = (s.blacklist_size * interval + len(self.blacklist)) / (interval + 1)

        # print(f'final number of rules={len(self.blacklist)}')

        s.complete()

        if self.image_gen is not None:
            s.hhh_image = self.image_gen.generate_image(hhh_algo=self.hhh, hhh_query_result=None)

        if self.time_index == self.trace.trace_sampler.maxtime + 1:  # rate grid ended
            self.trace_ended = True

        return self.trace_ended, self.state, self.blacklist_history

    def resolve_action(self, action, s):
        if self.is_rejection:
            resolved_action = self.actionset.resolve(action)
            if len(resolved_action) == 2:  # only use phi and threshold
                s.phi, s.thresh = resolved_action
                s.min_prefix = 17
            elif len(resolved_action) == 3:  # use phi, threshold and min prefix
                s.phi, s.thresh, s.min_prefix = resolved_action
            else:
                raise ValueError('Unexpected resolved actions')
        else:
            s.phi, s.min_prefix = self.actionset.resolve(action)

    def pre_sample(self):  # Initialize the HHH instance with pre-sampled items
        self.state.trace_start = 0.0
        time_index_finished = False
        interval = 0
        while not (time_index_finished and interval == self.action_interval):
            p, time_index_finished = self.trace.next()
            self.hhh.update(p.ip)

            self.state.lowest_ip = min(self.state.lowest_ip, p.ip)
            self.state.highest_ip = max(self.state.highest_ip, p.ip)

            if time_index_finished:
                interval += 1
                self.blacklist_history.append(self.blacklist)
                self.time_index += 1

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
