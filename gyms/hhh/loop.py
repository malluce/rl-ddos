#!/usr/bin/env python
import time
from ipaddress import IPv4Address

import gin
import numpy as np
from absl import logging
from collections import Counter, defaultdict, namedtuple
from gym.spaces import Box, Discrete, MultiDiscrete
from math import log2, log10, sqrt, exp

from gyms.hhh.cpp.hhhmodule import SketchHHH as HHHAlgo

from gyms.hhh.actionset import HafnerActionSet, RejectionActionSet
from gyms.hhh.images import ImageGenerator
from gyms.hhh.label import Label
from gyms.hhh.state import State
from gyms.hhh.obs import DistVol, DistVolStd


@gin.configurable
class RulePerformanceTable:
    CACHE_CAP = 100
    EWMA_WEIGHT = 0.5

    def __init__(self, use_cache, metric, cache_capacity=CACHE_CAP, ewma_weight=EWMA_WEIGHT):
        self.RulePerformance = namedtuple('RulePerformance',
                                          ('num_pkt', 'num_mal_pkt', 'num_ben_pkt', 'rule_perf'))
        # stores (start IP, end IP, len) -> RulePerformance, both IPs are inclusive
        self.table = {}

        assert metric in ['fpr', 'prec']
        self.metric = metric
        logging.info(f'using {self.metric} metric for rule performance')

        self.use_cache = use_cache
        logging.info(f'using cache: {self.use_cache} (capacity={cache_capacity})')

        # stores (start IP, end IP, len) -> RulePerformance, both IPs are inclusive
        self.cache = {}
        self.cache_capacity = cache_capacity
        self.ewma_weight = ewma_weight

    def print_rules(self):
        if logging.get_verbosity() == 'debug':
            logging.debug('all rules: ')
            for start_ip, end_ip, hhh_len in self.table.keys():
                logging.debug(
                    f' {str(IPv4Address(start_ip))}/{hhh_len} (perf={self.table[(start_ip, end_ip, hhh_len)]})')

    def print_cache(self):
        if self.use_cache and logging.get_verbosity() == 'debug':
            logging.debug(f'all cache entries: (total={len(self.cache)})')
            len_to_count = defaultdict(lambda: 0)
            for start_ip, end_ip, hhh_len in self.cache.keys():
                len_to_count[hhh_len] += 1
                logging.debug(
                    f' {str(IPv4Address(start_ip))}/{hhh_len} (perf={self.cache[(start_ip, end_ip, hhh_len)]})')
            for l in sorted(len_to_count.keys()):
                logging.debug(f' {l}:{len_to_count[l]}')

    def set_rules(self, hhhs):
        self.table = {}
        for hhh in hhhs:
            start_ip = hhh.id
            end_ip = start_ip + Label.subnet_size(hhh.len) - 1
            self.table[(start_ip, end_ip, hhh.len)] = self.RulePerformance(0, 0, 0, None)

    def filter_hhhs(self, hhhs):
        if not self.use_cache:
            return hhhs
        # only return hhhs that are not in cache
        result = []
        for hhh in hhhs:
            start_ip = hhh.id
            end_ip = start_ip + Label.subnet_size(hhh.len) - 1
            if (start_ip, end_ip, hhh.len) not in self.cache:
                result.append(hhh)
            else:
                logging.debug(f'not applying HHH {str(IPv4Address(start_ip))}/{hhh.len}, because in cache')
        return result

    def update(self, id, is_malicious, num):
        for start_ip, end_ip, hhh_len in self.table.keys():
            if start_ip <= id <= end_ip:
                self._update_entry_in(self.table, start_ip, end_ip, hhh_len, is_malicious, num=num)

    def _compute_rule_performance(self, n, nb, nm, total_benign):
        if self.metric == 'fpr':
            rule_fpr = nb / total_benign if total_benign != 0 else 0.0
            return max(0, 1 - rule_fpr)
        elif self.metric == 'prec':
            return nm / n if n != 0 else 1.0
        else:
            raise ValueError(f'Unexpected metric for rule performance: {self.metric}')

    def _update_entry_in(self, table_to_update, start_ip, end_ip, hhh_len, is_malicious, num):
        """
        Updates the counters of a given rule in a given table without re-computing the rule performance.
        """
        # current perf
        n, nm, nb, perf = table_to_update[(start_ip, end_ip, hhh_len)]

        # update rule performance entries
        n += num

        if is_malicious:
            nm += num
        else:
            nb += num

        # update counters, keep old performance
        table_to_update[(start_ip, end_ip, hhh_len)] = self.RulePerformance(n, nm, nb, perf)

    def refresh_table_perf(self, total_benign):
        """
        Update the performance of each TABLE RULE by computing performance based on counters. Called each time idx.
        :param total_benign: number of benign packets of the time idx
        """
        for rule in self.table.keys():
            n, nm, nb, _ = self.table[rule]
            rule_perf = self._compute_rule_performance(n=n, nm=nm, nb=nb, total_benign=total_benign)
            self.table[rule] = self.RulePerformance(n, nm, nb, rule_perf)

    def _add_to_cache(self, start_ip, end_ip, hhh_len, rule_perf):
        assert len(self.cache) <= self.cache_capacity

        if len(self.cache) == self.cache_capacity:
            # replace old rule if cache full and new rule is worse than current best rule in cache
            cache_by_performance = sorted(self.cache.items(), key=lambda x: x[1].rule_perf)
            # best performing rule in cache
            best_rule, best_performance = cache_by_performance[-1]
            if rule_perf <= best_performance.rule_perf:
                logging.debug(f' Replacement of {best_rule, best_performance}')
                self.cache.pop(best_rule)
                self.cache[(start_ip, end_ip, hhh_len)] = self.RulePerformance(0, 0, 0, rule_perf)
            else:
                logging.debug(f' Not replacing, performance better')
        elif len(self.cache) < self.cache_capacity:
            logging.debug(f' Cache not full, just adding')
            self.cache[(start_ip, end_ip, hhh_len)] = self.RulePerformance(0, 0, 0, rule_perf)

        assert len(self.cache) <= self.cache_capacity

    def update_cache(self, ip, is_malicious, num):
        if not self.use_cache:
            return
        for start_ip, end_ip, hhh_len in self.cache.keys():
            if start_ip <= ip <= end_ip:
                self._update_entry_in(self.cache, start_ip, end_ip, hhh_len, is_malicious, num=num)

    def refresh_cache(self, perf_thresh, total_benign):
        if not self.use_cache:
            return
        for rule in list(self.cache):
            # update perf via EWMA
            n, nm, nb, old_perf = self.cache[rule]
            if n == 0:  # no packet applied to this rule, incentivize deletion of this rule
                new_perf = 1.0
            else:
                new_perf = self._compute_rule_performance(n=n, nm=nm, nb=nb, total_benign=total_benign)
            ewma_perf = max(0, self.ewma_weight * new_perf + (1 - self.ewma_weight) * old_perf)
            if ewma_perf >= perf_thresh:
                logging.debug(
                    f' removing rule {str(IPv4Address(rule[0]))}/{rule[2]} from cache because it got better {n, nm, nb, ewma_perf}')
                self.cache.pop(rule)  # delete rules that have better performance
            else:
                self.cache[rule] = self.RulePerformance(0, 0, 0, ewma_perf)  # otherwise reset counters

    def get_rejected_rules(self, performance_threshold):
        for start_ip, end_ip, hhh_len in list(self.table):
            rule_perf = self.table[(start_ip, end_ip, hhh_len)].rule_perf
            if rule_perf < performance_threshold:
                if self.use_cache:
                    logging.debug(
                        f' rejecting rule {str(IPv4Address(start_ip))}/{hhh_len} (perf={rule_perf}); adding to cache')
                    self._add_to_cache(start_ip, end_ip, hhh_len, rule_perf)

                # remove rule from table and return it
                self.table.pop((start_ip, end_ip, hhh_len))
                yield start_ip, end_ip, hhh_len


class Blacklist(object):

    def __init__(self, hhhs):
        self.hhhs = hhhs
        self._compute_filter_bitmap()

    def _compute_filter_bitmap(self):
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
        self._compute_filter_bitmap()  # re-compute after removing rules


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
        self.weight = int(1.0 / sampling_rate)
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
        logging.info(f'is rejection: {self.is_rejection}')
        logging.info(f'is hafner: {self.is_hafner}')
        if self.is_rejection:
            self.rule_perf_table = RulePerformanceTable()

    def reset(self):
        self.blacklist = Blacklist([])
        self.hhh.clear()
        self.state = self.create_state_fn()
        self.trace_ended = False
        self.time_index = 0

        if self.is_hafner:
            self.actionset.re_roll_phi()
            self.step(0)  # execute one step with randomly chosen phi
            return 0, self.blacklist_history
        else:
            first_action = self.actionset.get_initialization()
            logging.info(f'first action: {first_action}')
            self.step(first_action)
            return first_action, self.blacklist_history

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
            logging.debug('Setting rules.')
            # don't remove overlaps (overlapped rules might be useful after rejecting more coarse-grained rules)
            hhhs = self.rule_perf_table.filter_hhhs(hhhs)
            self.rule_perf_table.set_rules(hhhs)
        else:
            hhhs = remove_overlapping_rules(hhhs)

        self.blacklist = Blacklist(hhhs)
        s.blacklist_size = len(self.blacklist)
        s.blacklist_coverage = self._calc_blacklist_coverage(hhhs)
        if self.use_hhh_distvol:
            self._calc_hhh_distance_metrics(hhhs, s)
        logging.debug(f'initial number of rules={len(self.blacklist.hhhs)}')

        if self.image_gen is not None:
            s.image = self.image_gen.generate_image(hhh_algo=self.hhh, hhh_query_result=hhhs)

        # All necessary monitoring information has been extracted from
        # the HHH instance in this step. Reset the HHH algorithm to
        # get rid of stale monitoring information.
        self.hhh.clear()
        s.samples = 0
        time_index_finished = False
        interval = 0

        benign_idx = 0
        malicious_idx = 0
        malicious_blocked_idx = 0
        blocked_idx = 0
        benign_passed_idx = 0
        sampled_benign_blocked_idx = 0
        benign_blocked_idx = 0
        while not (time_index_finished and interval == self.action_interval):
            try:
                p, time_index_finished = self.trace.next()
            except StopIteration:
                logging.info(
                    f'Encountered StopIteration at time index {self.time_index}, no packet at this time index.')
                p = None
                time_index_finished = True

            if p is not None:
                s.lowest_ip = min(s.lowest_ip, p.ip)
                s.highest_ip = max(s.highest_ip, p.ip)

                s.total += 1
                s.packets_per_step += 1

                if p.malicious:
                    s.malicious += 1
                    malicious_idx += 1
                else:
                    benign_idx += 1

                if self.blacklist.covers(p.ip):
                    blocked_idx += 1
                    s.blocked += 1

                    if p.malicious:
                        s.malicious_blocked += 1
                        malicious_blocked_idx += 1
                    else:
                        benign_blocked_idx += 1

                    rand = np.random.random()
                    if rand < self.sampling_rate:
                        s.samples += 1
                        self.hhh.update(p.ip, int(self.weight))
                        if self.is_rejection:
                            self.rule_perf_table.update(p.ip, p.malicious, num=self.weight)
                            self.rule_perf_table.update_cache(p.ip, p.malicious, num=self.weight)
                        # Estimate the number of mal packets
                        # filtered by the blacklist by sampling
                        if p.malicious:
                            s.estimated_malicious_blocked += 1
                        else:
                            s.estimated_benign_blocked += 1
                            sampled_benign_blocked_idx += 1
                            logging.debug(f'[sampled] benign blocked: {str(IPv4Address(p.ip))}')
                    else:
                        if not p.malicious:
                            logging.debug(f'[not sampled] benign blocked: {str(IPv4Address(p.ip))}')
                else:
                    self.hhh.update(p.ip)
                    if self.is_rejection:  # update cache with non-blocked traffic
                        self.rule_perf_table.update_cache(p.ip, p.malicious, num=1)

                    if p.malicious:
                        s.malicious_passed += 1
                    else:
                        s.benign_passed += 1
                        benign_passed_idx += 1

            if time_index_finished:
                logging.debug(f'benign per idx={benign_idx}')
                interval += 1
                self.blacklist_history.append(self.blacklist)
                self.time_index += 1
                if self.is_rejection:
                    logging.debug('end of time index; before rejection and updates')
                    self.rule_perf_table.print_rules()
                    self.rule_perf_table.print_cache()
                    estimated_benign_idx = benign_passed_idx + self.weight * sampled_benign_blocked_idx
                    # refresh cached rule performance
                    logging.debug('Refreshing cache.')
                    self.rule_perf_table.refresh_cache(s.thresh, estimated_benign_idx)
                    # delete rejected rules
                    logging.debug('Updating performances for table.')
                    self.rule_perf_table.refresh_table_perf(estimated_benign_idx)
                    logging.debug('Rejecting rules.')
                    for rejected_rule in self.rule_perf_table.get_rejected_rules(s.thresh):
                        start_ip, end_ip, hhh_len = rejected_rule
                        self.blacklist.remove_rule(start_ip, hhh_len)

                    logging.debug('end of time index; after rejection and updates')
                    self.rule_perf_table.print_rules()
                    self.rule_perf_table.print_cache()

                    # adapt average bl size for state and reward
                    s.blacklist_size = (s.blacklist_size * interval + len(self.blacklist)) / (interval + 1)

                    s.recall_per_idx.append((malicious_blocked_idx / malicious_idx) if malicious_idx != 0 else 1.0)
                    s.precision_per_idx.append((malicious_blocked_idx / blocked_idx) if blocked_idx != 0 else 1.0)
                    s.fpr_per_idx.append((benign_blocked_idx / benign_idx) if benign_idx != 0 else 0.0)
                    s.blacksize_per_idx.append(len(self.blacklist))

                    len_to_count = defaultdict(lambda: 0)
                    for start_ip, end_ip, hhh_len in self.rule_perf_table.cache.keys():
                        len_to_count[hhh_len] += 1

                    s.cache_per_idx.append(len_to_count if self.rule_perf_table.use_cache else None)

                benign_idx = 0
                malicious_idx = 0
                blocked_idx = 0
                malicious_blocked_idx = 0
                benign_passed_idx = 0
                sampled_benign_blocked_idx = 0
                benign_blocked_idx = 0

        logging.debug(f'final number of rules={len(self.blacklist)}')

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
