#!/usr/bin/env python
import abc
import sys
import time
from abc import ABC
from ipaddress import IPv4Address

import gin
import numpy as np
from absl import logging
from collections import Counter, OrderedDict, defaultdict, namedtuple
from gym.spaces import Box, Discrete, MultiDiscrete
from math import log2, log10, sqrt, exp

from gyms.hhh.cpp.hhhmodule import SketchHHH as HHHAlgo
from numpy.random import default_rng

from gyms.hhh.actionset import HafnerActionSet, RejectionActionSet
from gyms.hhh.images import ImageGenerator
from gyms.hhh.label import Label
from gyms.hhh.state import State
from gyms.hhh.obs import DistVol, DistVolStd


def assert_hhh_asc_sorted(hhhs):
    is_sorted = True
    val = hhhs[0].len if len(hhhs) > 0 else None
    for h in hhhs:
        if h.len < val:
            is_sorted = False
    assert is_sorted


RulePerformance = namedtuple('RulePerformance',
                             ('num_pkt', 'num_mal_pkt', 'num_ben_pkt', 'rule_perf'))


class PerformanceTracker:
    def __init__(self, metric):
        assert metric in ['fpr', 'prec']
        self.metric = metric
        logging.info(f'using {self.metric} metric for rule performance')

    def compute_rule_performance(self, n, nb, nm, total_benign):
        if self.metric == 'fpr':
            rule_fpr = nb / total_benign if total_benign != 0 else 0.0
            return max(0, 1 - rule_fpr)
        elif self.metric == 'prec':
            return nm / n if n != 0 else 1.0
        else:
            raise ValueError(f'Unexpected metric for rule performance: {self.metric}')


class WorstOffenderCache(ABC):
    @abc.abstractmethod
    def print(self):
        pass

    @abc.abstractmethod
    def add(self, start_ip, end_ip, hhh_len, **kwargs):
        """
        Adds a rule to the WOC.
        """
        pass

    @abc.abstractmethod
    def rule_recovery(self, **kwargs):
        """
        Recovers rules from WOC, called at the end of each time index.
        """
        pass

    @abc.abstractmethod
    def update(self, ip, is_malicious, num):
        """
        Updates rule performance for tracking caches.
        """
        pass


@gin.configurable
class TimedWorstOffenderCache(WorstOffenderCache):
    BACKGROUND_DECREMENT = 0.5
    ACTIVE_DECREMENT = 1
    INITIAL_TIMER = 10

    def __init__(self, background_decrement=BACKGROUND_DECREMENT, active_decrement=ACTIVE_DECREMENT,
                 initial_timer=INITIAL_TIMER):
        self.active_timers = {}
        self.background_timers = {}
        self.background_decrement = background_decrement
        self.active_decrement = active_decrement
        self.initial_timer = initial_timer

    @property
    def cache_entries(self):
        return self.active_timers

    def print(self):
        logging.debug(f'active timers:')
        for (start_ip, end_ip, hhh_len) in self.active_timers:
            logging.debug(
                f'  {str(IPv4Address(start_ip))}/{hhh_len}: {self.active_timers[(start_ip, end_ip, hhh_len)]}')

        logging.debug(f'background timers:')
        for (start_ip, end_ip, hhh_len) in self.background_timers:
            logging.debug(
                f'  {str(IPv4Address(start_ip))}/{hhh_len}: {self.background_timers[(start_ip, end_ip, hhh_len)]}')

    def add(self, start_ip, end_ip, hhh_len, **kwargs):
        assert (start_ip, end_ip, hhh_len) not in self.active_timers
        if (start_ip, end_ip, hhh_len) in self.background_timers:
            logging.debug(
                f'doubling {str(IPv4Address(start_ip))}/{hhh_len}: {self.background_timers[(start_ip, end_ip, hhh_len)]}->{2 * self.background_timers[(start_ip, end_ip, hhh_len)]}')

            initial_timer = max(self.initial_timer,
                                self.initial_timer + self.background_timers[(start_ip, end_ip, hhh_len)])
        else:
            initial_timer = self.initial_timer

        self.background_timers[(start_ip, end_ip, hhh_len)] = initial_timer
        self.active_timers[(start_ip, end_ip, hhh_len)] = initial_timer

    def rule_recovery(self, **kwargs):
        if logging.level_debug():
            logging.debug('===== before recovery =====')
            self.print()

        for (start_ip, end_ip, hhh_len) in list(self.active_timers.keys()):
            self.active_timers[(start_ip, end_ip, hhh_len)] -= self.active_decrement
            if self.active_timers[(start_ip, end_ip, hhh_len)] <= 0:
                self.active_timers.pop((start_ip, end_ip, hhh_len))
                logging.debug(f'**removing active timer for{str(IPv4Address(start_ip))}/{hhh_len}')

        for (start_ip, end_ip, hhh_len) in list(self.background_timers.keys()):
            self.background_timers[(start_ip, end_ip, hhh_len)] -= self.background_decrement
            if self.background_timers[(start_ip, end_ip, hhh_len)] <= 0:
                logging.debug(f'**removing background timer for{str(IPv4Address(start_ip))}/{hhh_len}')
                self.background_timers.pop((start_ip, end_ip, hhh_len))

        if logging.level_debug():
            logging.debug('===== after recovery =====')
            self.print()

    def update(self, ip, is_malicious, num):
        # no performance tracking -> no need to update anything here
        pass


@gin.configurable
class TrackingWorstOffenderCache(WorstOffenderCache, PerformanceTracker):
    CACHE_CAP = 100

    def __init__(self, metric, capacity=CACHE_CAP):
        self.capacity = np.inf if capacity == 'inf' else capacity
        logging.info(f'cache capacity={self.capacity})')

        # stores (start IP, end IP, len) -> RulePerformance, both IPs are inclusive
        self.cache_entries = OrderedDict()

        super().__init__(metric)

    def print(self):
        logging.debug(f'all cache entries: (total={len(self.cache_entries)})')
        len_to_count = defaultdict(lambda: 0)
        for start_ip, end_ip, hhh_len in self.cache_entries.keys():
            len_to_count[hhh_len] += 1
            logging.debug(
                f' {str(IPv4Address(start_ip))}/{hhh_len} (perf={self.cache_entries[(start_ip, end_ip, hhh_len)]})')
        for l in sorted(len_to_count.keys()):
            logging.debug(f' {l}:{len_to_count[l]}')

    def add(self, start_ip, end_ip, hhh_len, **kwargs):
        assert 'rule_perf' in kwargs
        assert len(self.cache_entries) <= self.capacity

        if len(self.cache_entries) == self.capacity:
            # replace old rule if cache full and new rule is worse than current best rule in cache
            cache_by_performance = sorted(self.cache_entries.items(), key=lambda x: x[1].rule_perf)
            # best performing rule in cache
            best_rule, best_performance = cache_by_performance[-1]
            if kwargs['rule_perf'] <= best_performance.rule_perf:
                logging.debug(f' Replacement of {best_rule, best_performance}')
                self.cache_entries.pop(best_rule)
                self.cache_entries[(start_ip, end_ip, hhh_len)] = RulePerformance(0, 0, 0, 0)
            else:
                logging.debug(f' Not replacing, performance better')
        elif len(self.cache_entries) < self.capacity:
            logging.debug(f' Cache not full, just adding')
            self.cache_entries[(start_ip, end_ip, hhh_len)] = RulePerformance(0, 0, 0, 0)

        assert len(self.cache_entries) <= self.capacity

    def rule_recovery(self, **kwargs):
        assert 'perf_thresh' in kwargs and 'total_benign' in kwargs
        perf_thresh, total_benign = kwargs['perf_thresh'], kwargs['total_benign']
        for rule in list(self.cache_entries):
            # update perf
            n, nm, nb, old_perf = self.cache_entries[rule]
            if n == 0:  # no packet applied to this rule, delete it from cache
                new_perf = 1.0
            else:
                new_perf = self.compute_rule_performance(n=n, nm=nm, nb=nb, total_benign=total_benign)

            if new_perf >= perf_thresh:
                logging.debug(
                    f' removing rule {str(IPv4Address(rule[0]))}/{rule[2]} from cache because it got better {n, nm, nb, new_perf}')
                self.cache_entries.pop(rule)  # delete rules that have better performance
            else:
                self.cache_entries[rule] = RulePerformance(0, 0, 0, new_perf)  # otherwise reset counters

    def update(self, ip, is_malicious, num):
        for start_ip, end_ip, hhh_len in self.cache_entries.keys():
            if start_ip <= ip <= end_ip:
                self._update_single_rule(start_ip, end_ip, hhh_len, is_malicious, num)

    def _update_single_rule(self, start_ip, end_ip, hhh_len, is_malicious, num):
        # current perf
        n, nm, nb, perf = self.cache_entries[(start_ip, end_ip, hhh_len)]
        # update rule performance entries
        n += num
        if is_malicious:
            nm += num
        else:
            nb += num
        # update counters, keep old performance
        self.cache_entries[(start_ip, end_ip, hhh_len)] = RulePerformance(n, nm, nb, perf)

    # only for testing purposes, updates only LPM from WOC; performs poorly
    def update_lpm(self, ip, is_malicious, num):
        for start_ip, end_ip, hhh_len in sorted(self.cache_entries.keys(), key=lambda x: x[2], reverse=False):
            if start_ip <= ip <= end_ip:
                self._update_single_rule(start_ip, end_ip, hhh_len, is_malicious, num)
                return


@gin.configurable
class RulePerformanceTable(PerformanceTracker):

    def __init__(self, use_cache, metric, cache_class):
        # stores (start IP, end IP, len) -> (enabled, RulePerformance) both IPs are inclusive
        super().__init__(metric)
        self.table = OrderedDict()

        self.use_cache = use_cache
        logging.info(f'using cache: {self.use_cache}')
        if self.use_cache:
            self.cache = cache_class()

    def print_rules(self):
        if logging.get_verbosity() == 'debug':
            logging.debug('all rules: ')
            for start_ip, end_ip, hhh_len in self.table.keys():
                logging.debug(
                    f' {str(IPv4Address(start_ip))}/{hhh_len} (perf={self.table[(start_ip, end_ip, hhh_len)]})')

    def print_cache(self):
        if self.use_cache and logging.get_verbosity() == 'debug':
            self.cache.print()

    def set_rules(self, hhhs):
        assert_hhh_asc_sorted(hhhs)
        self.table = OrderedDict()
        for hhh in hhhs:
            start_ip = hhh.id
            end_ip = start_ip + Label.subnet_size(hhh.len) - 1
            self.table[(start_ip, end_ip, hhh.len)] = (True, RulePerformance(0, 0, 0, None))

    def filter_hhhs(self, hhhs):
        if not self.use_cache:
            return hhhs
        # only return hhhs that are not in cache
        result = []
        for hhh in hhhs:
            start_ip = hhh.id
            end_ip = start_ip + Label.subnet_size(hhh.len) - 1
            if (start_ip, end_ip, hhh.len) not in self.cache.cache_entries:
                result.append(hhh)
            else:
                logging.debug(f'not applying HHH {str(IPv4Address(start_ip))}/{hhh.len}, because in cache')
        return result

    # deprecated (updates not only LPM match, but all)
    def update_rpt_old(self, id, is_malicious, num):
        for start_ip, end_ip, hhh_len in self.table.keys():
            if start_ip <= id <= end_ip:
                # current perf
                enable, rperf = self.table[(start_ip, end_ip, hhh_len)]
                if not enable:
                    continue

                n, nm, nb, perf = rperf
                # update rule performance entries
                n += num

                if is_malicious:
                    nm += num
                else:
                    nb += num

                # update counters, keep old performance
                self.table[(start_ip, end_ip, hhh_len)] = (enable, RulePerformance(n, nm, nb, perf))

    def update_rpt(self, lpm_hhh, is_malicious, num):
        """
        Update the counters for RPT rule based on ID/Tag lpm_hhh and IDS classification is_malicious.
        """
        start_ip, end_ip, hhh_len = list(self.table.keys())[lpm_hhh]

        # current perf
        enabled, rule_perf = self.table[(start_ip, end_ip, hhh_len)]

        if not enabled:
            return

        n, nm, nb, perf = rule_perf

        # update rule performance counters
        n += num

        if is_malicious:
            nm += num
        else:
            nb += num

        self.table[(start_ip, end_ip, hhh_len)] = (enabled, RulePerformance(n, nm, nb, perf))

    def refresh_table_perf(self, total_benign):
        """
        Update the performance of each TABLE RULE by computing performance based on counters. Called each time idx.
        :param total_benign: number of benign packets of the time idx
        """
        for rule in self.table.keys():
            enabled, perf = self.table[rule]

            if not enabled:
                continue

            n, nm, nb, _ = perf
            rule_perf = self.compute_rule_performance(n=n, nm=nm, nb=nb, total_benign=total_benign)
            self.table[rule] = (enabled, RulePerformance(n, nm, nb, rule_perf))

    # only for testing purposes, updates only LPM from WOC; performs poorly
    def update_cache_lpm(self, ip, is_malicious, num):
        if not self.use_cache:
            return
        self.cache.update_lpm(ip, is_malicious, num)

    def update_cache(self, ip, is_malicious, num):
        if not self.use_cache:
            return
        self.cache.update(ip, is_malicious, num)

    def rule_recovery(self, perf_thresh, total_benign):
        """
        Refreshes the rule performance for each cached rule by using the per-rule counters from the previous time idx.
        Afterwards recovers rules whose performance now outdoes the threshold.
        :param perf_thresh: the performance threshold
        :param total_benign: the number of total estimated benign packets in the time index
        """
        if not self.use_cache:
            return
        if isinstance(self.cache, TrackingWorstOffenderCache):
            kwargs = {
                'perf_thresh': perf_thresh,
                'total_benign': total_benign
            }
        else:
            kwargs = {}
        self.cache.rule_recovery(**kwargs)

    def reject_rules(self, performance_threshold):
        for start_ip, end_ip, hhh_len in list(self.table):
            enable, rperf = self.table[(start_ip, end_ip, hhh_len)]
            if not enable:
                continue
            rule_perf = rperf.rule_perf
            if rule_perf < performance_threshold:
                if self.use_cache:
                    logging.debug(
                        f' rejecting rule {str(IPv4Address(start_ip))}/{hhh_len} (perf={rule_perf}); adding to cache')
                    self.cache.add(start_ip, end_ip, hhh_len, rule_perf=rule_perf)

                # disable rule from table and return it
                self.table[(start_ip, end_ip, hhh_len)] = (False, self.table[(start_ip, end_ip, hhh_len)][1])
                yield start_ip, end_ip, hhh_len


class Blacklist(object):

    def __init__(self, hhhs, sampling_weight):
        assert_hhh_asc_sorted(hhhs)

        self.initial_hhhs = np.array(hhhs)
        self.hhh_enabled = np.full((len(self.initial_hhhs)), True)  # bitmap: HHH in initial_hhhs enabled?

        self.match_counter = np.zeros_like(self.initial_hhhs, dtype=int)
        self.sample_counter = np.zeros_like(self.initial_hhhs, dtype=int)
        self.sample_modulus = default_rng().choice(range(0, sampling_weight), size=len(self.initial_hhhs))

        self._compute_filter_bitmap()

    def get_enabled_hhhs(self):
        return self.initial_hhhs[self.hhh_enabled]

    def _compute_filter_bitmap(self):
        # set up bitmap that indicates whether each IP gets blocked or not
        self.filter_bitmap = np.full((2 ** Loop.ADDRESS_SPACE), False)

        # set up bit-matrix that indicates whether each IP is covered by HHH
        self.hhh_map = np.full((2 ** Loop.ADDRESS_SPACE, len(self.initial_hhhs)), False)

        for idx, h in enumerate(self.initial_hhhs):
            if self.hhh_enabled[idx]:
                start = h.id
                end = start + Label.subnet_size(h.len)
                self.filter_bitmap[start:end] = True
                self.hhh_map[start:end, idx] = True

    def covers(self, ip):
        return self.filter_bitmap[ip]

    # deprecated, samples based on n_r and m_r of ALL matching rules (not only LPM)
    def should_be_sampled_old(self, ip, sampling_weight):
        # get all HHH indices that cover ip
        ip_covering_hhhs = np.nonzero(self.hhh_map[ip, :])

        if ip_covering_hhhs[0].shape[0] > 1:
            logging.debug(f' {str(IPv4Address(ip))} matches..')
            for h in self.initial_hhhs[ip_covering_hhhs]:
                logging.debug(f'  - {str(IPv4Address(h.id))}/{h.len}')

        # ip should be sampled if at least one matching HHH reached sample counter
        do_sample = np.max(
            self.match_counter[ip_covering_hhhs] % sampling_weight == self.sample_modulus[ip_covering_hhhs])

        if do_sample:
            self.sample_counter[ip_covering_hhhs] += 1

        # increase match counter
        self.match_counter[ip_covering_hhhs] += 1

        lpm = None

        return do_sample, lpm

    def should_be_sampled(self, ip, sampling_weight):
        # get indices of all enabled HHHs that cover ip
        ip_covering_hhhs = np.nonzero(self.hhh_map[ip, :])

        # longest prefix match
        lpm_hhh = ip_covering_hhhs[0][-1]

        # ip should be sampled if LPM rule reached sample counter
        do_sample = (self.match_counter[lpm_hhh] % sampling_weight == self.sample_modulus[lpm_hhh])

        if do_sample:
            self.sample_counter[lpm_hhh] += 1

        # increase match counter
        self.match_counter[lpm_hhh] += 1

        return do_sample, lpm_hhh  # lpm_hhh serves as rule ID

    # only for debugging
    def increase_counters(self, ip, sampled):
        # get all HHH indices that cover ip
        ip_covering_hhhs = np.nonzero(self.hhh_map[ip, :] == np.max(self.hhh_map[ip, :]))

        # increase match counter
        self.match_counter[ip_covering_hhhs] += 1

        if sampled:
            self.sample_counter[ip_covering_hhhs] += 1

    def __len__(self):
        return len(self.initial_hhhs[self.hhh_enabled])

    def to_serializable(self):
        return [{'id': h.id, 'len': h.len, 'hi': h.hi, 'lo': h.lo}
                for h in self.initial_hhhs[self.hhh_enabled]]

    def remove_rule(self, ip_start, hhh_len):
        for idx, h in enumerate(self.initial_hhhs):  # iterate over all HHHs that are still enabled
            if self.hhh_enabled[idx]:
                if h.id == ip_start and h.len == hhh_len:
                    self.hhh_enabled[idx] = False
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
        self.blacklist = Blacklist([], self.weight)
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
        self.blacklist = Blacklist([], self.weight)
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
            # overlapping rules can be safely removed because rules do not change until next time step
            hhhs = remove_overlapping_rules(hhhs)

        self.blacklist = Blacklist(hhhs, self.weight)
        s.blacklist_size = len(self.blacklist)
        s.blacklist_coverage = self._calc_blacklist_coverage(hhhs)
        if self.use_hhh_distvol:
            self._calc_hhh_distance_metrics(hhhs, s)
        logging.debug(f'initial number of rules={len(self.blacklist.initial_hhhs)}')

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

                    sample, lpm_hhh = self.blacklist.should_be_sampled(p.ip, self.weight)
                    if sample:
                        s.samples += 1
                        self.hhh.update(p.ip, self.weight)
                        if self.is_rejection:
                            self.rule_perf_table.update_rpt(lpm_hhh, p.malicious, num=self.weight)
                            self.rule_perf_table.update_cache(p.ip, p.malicious, num=self.weight)
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
                self.blacklist_history.append(self.blacklist.to_serializable())
                self.time_index += 1
                if self.is_rejection:
                    logging.debug('end of time index; before rejection and updates')
                    self.rule_perf_table.print_rules()
                    self.rule_perf_table.print_cache()
                    estimated_benign_idx = benign_passed_idx + self.weight * sampled_benign_blocked_idx
                    # refresh cached rule performance
                    logging.debug('Refreshing cache.')
                    self.rule_perf_table.rule_recovery(s.thresh, estimated_benign_idx)
                    # delete rejected rules
                    logging.debug('Updating performances for table.')
                    self.rule_perf_table.refresh_table_perf(estimated_benign_idx)
                    logging.debug('Rejecting rules.')
                    for rejected_rule in self.rule_perf_table.reject_rules(s.thresh):
                        start_ip, end_ip, hhh_len = rejected_rule
                        self.blacklist.remove_rule(start_ip, hhh_len)

                    logging.debug('end of time index; after rejection and updates')
                    self.rule_perf_table.print_rules()
                    self.rule_perf_table.print_cache()

                    self.save_metrics_to_state(benign_blocked_idx, benign_idx, blocked_idx, interval,
                                               malicious_blocked_idx,
                                               malicious_idx, s)

                benign_idx = 0
                malicious_idx = 0
                blocked_idx = 0
                malicious_blocked_idx = 0
                benign_passed_idx = 0
                sampled_benign_blocked_idx = 0
                benign_blocked_idx = 0

        if logging.level_debug():
            logging.debug(f'final number of rules={len(self.blacklist)}')
            logging.debug(f'samples per timestep={s.samples}')
            sample_rates = np.true_divide(self.blacklist.sample_counter, self.blacklist.match_counter)
            # logging.debug(
            #    f'match count, sample rates: {np.array(list(zip(self.blacklist.match_counter, sample_rates)))}')
            logging.debug(
                f'sample rates mean: {np.mean(sample_rates)}, stddev: {np.std(sample_rates)}')
            logging.debug(
                f'sample rates 0.1q: {np.quantile(sample_rates, 0.1) if len(sample_rates) > 0 else None}, 0.2q: {np.quantile(sample_rates, 0.2) if len(sample_rates) > 0 else None}, 0.3q: {np.quantile(sample_rates, 0.3) if len(sample_rates) > 0 else None}')

        s.complete()

        if self.image_gen is not None:
            s.hhh_image = self.image_gen.generate_image(hhh_algo=self.hhh, hhh_query_result=None)

        if self.time_index == self.trace.trace_sampler.maxtime + 1:  # rate grid ended
            self.trace_ended = True

        return self.trace_ended, self.state, self.blacklist_history

    def save_metrics_to_state(self, benign_blocked_idx, benign_idx, blocked_idx, interval, malicious_blocked_idx,
                              malicious_idx,
                              s):
        # adapt average bl size for state and reward
        s.blacklist_size = (s.blacklist_size * interval + len(self.blacklist)) / (interval + 1)
        s.recall_per_idx.append((malicious_blocked_idx / malicious_idx) if malicious_idx != 0 else 1.0)
        s.precision_per_idx.append((malicious_blocked_idx / blocked_idx) if blocked_idx != 0 else 1.0)
        s.fpr_per_idx.append((benign_blocked_idx / benign_idx) if benign_idx != 0 else 0.0)
        s.blacksize_per_idx.append(len(self.blacklist))
        if self.rule_perf_table.use_cache:
            len_to_count = defaultdict(lambda: 0)
            for start_ip, end_ip, hhh_len in self.rule_perf_table.cache.cache_entries.keys():
                len_to_count[hhh_len] += 1

            s.cache_per_idx.append(len_to_count if self.rule_perf_table.use_cache else None)
        else:
            s.cache_per_idx = None

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
                self.blacklist_history.append(self.blacklist.to_serializable())
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
