import abc
from abc import ABC
from collections import OrderedDict, defaultdict, namedtuple
from ipaddress import IPv4Address

import gin
import numpy as np
from absl import logging

from gyms.hhh.label import Label
from gyms.hhh.util import assert_hhh_asc_sorted

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
class PerformanceTrackingWorstOffenderCache(WorstOffenderCache, PerformanceTracker):
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
        if isinstance(self.cache, PerformanceTrackingWorstOffenderCache):
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
