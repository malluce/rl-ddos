#!/usr/bin/env python
from ipaddress import IPv4Address

import gin
import numpy as np
from absl import logging
from collections import defaultdict

from gyms.hhh.cpp.hhhmodule import SketchHHH as HHHAlgo
from numpy.random import default_rng

from gyms.hhh.action import RejectionActionSpace
from gyms.hhh.images import ImageGenerator
from gyms.hhh.label import Label
from gyms.hhh.tables import RulePerformanceTable
from gyms.hhh.util import assert_hhh_asc_sorted


class Blacklist(object):

    def __init__(self, hhhs, sampling_weight):
        assert_hhh_asc_sorted(hhhs)

        self.initial_hhhs = np.array(hhhs)
        self.hhh_enabled = np.full((len(self.initial_hhhs)), True)  # bitmap: HHH in initial_hhhs enabled?

        self.match_counter = np.zeros_like(self.initial_hhhs, dtype=int)
        self.sample_counter = np.zeros_like(self.initial_hhhs, dtype=int)
        self.sample_remainder = default_rng().choice(range(0, sampling_weight), size=len(self.initial_hhhs))

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

    def should_be_sampled(self, ip, sampling_weight):
        # get indices of all enabled HHHs that cover ip
        ip_covering_hhhs = np.nonzero(self.hhh_map[ip, :])

        # longest prefix match
        lpm_hhh = ip_covering_hhhs[0][-1]

        # ip should be sampled if LPM rule reached sample counter
        do_sample = (self.match_counter[lpm_hhh] % sampling_weight == self.sample_remainder[lpm_hhh])

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


def remove_overlapping_rules(hhhs):
    """remove specific rules that are covered by more general rules"""
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


@gin.configurable
class Loop(object):
    ADDRESS_SPACE = 16
    ACTION_INTERVAL = 10
    SAMPLING_RATE = 0.3
    HHH_EPSILON = 0.0001

    def __init__(self, trace, create_state_fn, action_space, image_gen: ImageGenerator = None, epsilon=HHH_EPSILON,
                 sampling_rate=SAMPLING_RATE,
                 action_interval=ACTION_INTERVAL):
        self.create_state_fn = create_state_fn
        self.state = create_state_fn()
        self.trace = trace
        self.action_space = action_space
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

        self.is_rejection = isinstance(self.action_space, RejectionActionSpace)
        logging.info(f'is rejection: {self.is_rejection}')
        if self.is_rejection:
            self.rule_perf_table = RulePerformanceTable()

    def reset(self):
        self.blacklist = Blacklist([], self.weight)
        self.hhh.clear()
        self.state = self.create_state_fn()
        self.trace_ended = False
        self.time_index = 0

        # random first action 
        first_action = self.action_space.get_initialization()
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
        logging.debug(f'initial number of rules={len(self.blacklist.initial_hhhs)}')

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
            resolved_action = self.action_space.resolve(action)
            if len(resolved_action) == 2:  # only use phi and threshold
                s.phi, s.thresh = resolved_action
                s.min_prefix = 16
            elif len(resolved_action) == 3:  # use phi, threshold and min prefix
                s.phi, s.thresh, s.min_prefix = resolved_action
            else:
                raise ValueError('Unexpected resolved actions')
        else:
            s.phi, s.min_prefix = self.action_space.resolve(action)

    def pre_sample(self):  # Initialize the HHH instance with pre-sampled items
        self.state.trace_start = 0.0
        time_index_finished = False
        interval = 0
        while not (time_index_finished and interval == self.action_interval):
            p, time_index_finished = self.trace.next()
            self.hhh.update(p.ip)

            if time_index_finished:
                interval += 1
                self.blacklist_history.append(self.blacklist.to_serializable())
                self.time_index += 1
