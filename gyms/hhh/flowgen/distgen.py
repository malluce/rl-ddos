#!/usr/bin/env python

import math
import numpy as np

from gyms.hhh.cpp.hhhmodule import HIERARCHY_SIZE

from gyms.hhh.packet import Packet


class Sampler(object):

    def __init__(self):
        pass

    def sample(self, num_samples):
        pass


class UniformSampler(Sampler):

    def __init__(self, start, end, seed=None):
        super().__init__()
        self.rng = np.random.default_rng(seed=seed)
        self.start = start
        self.end = end

    def sample(self, num_samples):
        samples = self.rng.uniform(self.start, self.end, num_samples)
        samples = samples.astype(int)

        return samples


class WeibullSampler(Sampler):

    @staticmethod
    def quantile(x, weibull_a):
        return (-math.log(1 - (x / 100))) ** (1 / weibull_a)

    def __init__(self, weibull_a, scale, seed=None):
        super().__init__()
        self.rng = np.random.default_rng(seed=seed)
        self.a = weibull_a
        self.scale = scale

    def sample(self, num_samples):
        samples = self.scale * self.rng.weibull(self.a, size=(num_samples))
        samples = samples.astype(int)

        return samples


class NormalSampler(Sampler):

    def __init__(self, mean, stddev, min=None, max=None, seed=None):
        super().__init__()
        self.rng = np.random.default_rng(seed=seed)
        self.mean = mean
        self.stddev = stddev
        self.min = min
        self.max = max

    def _crop(self, samples):
        if self.min is not None:
            samples = np.where(samples < self.min, self.min, samples)

        if self.max is not None:
            samples = np.where(samples > self.max, self.max, samples)

        return samples

    def sample(self, num_samples):
        samples = self.rng.normal(self.mean, self.stddev, num_samples)
        samples = samples.astype(int)

        return self._crop(samples)


class HHHEntry(object):

    def __init__(self, other):
        self.id = other.id
        self.len = other.len
        self.size = 1 << (HIERARCHY_SIZE - self.len)
        self.end = self.id + self.size
        self.lo = other.lo
        self.hi = other.hi

    def contains(self, other):
        return (self.size > other.size and self.id <= other.id
                and self.end >= other.end)


class FlowGroupSampler(object):

    def __init__(self, num_flows, start_sampler, duration_sampler,
                 address_sampler, attack=False):
        self.num_flows = num_flows
        self.start_sampler = start_sampler
        self.duration_sampler = duration_sampler
        self.address_sampler = address_sampler
        self.attack = attack

    def sample(self):
        """
        Generates flows according to the provided samplers.

        :returns: np array specifying the generated flows
         ('start','end', 'duration', 'addr', 'rate', 'attack'). Rate is always 1. Attack can be 0 or 1.
        """
        addr = self.address_sampler.sample(self.num_flows)
        start = self.start_sampler.sample(self.num_flows)
        duration = self.duration_sampler.sample(self.num_flows)
        end = start + duration

        # Constant data rate for all flows
        rate = np.ones_like(start)

        if self.attack:
            attk = np.ones_like(start)
        else:
            attk = np.zeros_like(start)

        return np.array((start, end, duration, addr, rate, attk)).transpose()


class TraceSampler(object):

    def __init__(self, flowsamplers, maxtime=None):
        self.flowsamplers = flowsamplers
        self.maxtime = maxtime
        self.benign_flows = None
        self.attack_flows = None
        self.flows = None
        self.rate_grid = None
        self.attack_grid = None
        self.num_samples = None

    def init_flows(self):
        """
        Samples flows from flowsamplers to flows, sets flowgrid (shape (maxtime, 2*maxaddr))
        and num_samples (number of non-zero entries in flowgrid).
        """
        self.flows = np.concatenate([s.sample() for s in self.flowsamplers])

        maxtime = self.flows[:, 1].max()
        maxaddr = self.flows[:, 3].max()

        if self.maxtime is not None:
            maxtime = min(self.maxtime, maxtime)

        self.rate_grid = np.zeros((maxtime, maxaddr + 1), dtype=int)
        self.attack_grid = np.zeros((maxtime, maxaddr + 1), dtype=int)

        for f in self.flows:
            # Assign total rate to rate grid from flow start to flow end
            # for the respective address
            self.rate_grid[f[0]: f[1] + 1, f[3]] = f[4]
            # Same with attack rate
            self.attack_grid[f[0]: f[1] + 1, f[3]] = f[5]

        self.num_samples = self.rate_grid.sum()
        self.num_attack_samples = self.attack_grid.sum()

    def samples(self):
        for time_index in range(len(self.rate_grid)):
            time_index_total_rates = self.rate_grid[time_index]
            time_index_attack_rates = self.attack_grid[time_index]
            addrseq = np.nonzero(time_index_total_rates)[0]
            # Avoid strict order of addresses, since it would model
            # atypical traffic behavior and impact HHH accuracy.
            addrseq = np.random.permutation(addrseq)
            addrseq_len = len(addrseq)

            for i in range(addrseq_len):
                addr = addrseq[i]
                addr_total_rate = time_index_total_rates[addr]
                addr_attack_rate = time_index_attack_rates[addr]
                time_index_finished = i == addrseq_len - 1

                for j in range(addr_total_rate):
                    malicious = j < addr_attack_rate
                    address_finished = j == addr_total_rate - 1
                    yield (Packet(addr.item(), malicious.item()),
                           time_index_finished and address_finished)
