#!/usr/bin/env python

import math
import numpy as np

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


class ChoiceSampler(Sampler):
    def __init__(self, population, replace, seed=None):
        super().__init__()
        self.rng = np.random.default_rng(seed=seed)
        self.population = population
        self.replace = replace

    def sample(self, num_samples):
        return self.rng.choice(self.population, num_samples, replace=self.replace)


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
            np.clip(samples, a_min=self.min, a_max=None, out=samples)

        if self.max is not None:
            np.clip(samples, a_min=None, a_max=self.max, out=samples)

        return samples

    def sample(self, num_samples):
        samples = self.rng.normal(self.mean, self.stddev, num_samples)
        samples = samples.astype(int)

        return self._crop(samples)


class FlowGroupSampler(object):

    def __init__(self, num_flows, start_sampler, duration_sampler,
                 address_sampler, attack=False, rate_sampler=None):
        self.num_flows = num_flows
        self.start_sampler = start_sampler
        self.duration_sampler = duration_sampler
        self.address_sampler = address_sampler
        self.attack = attack
        self.rate_sampler = rate_sampler

    def sample(self):
        """
        Generates flows according to the provided samplers.

        :returns: np array specifying the generated flows
         ('start','end', 'duration', 'addr', 'rate', 'attack').
        """
        addr = self.address_sampler.sample(self.num_flows)
        start = self.start_sampler.sample(self.num_flows)
        duration = self.duration_sampler.sample(self.num_flows)
        end = start + duration

        if self.rate_sampler is None:
            rate = np.ones_like(start)
        else:
            rate = self.rate_sampler.sample(self.num_flows)

        if self.attack:
            attk = rate
        else:
            attk = np.zeros_like(start)

        return np.array((start, end, duration, addr, rate, attk)).transpose()


class TraceSampler(object):
    """Samples flows from provided FlowGroupSamplers."""

    @staticmethod
    def load(flow_grid, rate_grid, attack_grid):
        sampler = TraceSampler(None)
        sampler.flows = flow_grid
        sampler.rate_grid = rate_grid
        sampler.attack_grid = attack_grid
        sampler.maxtime = sampler.flows[:, 1].max()
        sampler.maxaddr = sampler.flows[:, 3].max()
        sampler.num_samples = sampler.rate_grid.sum()
        sampler.num_attack_samples = sampler.attack_grid.sum()
        return sampler

    def __init__(self, flowsamplers, maxtime=None):
        self.flowsamplers = flowsamplers
        self.maxtime = maxtime
        self.flows = None
        self.rate_grid = None
        self.attack_grid = None
        self.num_samples = None

    def init_flows(self):
        """
        Prepares traffic by sampling from the given distributions, i.e. flowsamplers.
        """
        self.flows = np.concatenate([s.sample() for s in self.flowsamplers])

        maxtime = self.flows[:, 1].max()
        maxaddr = self.flows[:, 3].max()

        if self.maxtime is not None:
            maxtime = min(self.maxtime, maxtime)

        self.rate_grid = np.zeros((maxtime + 1, maxaddr + 1), dtype=int)
        self.attack_grid = np.zeros((maxtime + 1, maxaddr + 1), dtype=int)

        for f in self.flows:
            flow_start = f[0] if f[0] >= 0 else 0  # flows that virtually started before trace start at trace
            flow_end = f[1] if f[1] >= 0 else 0  # avoid flow_start=0, flow_end=-x

            # Assign total rate to rate grid from flow start to flow end
            # for the respective address
            self.rate_grid[flow_start: flow_end + 1, f[3]] += f[4]
            # Same with attack rate
            self.attack_grid[flow_start: flow_end + 1, f[3]] += f[5]

        self.num_samples = self.rate_grid.sum()
        self.num_attack_samples = self.attack_grid.sum()

    def samples(self):
        """
        Samples the next packet from the generated traffic, returns (packet, time_index_finished, address_finished).
        """
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
