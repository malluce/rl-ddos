#!/usr/bin/env python

import abc
import argparse
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgrid
import numpy as np
import pandas as pd
import time

from gyms.hhh.cpp.hhhmodule import SketchHHH as HHHAlgo
from gyms.hhh.cpp.hhhmodule import HIERARCHY_SIZE
from gyms.hhh.label import Label


def cmdline():
    argp = argparse.ArgumentParser(description='Simulated HHH generation')

    argp.add_argument('--benign', type=int, default=2000,
                      help='Number of benign flows')
    argp.add_argument('--attack', type=int, default=1000,
                      help='Number of attack flows')
    argp.add_argument('--steps', type=int, default=7500,
                      help='Number of time steps')
    argp.add_argument('--maxaddr', type=int, default=0xfff,
                      help='Size of address space')
    argp.add_argument('--epsilon', type=float, default=.005,
                      help='Error bound')
    argp.add_argument('--phi', type=float, default=.02,
                      help='Query threshold')
    argp.add_argument('--minprefix', type=int, default=0,
                      help='Minimum prefix length')

    return argp.parse_args()


class Sampler(object):

    def __init__(self):
        pass

    def sample(self, num_samples):
        pass


class UniformSampler(Sampler):

    def __init__(self, start, end):
        super().__init__()
        self.start = start
        self.end = end

    def sample(self, num_samples):
        samples = np.random.uniform(self.start, self.end, num_samples)
        samples = samples.astype(int)

        return samples


class WeibullSampler(Sampler):

    @staticmethod
    def quantile(x, weibull_a):
        return (-math.log(1 - (x / 100))) ** (1 / weibull_a)

    def __init__(self, weibull_a, scale):
        super().__init__()
        self.a = weibull_a
        self.scale = scale

    def sample(self, num_samples):
        samples = self.scale * np.random.weibull(self.a, size=(num_samples))
        samples = samples.astype(int)

        return samples


class NormalSampler(Sampler):

    def __init__(self, mean, stddev, min=None, max=None):
        super().__init__()
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
        samples = np.random.normal(self.mean, self.stddev, num_samples)
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
        addr = self.address_sampler.sample(self.num_flows)
        start = self.start_sampler.sample(self.num_flows)
        duration = self.duration_sampler.sample(self.num_flows)
        end = start + duration

        if self.attack:
            attk = np.ones_like(start)
            rate = np.ones_like(start)
        else:
            attk = np.zeros_like(start)
            rate = np.ones_like(start)

        return pd.DataFrame(
            np.array((start, end, duration, addr, rate, attk)).transpose(),
            columns=['start', 'end', 'duration', 'addr', 'rate', 'attack'])


class Timer(object):

    def __init__(self):
        self.elapsed_time = 0
        self.start_time = 0

    def start(self):
        self.start_time = time.time()
        return self

    def stop(self):
        self.elapsed_time += time.time() - self.start_time
        return self

    def accumulate(self, func):
        self.start()
        func()
        self.stop()

    def __str__(self):
        return '{:s}'.format(str(self.elapsed_time))


class TraceSampler(object):

    def __init__(self, flowsamplers, maxtime=None):
        self.flowsamplers = flowsamplers
        self.maxtime = maxtime
        self.benign_flows = None
        self.attack_flows = None
        self.flows = None
        self.flowgrid = None
        self.num_samples = None

    def _merge_flows(self, flows):
        if len(flows) == 1:
            return flows[0]

        f = pd.concat(flows)
        f.sort_values(['start'], inplace=True)
        f.reset_index(drop=True, inplace=True)

        return f

    def init_flows(self):
        self.flows = pd.concat([s.sample() for s in self.flowsamplers])

        maxaddr = self.flows['addr'].max()
        maxtime = self.flows['end'].max()

        if self.maxtime is not None:
            maxtime = min(self.maxtime, maxtime)

        rg = pd.DataFrame(np.zeros((maxtime, (maxaddr + 1))),
                          columns=range(maxaddr + 1), dtype=int)
        ag = pd.DataFrame(np.zeros((maxtime, (maxaddr + 1))),
                          columns=range(maxaddr + 1), dtype=int)

        for _, f in self.flows.iterrows():
            timespan = ((rg.index >= f['start']) & (rg.index <= f['end']))
            rg.loc[timespan, f['addr']] += f['rate']
            # TODO: handle overlap of benign and attack flows
            ag.loc[timespan, f['addr']] += f['attack']

        self.flowgrid = pd.concat({'rate': rg, 'attack': ag}, axis=1)
        self.num_samples = np.count_nonzero(self.flowgrid['rate'])

    def samples(self):
        step_finished = False

        for t in self.flowgrid.index:
            step = self.flowgrid.loc[self.flowgrid.index == t]
            sr = step.loc[t, ('rate')]
            sa = step.loc[t, ('attack')]

            for addr in sr.index[(sr.values > 0)]:
                yield addr, sr[addr], sa[addr], step_finished
                step_finished = False

            step_finished = True

    def next(self):
        for addr, rate, attack, step_finished in self.samples():
            yield addr, rate, attack, step_finished

    def __next__(self):
        return self.next()


class ProgressBar(object):

    def __init__(self, maxprogress, step=100, displaysteps=10):
        self.maxprogress = maxprogress
        self.step = step
        self.displaysteps = displaysteps
        self.i = 0

    def increment(self):
        self.i += 1

        return (self.i % (self.maxprogress / self.step) == 0
                or self.i + 1 == self.maxprogress)

    def update(self):
        p = self.i / self.maxprogress
        e = '\n' if self.i + 1 == self.maxprogress else ''
        fmt = '\r [ {:' + str(self.displaysteps) + 's} ] {:3d}%'
        print(fmt.format('#' * round(self.displaysteps * p), round(100 * p)), end=e)


def playthrough(traffic_sampler, epsilon, phi, minprefix):
    bar = ProgressBar(traffic_sampler.maxtime)
    res = pd.DataFrame(np.zeros_like(traffic_sampler.flowgrid['rate']),
                       columns=traffic_sampler.flowgrid['rate'].columns)
    t = 0

    # play through non-zero entries in flowgrid
    for addr, rate, attack, step_finished in traffic_sampler.samples():
        if t % 5 == 0:
            h = HHHAlgo(epsilon)

        # update hhh algorithm
        h.update(addr, rate)

        # perform query and calculate hhh coverage
        if step_finished:
            t += 1
            hhhs = [HHHEntry(_) for _ in h.query(phi, minprefix)]
            hhhs = sorted(hhhs, key=lambda x: x.size)

            for i in range(len(hhhs)):
                x = hhhs[i]

                for y in hhhs[i + 1:]:
                    if y.contains(x):
                        y.hi = max(0, y.hi - x.hi)
                        y.lo = max(0, y.lo - x.lo)

                span = ((res.columns >= x.id) & (res.columns < (x.end)))
                res.loc[t, span] += x.hi

            if bar.increment():
                bar.update()

    return res


def plot(flows, flowgrid, hhhgrid):
    fig = plt.figure(figsize=(16, 8))
    gs = mgrid.GridSpec(3, 3)

    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[:, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 2])
    ax5 = fig.add_subplot(gs[2, 2])

    titlesize = 11
    labelsize = 10
    benign_color = 'lightsteelblue'
    attack_color = 'crimson'
    combined_color = 'darkgrey'

    benign_flows = flows[flows['attack'] == 0]
    attack_flows = flows[flows['attack'] > 0]
    rategrid = flowgrid['rate']
    attkgrid = flowgrid['attack']

    ax1.set_title('Traffic intensity', fontsize=titlesize)
    ax1.set_xlabel('Address space (percent)', fontsize=labelsize)
    ax1.set_ylabel('Time (percent)', fontsize=labelsize)
    im = rategrid.groupby(pd.cut(flowgrid.index, 400, labels=False)).sum()
    im = im.groupby(pd.cut(im.columns, 200, labels=False), axis=1).sum()
    from matplotlib import colors
    ax1.pcolormesh(np.arange(200) / 2, np.arange(400) / 4, im.values,
                   cmap='gist_heat_r', shading='auto')

    plt.subplots_adjust(wspace=.2, hspace=.8)
    ax2.set_title('Generated HHHs', fontsize=titlesize)
    ax2.set_xlabel('Address space (percent)', fontsize=labelsize)
    ax2.set_ylabel('Time (percent)', fontsize=labelsize)
    im = hhhgrid.groupby(pd.cut(hhhgrid.index, 400, labels=False)).sum()
    im = im.groupby(pd.cut(im.columns, 200, labels=False), axis=1).sum()
    ax2.pcolormesh(np.arange(200) / 2, np.arange(400) / 4, im.values,
                   cmap='gist_heat_r', shading='auto')

    ax3.set_title('Flow length distribution', fontsize=titlesize)
    ax3.set_xlabel('Flow length', fontsize=labelsize)
    ax3.set_ylabel('Frequency', fontsize=labelsize)
    maxd = attack_flows['duration'].max()
    bins = np.arange(0, maxd, int(maxd / 60))
    benign_bins, _ = pd.cut(benign_flows['duration'], bins=bins,
                            include_lowest=True, right=False, retbins=True)
    attack_bins, _ = pd.cut(attack_flows['duration'], bins=bins,
                            include_lowest=True, right=False, retbins=True)
    m, s, b = ax3.stem(bins[1:], benign_bins.value_counts(sort=False).values)
    plt.setp(m, color=benign_color, markersize=3)  # marker
    plt.setp(s, color=benign_color)  # stemline
    m, s, b = ax3.stem(bins[1:], attack_bins.value_counts(sort=False).values)
    plt.setp(m, color=attack_color, markersize=3)  # marker
    plt.setp(s, color=attack_color)  # stemline
    plt.setp(b, color='k')  # baseline

    ax4.set_title('Data rate distribution over IP space', fontsize=titlesize)
    ax4.set_xlabel('Address space (percent)', fontsize=labelsize)
    ax4.set_ylabel('Data rate', fontsize=labelsize)
    rates = rategrid.sum()
    rates = rates.groupby(pd.cut(rates.index, 100, labels=False)).sum()
    m, s, b = ax4.stem(rates.index, rates.values)
    plt.setp(m, color=combined_color, markersize=2)  # marker
    plt.setp(s, color=combined_color)  # stemline
    plt.setp(b, color='k')  # baseline
    rates = rategrid[attkgrid > 0].sum()
    rates = rates.groupby(pd.cut(rates.index, 100, labels=False)).sum()
    m, s, b = ax4.stem(rates.index, rates.values)
    plt.setp(m, color=attack_color, markersize=2)  # marker
    plt.setp(s, color=attack_color)  # stemline
    plt.setp(b, color='k')  # baseline
    rates = rategrid[attkgrid == 0].sum()
    rates = rates.groupby(pd.cut(rates.index, 100, labels=False)).sum()
    m, s, b = ax4.stem(rates.index, rates.values)
    plt.setp(m, color=benign_color, markersize=2)  # marker
    plt.setp(s, color=benign_color)  # stemline
    plt.setp(b, color='k')  # baseline

    ax5.set_title('Data rate over time', fontsize=titlesize)
    ax5.set_xlabel('Time (percent)', fontsize=labelsize)
    ax5.set_ylabel('Data rate', fontsize=labelsize)
    rates = flowgrid.sum(axis=1)
    rates = rates.groupby(pd.cut(rates.index, 100, labels=False)).sum()
    m, s, b = ax5.stem(rates.index, rates.values)
    plt.setp(m, color=combined_color, markersize=2)  # marker
    plt.setp(s, color=combined_color)  # stemline
    plt.setp(b, color='k')  # baseline
    rates = rategrid[attkgrid > 0].sum(axis=1)
    rates = rates.groupby(pd.cut(rates.index, 100, labels=False)).sum()
    m, s, b = ax5.stem(rates.index, rates.values)
    plt.setp(m, color=attack_color, markersize=2)  # marker
    plt.setp(s, color=attack_color)  # stemline
    plt.setp(b, color='k')  # baseline
    rates = rategrid[attkgrid == 0].sum(axis=1)
    rates = rates.groupby(pd.cut(rates.index, 100, labels=False)).sum()
    m, s, b = ax5.stem(rates.index, rates.values)
    plt.setp(m, color=benign_color, markersize=2)  # marker
    plt.setp(s, color=benign_color)  # stemline
    plt.setp(b, color='k')  # baseline

    plt.subplots_adjust(wspace=.5, hspace=.5)
    plt.show()


def main():
    args = cmdline()

    maxaddr = args.maxaddr
    maxtime = args.steps

    #	benign_start_time_sampler = UniformSampler(0, .95 * maxtime)
    #	attack_start_time_sampler = UniformSampler(0, 2/3 * maxtime)
    #	benign_duration_sampler = WeibullSampler(3/2,
    #		(1 / WeibullSampler.quantile(99, 3/2)) * 1/8 * maxtime)
    #	# We calculate the stretch factor of the Weibull distribution
    #	# so that 99% of all malicious flows will finish before maxtime
    #	attack_duration_sampler = WeibullSampler(2,
    #		(1 / WeibullSampler.quantile(99, 2)) * 1/3 * maxtime)
    #	benign_address_samplers = [
    #		NormalSampler(1/2 * maxaddr, .17 * maxaddr, 1, maxaddr)
    #	]
    #	attack_address_samplers = [
    #		NormalSampler(1/4 * maxaddr, .09 * maxaddr, 1, maxaddr),
    #		NormalSampler(3/4 * maxaddr, .09 * maxaddr, 1, maxaddr)
    #	]
    #
    #	traffic_sampler = TrafficSampler(args.benign, args.attack,
    #		args.steps, args.maxaddr,
    #		benign_start_time_sampler, attack_start_time_sampler,
    #		benign_duration_sampler, attack_duration_sampler,
    #		benign_address_samplers, attack_address_samplers)
    #	traffic_sampler.init_flows()

    flowsamplers = [
        # 1st set of benign flows
        FlowGroupSampler(args.benign,
                         UniformSampler(0, .95 * maxtime),
                         # 99% of all flows shall end before maxtime
                         WeibullSampler(3 / 2,
                                        (1 / WeibullSampler.quantile(99, 3 / 2)) * 1 / 8 * maxtime),
                         NormalSampler(1 / 2 * maxaddr, .17 * maxaddr, 1, maxaddr),
                         attack=False),
        # 1st set of attack flows
        FlowGroupSampler(args.attack // 3,
                         UniformSampler(0, 3 / 6 * maxtime),
                         WeibullSampler(2,
                                        (1 / WeibullSampler.quantile(99, 2)) * 1 / 6 * maxtime),
                         NormalSampler(1 / 4 * maxaddr, .09 * maxaddr, 1, maxaddr),
                         attack=True),
        # 2nd set of attack flows
        FlowGroupSampler(args.attack // 3,
                         UniformSampler(2 / 6 * maxtime, 5 / 6 * maxtime),
                         WeibullSampler(2,
                                        (1 / WeibullSampler.quantile(99, 2)) * 1 / 6 * maxtime),
                         NormalSampler(3 / 4 * maxaddr, .09 * maxaddr, 1, maxaddr),
                         attack=True),
        # 3rd set of attack flows
        FlowGroupSampler(args.attack // 6,
                         UniformSampler(0, 5 / 6 * maxtime),
                         WeibullSampler(2,
                                        (1 / WeibullSampler.quantile(99, 2)) * 1 / 6 * maxtime),
                         NormalSampler(1 / 8 * maxaddr, .05 * maxaddr, 1, maxaddr),
                         attack=True),
        # 4th set of attack flows
        FlowGroupSampler(args.attack // 6,
                         UniformSampler(2 / 6, 5 / 6 * maxtime),
                         WeibullSampler(2,
                                        (1 / WeibullSampler.quantile(99, 2)) * 1 / 6 * maxtime),
                         NormalSampler(3 / 8 * maxaddr, .05 * maxaddr, 1, maxaddr),
                         attack=True),
    ]

    traffic_sampler = TraceSampler(flowsamplers, maxtime)
    traffic_sampler.init_flows()

    print('Calculating HHHs...')
    hhhgrid = playthrough(traffic_sampler, args.epsilon, args.phi,
                          args.minprefix)

    #	hhhgrid = traffic_sampler.flowgrid['rate']
    plot(traffic_sampler.flows, traffic_sampler.flowgrid, hhhgrid)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
