#!/usr/bin/env python

import argparse
import math
from ipaddress import IPv4Address

import matplotlib.pyplot as plt
import matplotlib.gridspec as mgrid
import numpy as np
import pandas as pd
import time

from gyms.hhh.cpp.hhhmodule import SketchHHH as HHHAlgo
from gyms.hhh.cpp.hhhmodule import HIERARCHY_SIZE

from gyms.hhh.actionset import ContinuousActionSet
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

        :returns: pd.DataFrame with self.num_flows many rows and six columns specifying the generated flows
         ('start','end', 'duration', 'addr', 'rate', 'attack'). Rate is always 1. Attack can be 0 or 1.
        """
        addr = self.address_sampler.sample(self.num_flows)
        start = self.start_sampler.sample(self.num_flows)
        duration = self.duration_sampler.sample(self.num_flows)
        end = start + duration

        rate = np.ones_like(start)

        if self.attack:
            attk = np.ones_like(start)
        else:
            attk = np.zeros_like(start)

        df = pd.DataFrame(
            np.array((start, end, duration, addr, rate, attk)).transpose(),
            columns=['start', 'end', 'duration', 'addr', 'rate', 'attack'])
        return df


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
        """
        Samples flows from flowsamplers to flows, sets flowgrid (shape (maxtime, 2*maxaddr))
        and num_samples (number of non-zero entries in flowgrid).
        """
        self.flows = pd.concat([s.sample() for s in self.flowsamplers])  # flows=[(num_flows, 6)]

        maxaddr = self.flows['addr'].max()
        maxtime = self.flows['end'].max()

        if self.maxtime is not None:
            maxtime = min(self.maxtime, maxtime)

        rate_grid = pd.DataFrame(np.zeros((maxtime, (maxaddr + 1))),
                                 columns=range(maxaddr + 1), dtype=int)
        attack_grid = pd.DataFrame(np.zeros((maxtime, (maxaddr + 1))),
                                   columns=range(maxaddr + 1), dtype=int)

        for _, f in self.flows.iterrows():  # aggregate sampled flow to grids (timesteps,addresses)
            timespan = ((rate_grid.index >= f['start']) & (rate_grid.index <= f['end']))
            rate_grid.loc[timespan, f['addr']] += f['rate']
            # TODO: handle overlap of benign and attack flows
            attack_grid.loc[timespan, f['addr']] += f['attack']

        self.flowgrid = pd.concat({'rate': rate_grid, 'attack': attack_grid}, axis=1)
        self.num_samples = self.flowgrid['rate'].sum().sum()
        self.num_attack_samples = self.flowgrid['attack'].sum().sum()

    def samples(self):
        step_finished = False
        for t in self.flowgrid.index:  # iterate over all rows (= time steps)
            step = self.flowgrid.loc[self.flowgrid.index == t]
            step_rates = step.loc[t, ('rate')]
            step_attack = step.loc[t, ('attack')]

            for addr in np.random.permutation(step_rates.index[(step_rates.values > 0)]):
                rate = step_rates[addr]
                attack_rate = step_attack[addr]

                for i in range(rate):  # yield all packets of addr (can be malicious or benign)
                    malicious = i < attack_rate
                    finished = step_finished and i == rate - 1
                    yield Packet(addr, malicious), finished

                step_finished = False

            step_finished = True

    def next(self):
        """
        Yields the next packet and a bool indicating whether the current step finished.
        """
        for packet, step_finished in self.samples():
            yield packet, step_finished

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


def cmdline():
    argp = argparse.ArgumentParser(description='Simulated HHH generation')
    PHI = 0.57
    argp.add_argument('--benign', type=int, default=50, help='Number of benign flows')
    argp.add_argument('--attack', type=int, default=100, help='Number of attack flows')
    argp.add_argument('--steps', type=int, default=300, help='Number of time steps')
    argp.add_argument('--maxaddr', type=int, default=0xffff, help='Size of address space')
    argp.add_argument('--epsilon', type=float, default=0.0001, help='Error bound')
    argp.add_argument('--phi', type=float, default=PHI, help='Query threshold')
    argp.add_argument('--minprefix', type=int, default=21,  # default=ContinuousActionSet().phi_to_prefixlen(PHI),
                      help='Minimum prefix length')
    argp.add_argument('--nohhh', action='store_true', help='Skip HHH calculation')

    return argp.parse_args()


def playthrough(trace_sampler, epsilon, phi, minprefix):
    bar = ProgressBar(trace_sampler.maxtime)
    frequencies = pd.DataFrame(np.zeros_like(trace_sampler.flowgrid['rate']),
                               columns=trace_sampler.flowgrid['rate'].columns)

    h = HHHAlgo(epsilon)
    step = 0

    # play through non-zero entries in flowgrid
    for packet, step_finished in trace_sampler.samples():
        # update hhh algorithm
        h.update(packet.ip, 1)

        # perform query and calculate hhh coverage
        if step_finished:
            hhhs = h.query(phi, minprefix)
            hhhs = [HHHEntry(_) for _ in hhhs]
            hhhs = sorted(hhhs, key=lambda x: x.size)
            print(f'===== timestep {step} =====')
            for r in hhhs:
                print(f'{str(IPv4Address(r.id)).rjust(15)}/{r.len} {r.lo, r.hi}')
            for i in range(len(hhhs)):  # iterate over HHHs sorted by prefix length
                x = hhhs[i]

                for y in hhhs[i + 1:]:
                    if y.contains(x):
                        # decrement all f_max and f_min of HHHs contained in x
                        y.hi = max(0, y.hi - x.hi)
                        y.lo = max(0, y.lo - x.lo)

                span = ((frequencies.columns >= x.id) & (frequencies.columns < (x.end)))
                frequencies.loc[step, span] += x.hi / x.size

            step += 1
            # h = HHHAlgo(epsilon)

            if bar.increment():
                bar.update()

    return frequencies


def plot(args, flows, flowgrid, hhhgrid=None):
    fig = plt.figure(figsize=(16, 8))
    gs = mgrid.GridSpec(3, 5)

    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[:, 1])
    ax2 = fig.add_subplot(gs[:, 2])
    ax3 = fig.add_subplot(gs[:, 3])
    ax5 = fig.add_subplot(gs[0, 4])
    ax6 = fig.add_subplot(gs[1, 4])
    ax7 = fig.add_subplot(gs[2, 4])

    titlesize = 9
    labelsize = 9
    ticksize = 8
    benign_color = 'dodgerblue'
    attack_color = 'crimson'
    combined_color = 'darkgrey'

    benign_flows = flows[flows['attack'] == 0]
    attack_flows = flows[flows['attack'] > 0]
    attackgrid = flowgrid['attack']
    rategrid = flowgrid['rate']
    benigngrid = rategrid - attackgrid

    def plot_heatmap(axis, grid, vmin=None, vmax=None):
        ycut, ybins = pd.cut(flowgrid.index, 300, labels=False,
                             right=False, retbins=True)
        im = grid.groupby(ycut).sum()
        im = im.groupby(pd.cut(im.columns, 200, labels=False), axis=1).sum()
        vmin = vmin if vmin is not None else im.min().min()
        vmax = vmax if vmax is not None else im.max().max()
        mesh = axis.pcolormesh(np.arange(200) / 2, ybins[1:], im.values,
                               cmap='gist_heat_r', shading='nearest', vmin=vmin, vmax=vmax)
        axis.set_xticks(np.arange(0, 101, 20))
        axis.set_yticks(np.arange(0, ybins.max() + 1, 20))
        axis.tick_params(labelsize=ticksize)
        cb = fig.colorbar(mesh, ax=axis, aspect=100)
        cb.ax.tick_params(labelsize=ticksize)

        return vmin, vmax

    ax2.set_title('Combined data rate', fontsize=titlesize)
    ax2.set_xlabel('Address space (percent)', fontsize=labelsize)
    ax2.set_ylabel('Step', fontsize=labelsize)
    vmin, vmax = plot_heatmap(ax2, rategrid)

    ax0.set_title('Benign data rate', fontsize=titlesize)
    ax0.set_xlabel('Address space (percent)', fontsize=labelsize)
    ax0.set_ylabel('Step', fontsize=labelsize)
    plot_heatmap(ax0, benigngrid, vmin, vmax)

    ax1.set_title('Attack data rate', fontsize=titlesize)
    ax1.set_xlabel('Address space (percent)', fontsize=labelsize)
    ax1.set_ylabel('Step', fontsize=labelsize)
    plot_heatmap(ax1, attackgrid, vmin, vmax)

    if hhhgrid is not None:
        ax3.set_title(f'HHH frequency \n (phi={args.phi}, L={args.minprefix})', fontsize=titlesize)
        ax3.set_xlabel('Address space (percent)', fontsize=labelsize)
        ax3.set_ylabel('Step', fontsize=labelsize)
        plot_heatmap(ax3, hhhgrid)

    def plot_frequencies(axis, x, y, color):
        axis.fill_between(x, y, 0, facecolor=color, alpha=.6)
        axis.tick_params(labelsize=ticksize)

    ax5.set_title('Flow length distribution', fontsize=titlesize)
    ax5.set_xlabel('Flow length', fontsize=labelsize)
    ax5.set_ylabel('Frequency', fontsize=labelsize)
    maxd = attack_flows['duration'].max()
    bins = np.arange(0, maxd, max(1, int(maxd / 60)))
    benign_bins, _ = pd.cut(benign_flows['duration'], bins=bins,
                            include_lowest=True, right=False, retbins=True)
    attack_bins, _ = pd.cut(attack_flows['duration'], bins=bins,
                            include_lowest=True, right=False, retbins=True)
    plot_frequencies(ax5, bins[1:], attack_bins.value_counts(sort=False).values,
                     attack_color)
    plot_frequencies(ax5, bins[1:], benign_bins.value_counts(sort=False).values,
                     benign_color)

    ax6.set_title('Data rate distribution over IP space', fontsize=titlesize)
    ax6.set_xlabel('Address space (percent)', fontsize=labelsize)
    ax6.set_ylabel('Data rate', fontsize=labelsize)
    rates = rategrid.sum()
    rates = rates.groupby(pd.cut(rates.index, 100, labels=False)).sum()
    plot_frequencies(ax6, rates.index, rates.values, combined_color)
    rates = rategrid[attackgrid > 0].sum()
    rates = rates.groupby(pd.cut(rates.index, 100, labels=False)).sum()
    plot_frequencies(ax6, rates.index, rates.values, attack_color)
    rates = rategrid[attackgrid == 0].sum()
    rates = rates.groupby(pd.cut(rates.index, 100, labels=False)).sum()
    plot_frequencies(ax6, rates.index, rates.values, benign_color)

    ax7.set_title('Data rate over time', fontsize=titlesize)
    ax7.set_xlabel('Step', fontsize=labelsize)
    ax7.set_ylabel('Data rate', fontsize=labelsize)
    rates = rategrid.sum(axis=1)
    plot_frequencies(ax7, rates.index, rates.values, combined_color)
    rates = rategrid[attackgrid > 0].sum(axis=1)
    plot_frequencies(ax7, rates.index, rates.values, attack_color)
    rates = rategrid[attackgrid == 0].sum(axis=1)
    plot_frequencies(ax7, rates.index, rates.values, benign_color)

    plt.subplots_adjust(wspace=.5, hspace=.5)
    plt.tight_layout()
    plt.show()


def main():
    args = cmdline()

    maxaddr = args.maxaddr
    maxtime = args.steps

    flowsamplers = [
        # 1st set of benign flows
        FlowGroupSampler(args.benign,
                         UniformSampler(0, 1),
                         UniformSampler(maxtime, maxtime + 1),
                         UniformSampler(0x000, 0x7ff),  # subnet 0.0.0.0/21
                         attack=False),
        # 1st set of attack flows
        FlowGroupSampler(args.attack,
                         UniformSampler(0, 1),
                         UniformSampler(maxtime / 4, maxtime / 2),  #
                         UniformSampler(0x800, 0xfff),  # subnet 0.0.8.0/21
                         attack=True)
    ]

    trace_sampler = TraceSampler(flowsamplers, maxtime)
    trace_sampler.init_flows()

    if not args.nohhh:
        print('Calculating HHHs...')
        hhhgrid = playthrough(trace_sampler, args.epsilon, args.phi,
                              args.minprefix)
    else:
        hhhgrid = None

    plot(args, trace_sampler.flows, trace_sampler.flowgrid, hhhgrid)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
