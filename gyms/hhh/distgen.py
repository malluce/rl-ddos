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


class ProgressBar(object):

    def __init__(self, maxprogress, step=100, displaysteps=10):
        self.maxprogress = maxprogress
        self.step = step
        self.displaysteps = displaysteps
        self.i = 0

    def increment(self):
        self.i += 1

        return self.i % (self.maxprogress / self.step) == 0

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
    argp.add_argument('--steps', type=int, default=1000, help='Number of time steps')
    argp.add_argument('--maxaddr', type=int, default=0xffff, help='Size of address space')
    argp.add_argument('--epsilon', type=float, default=0.0001, help='Error bound')
    argp.add_argument('--phi', type=float, default=PHI, help='Query threshold')
    argp.add_argument('--interval', type=int, default=10, help='Number of grid rows after which the HHH are reset')
    argp.add_argument('--minprefix', type=int, default=21,  # default=ContinuousActionSet().phi_to_prefixlen(PHI),
                      help='Minimum prefix length')
    argp.add_argument('--nohhh', action='store_true', help='Skip HHH calculation')

    return argp.parse_args()


def playthrough(trace_sampler, epsilon, phi, minprefix, interval):
    bar = ProgressBar(trace_sampler.maxtime)
    frequencies = np.zeros_like(trace_sampler.rate_grid, dtype=np.float64)
    h = HHHAlgo(epsilon)
    time_index = 0
    hhhs = []
    time_index = 0

    # play through non-zero entries in rate_grid
    for packet, time_index_finished in trace_sampler.samples():
        # update hhh algorithm
        h.update(packet.ip)

        if not time_index_finished:
            continue

        # perform query and calculate hhh coverage
        if time_index % interval == interval - 1:
            hhhs = h.query(phi, minprefix)
            hhhs = [HHHEntry(_) for _ in hhhs]
            hhhs = sorted(hhhs, key=lambda x: x.size)
            print(f'===== time index {time_index} =====')
            for r in hhhs:
                print(f'{str(IPv4Address(r.id)).rjust(15)}/{r.len} {r.lo, r.hi}')
            for i in range(len(hhhs)):  # iterate over HHHs sorted by prefix length
                x = hhhs[i]

                for y in hhhs[i + 1:]:
                    if y.contains(x):
                        # decrement all f_max and f_min of HHHs contained in x
                        y.hi = max(0, y.hi - x.hi)
                        y.lo = max(0, y.lo - x.lo)

                # Render the HHHs onto to the HHH grid with equally
                # distributed item frequencies
                for hhh in hhhs:
                    if hhh.hi > 0:
                        frequencies[time_index, hhh.id: hhh.end] += hhh.hi / hhh.size
        time_index += 1

        if bar.increment():
            bar.update()

    return frequencies


def plot(args, flows, rate_grid, attack_grid, hhh_grid=None):
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

    benign_flows = flows[flows[:, 5] == 0]
    attack_flows = flows[flows[:, 5] != 0]
    benign_grid = rate_grid - attack_grid

    def scale(grid, num_bins, axis, transpose=False):
        bins = np.linspace(0, grid.shape[axis], num_bins + 1).astype(int)[1:]
        split = np.split(grid, bins[:-1], axis=axis)
        scaled = np.array([_.sum(axis=axis) for _ in split])
        if transpose: scaled = scaled.T
        return scaled, bins

    def plot_heatmap(axis, grid, vmin=None, vmax=None):
        scaled_grid, ybins = scale(grid, 300, 0)
        scaled_grid, _ = scale(scaled_grid, 200, 1, True)
        vmin = vmin if vmin is not None else scaled_grid.min()
        vmax = vmax if vmax is not None else scaled_grid.max()
        mesh = axis.pcolormesh(np.arange(200) / 2, ybins, scaled_grid, cmap='gist_heat_r', shading='nearest', vmin=vmin,
                               vmax=vmax)

        axis.set_xticks(np.arange(0, 101, 20))
        axis.set_yticks(np.arange(0, ybins.max() + 1, 20))
        axis.tick_params(labelsize=ticksize)
        cb = fig.colorbar(mesh, ax=axis, aspect=100)
        cb.ax.tick_params(labelsize=ticksize)

        return vmin, vmax

    ax2.set_title('Combined data rate', fontsize=titlesize)
    ax2.set_xlabel('Address space (percent)', fontsize=labelsize)
    ax2.set_ylabel('Time Index', fontsize=labelsize)
    vmin, vmax = plot_heatmap(ax2, rate_grid)

    ax0.set_title('Benign data rate', fontsize=titlesize)
    ax0.set_xlabel('Address space (percent)', fontsize=labelsize)
    ax0.set_ylabel('Time Index', fontsize=labelsize)
    plot_heatmap(ax0, benign_grid, vmin, vmax)

    ax1.set_title('Attack data rate', fontsize=titlesize)
    ax1.set_xlabel('Address space (percent)', fontsize=labelsize)
    ax1.set_ylabel('Time Index', fontsize=labelsize)
    plot_heatmap(ax1, attack_grid, vmin, vmax)

    if hhh_grid is not None:
        ax3.set_title(f'HHH frequency \n (phi={args.phi}, L={args.minprefix})', fontsize=titlesize)
        ax3.set_xlabel('Address space (percent)', fontsize=labelsize)
        ax3.set_ylabel('Time Index', fontsize=labelsize)
        plot_heatmap(ax3, hhh_grid)

    def plot_frequencies(axis, x, y, color):
        axis.fill_between(x, y, 0, facecolor=color, alpha=.6)
        axis.tick_params(labelsize=ticksize)

    ax5.set_title('Flow length distribution', fontsize=titlesize)
    ax5.set_xlabel('Flow length', fontsize=labelsize)
    ax5.set_ylabel('Frequency', fontsize=labelsize)
    benign_flow_durations = benign_flows[:, 2]
    attack_flow_durations = attack_flows[:, 2]
    maxd = max(benign_flow_durations.max(), attack_flow_durations.max())
    hist, bins = np.histogram(attack_flow_durations, bins=30, range=(0.0, maxd))
    plot_frequencies(ax5, bins[1:], hist, attack_color)
    hist, bins = np.histogram(benign_flow_durations, bins=30, range=(0.0, maxd))
    plot_frequencies(ax5, bins[1:], hist, benign_color)

    ax6.set_title('Data rate distribution over IP space', fontsize=titlesize)
    ax6.set_xlabel('Address space (percent)', fontsize=labelsize)
    ax6.set_ylabel('Data rate', fontsize=labelsize)
    scaled, bins = scale(rate_grid.sum(axis=0), 100, 0)
    plot_frequencies(ax6, 100 * bins / bins[-1], scaled, combined_color)
    scaled, _ = scale(attack_grid.sum(axis=0), 100, 0)
    plot_frequencies(ax6, 100 * bins / bins[-1], scaled, attack_color)
    scaled, _ = scale(benign_grid.sum(axis=0), 100, 0)
    plot_frequencies(ax6, 100 * bins / bins[-1], scaled, benign_color)

    ax7.set_title('Data rate over time', fontsize=titlesize)
    ax7.set_xlabel('Episode (percent)', fontsize=labelsize)
    ax7.set_ylabel('Data rate', fontsize=labelsize)
    scaled, bins = scale(rate_grid.sum(axis=1), 100, 0)
    plot_frequencies(ax7, 100 * bins / bins[-1], scaled, combined_color)
    scaled, _ = scale(attack_grid.sum(axis=1), 100, 0)
    plot_frequencies(ax7, 100 * bins / bins[-1], scaled, attack_color)
    scaled, _ = scale(benign_grid.sum(axis=1), 100, 0)
    plot_frequencies(ax7, 100 * bins / bins[-1], scaled, benign_color)

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
        hhh_grid = playthrough(trace_sampler, args.epsilon, args.phi,
                               args.minprefix, args.interval)
    else:
        hhh_grid = None

    plot(args, trace_sampler.flows, trace_sampler.rate_grid, trace_sampler.attack_grid, hhh_grid)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
