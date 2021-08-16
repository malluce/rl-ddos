import argparse
import json
from ipaddress import IPv4Address

import gzip
import numpy as np
from gyms.hhh.cpp.hhhmodule import SketchHHH as HHHAlgo
from matplotlib import pyplot as plt
import matplotlib.gridspec as mgrid

from gyms.hhh.flowgen.distgen import FlowGroupSampler, TraceSampler, UniformSampler
from gyms.hhh.flowgen.traffic_traces import T2, T3, THauke
from gyms.hhh.label import Label


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


class HHHEntry(object):

    @staticmethod
    def copy(other):
        return HHHEntry(other.id, other.len, other.lo, other.hi)

    @staticmethod
    def from_dict(d):
        return HHHEntry(d['id'], d['len'], d['lo'], d['hi'])

    def __init__(self, id, len, lo, hi):
        self.id = id
        self.len = len
        self.size = Label.subnet_size(self.len)
        self.end = self.id + self.size
        self.lo = lo
        self.hi = hi

    def contains(self, other):
        return (self.len < other.len
                and other.id & Label.PREFIXMASK[self.len] == self.id)

    def __str__(self):
        return 'HHH(id: {}, len: {}, end: {}, lo: {}, hi: {})'.format(
            self.id, self.len, self.end, self.lo, self.hi)


def reduce_hhhs(hhhs):
    # Subtract lower level frequencies from higher level HHHs
    # to avoid overcounting
    for i in range(len(hhhs)):
        x = hhhs[i]
        for y in hhhs[i + 1:]:
            if y.contains(x):
                y.hi = max(0, y.hi - x.hi)
                y.lo = max(0, y.lo - x.lo)
    return hhhs


def render_hhhs(hhhs, grid, index):
    # Render HHHs onto to a numpy grid with equally
    # distributed item frequencies
    for hhh in hhhs:
        if hhh.hi > 0:
            grid[index, hhh.id: hhh.end] += hhh.hi / hhh.size


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
            # HHHs are sorted in descending order of prefix length
            hhhs = [HHHEntry.copy(_) for _ in h.query(phi, minprefix)]
            hhhs = reduce_hhhs(hhhs)
            # Reset the HHH algorithm after querying
            h.clear()
            print(f'===== time index {time_index} =====')
            for r in hhhs:
                print(f'{str(IPv4Address(r.id)).rjust(15)}/{r.len} {r.lo, r.hi}')
        render_hhhs(hhhs, frequencies, time_index)
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


def load_numpy(filename):
    with gzip.GzipFile(filename, 'r') as f:
        return np.load(f)


def load_tracesampler(flow_file, rate_grid_file, attack_grid_file):
    return TraceSampler.load(load_numpy(flow_file), load_numpy(rate_grid_file),
                             load_numpy(attack_grid_file))


def render_blacklist_history(blacklist_file, maxtime, maxaddr):
    with open(blacklist_file, 'r') as f:
        episode_blacklist = json.load(f)
    hhhgrid = np.zeros((300, maxaddr + 1))
    for time_index in range(len(episode_blacklist)):
        hhhs = [HHHEntry.from_dict(_) for _ in sorted(episode_blacklist[time_index],
                                                      key=lambda x: x['len'], reverse=True)]
        render_hhhs(hhhs, hhhgrid, time_index)
    return hhhgrid


def cmdline():
    argp = argparse.ArgumentParser(description='Simulated HHH generation')
    argp.add_argument('flow_file', type=str, default=None, nargs='?',
                      help='Gzip-compressed numpy file containing flow information')
    argp.add_argument('rate_grid_file', type=str, default=None, nargs='?',
                      help='Gzip-compressed numpy file containing combined rate information')
    argp.add_argument('attack_grid_file', type=str, default=None, nargs='?',
                      help='Gzip-compressed numpy file containing attack rate information')
    argp.add_argument('blacklist_file', type=str, default=None, nargs='?', help='File containing blacklist information')
    argp.add_argument('--benign', type=int, default=500, help='Number of benign flows')
    argp.add_argument('--attack', type=int, default=1000, help='Number of attack flows')
    argp.add_argument('--steps', type=int, default=600, help='Number of time steps')
    argp.add_argument('--maxaddr', type=int, default=0xfff, help='Size of address space')
    argp.add_argument('--epsilon', type=float, default=.005, help='Error bound')
    argp.add_argument('--phi', type=float, default=.02, help='Query threshold')
    argp.add_argument('--minprefix', type=int, default=0, help='Minimum prefix length')
    argp.add_argument('--interval', type=int, default=1, help='HHH query interval')
    argp.add_argument('--nohhh', action='store_true', help='Skip HHH calculation')

    args = argp.parse_args()

    if (args.flow_file is not None and (args.rate_grid_file is None
                                        or args.attack_grid_file is None)):
        raise ValueError('flow_file requres rate_grid_file and attack_grid_file')

    return args


def main():
    args = cmdline()

    if args.flow_file is None:
        trace = T3(args.benign, args.attack, args.steps, args.maxaddr).get_flow_group_samplers()
        trace_sampler = TraceSampler(trace, args.steps)
        trace_sampler.init_flows()
    else:
        trace_sampler = load_tracesampler(args.flow_file, args.rate_grid_file,
                                          args.attack_grid_file)

    if args.blacklist_file:
        hhh_grid = render_blacklist_history(args.blacklist_file,
                                            trace_sampler.rate_grid.shape[0], trace_sampler.maxaddr)
    elif not args.nohhh:
        print('Calculating HHHs...')
        hhh_grid = playthrough(trace_sampler, args.epsilon, args.phi,
                               args.minprefix, args.interval)
    else:
        hhh_grid = None

    plot(args, trace_sampler.flows, trace_sampler.rate_grid,
         trace_sampler.attack_grid, hhh_grid)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
