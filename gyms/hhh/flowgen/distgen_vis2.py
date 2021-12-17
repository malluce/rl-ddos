import argparse
import json
from ipaddress import IPv4Address

import gzip

import matplotlib
import numpy as np
from gyms.hhh.cpp.hhhmodule import SketchHHH as HHHAlgo
from matplotlib import pyplot as plt
import matplotlib.gridspec as mgrid

from gyms.hhh.flowgen.distgen import FlowGroupSampler, TraceSampler, UniformSampler, WeibullSampler
from gyms.hhh.flowgen.traffic_traces import BotTrace, HafnerT1, HafnerT2, MixedNTPBot, MixedSSDPBot, SSDPTrace, T1, T2, \
    T3, T4, \
    THauke5, TRandomPatternSwitch, THauke
from gyms.hhh.label import Label
from gyms.hhh.loop import apply_hafner_heuristic


# TODO tmp file for inter slides, delete afterwards

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
    hhhs = []
    time_index = 0

    # play through non-zero entries in rate_grid
    for packet, time_index_finished in trace_sampler.samples():
        # update hhh algorithm
        h.update(packet.ip)
        if not time_index_finished:
            continue
        # perform query and calculate hhh coverage
        if time_index % interval == interval - 1 and time_index != trace_sampler.maxtime:
            # HHHs are sorted in descending order of prefix length
            hhhs = [HHHEntry.copy(_) for _ in h.query(phi, minprefix)]
            hhhs = reduce_hhhs(hhhs)
            # Reset the HHH algorithm after querying
            h.clear()
            print(f'===== time index {time_index} =====')
            print(f'number of rules={len(hhhs)}')
            for r in hhhs:
                print(f'{str(IPv4Address(r.id)).rjust(15)}/{r.len} {r.lo, r.hi}')
        render_hhhs(hhhs, frequencies, time_index)
        time_index += 1

        if bar.increment():
            bar.update()

    return frequencies


def plot(steps, phi, l, flows, rate_grid, attack_grid, hhh_grid=None, squash=False):
    fig = plt.figure(figsize=(6, 6))
    gs = mgrid.GridSpec(1, 3)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    titlesize = 13
    labelsize = 12
    ticksize = 8
    benign_grid = rate_grid - attack_grid

    def scale(grid, num_bins, axis, transpose=False):
        bins2 = np.linspace(0, grid.shape[axis], num_bins + 1).astype(int)[1:]
        split = np.split(grid, bins2[:-1], axis=axis)
        scaled2 = np.array([_.sum(axis=axis) for _ in split])
        if transpose: scaled2 = scaled2.T
        return scaled2, bins2

    def plot_heatmap(axis, grid, vmin=None, vmax=None):
        scaled_grid, ybins = scale(grid, steps + 1, 0)
        scaled_grid, _ = scale(scaled_grid, 200, 1, True)
        if squash:
            # squash values together for clearer visuals (e.g. for reflector-botnet switch)
            scaled_grid = np.sqrt(scaled_grid, out=np.zeros_like(scaled_grid, dtype=np.float))
        vmin = vmin if vmin is not None else scaled_grid.min()
        vmax = vmax if vmax is not None else scaled_grid.max()
        mesh = axis.pcolormesh(np.arange(200) / 2, ybins, scaled_grid, cmap='gist_heat_r', shading='nearest', vmin=vmin,
                               vmax=vmax)

        axis.set_xticks(np.arange(0, 101, 50))
        axis.set_xticklabels(['0', '$2^{15}$', '$2^{16}$'])
        axis.set_yticks(np.arange(0, ybins.max() + 1, 20))
        axis.tick_params(labelsize=ticksize)
        # cb = fig.colorbar(mesh, ax=axis, aspect=100)
        # cb.ax.tick_params(labelsize=ticksize)

        return vmin, vmax

    scaled_grid, ybins = scale(rate_grid, steps + 1, 0)
    scaled_grid, _ = scale(scaled_grid, 200, 1, True)
    if squash:
        # squash values together for clearer visuals (e.g. for reflector-botnet switch)
        scaled_grid = np.sqrt(scaled_grid, out=np.zeros_like(scaled_grid, dtype=np.float))
    vmin = scaled_grid.min()
    vmax = scaled_grid.max()

    ax0.set_title('Benign data rate', fontsize=titlesize)
    ax0.set_xlabel('Address space', fontsize=labelsize)
    ax0.set_ylabel('Time index', fontsize=labelsize)
    plot_heatmap(ax0, benign_grid, vmin, vmax)

    ax1.set_title('Attack data rate', fontsize=titlesize)
    ax1.set_xlabel('Address space', fontsize=labelsize)
    # ax1.set_ylabel('Time index', fontsize=labelsize)
    plot_heatmap(ax1, attack_grid, vmin, vmax)

    if hhh_grid is not None:
        ax2.set_title(f'Rule coverage', fontsize=titlesize)
        ax2.set_xlabel('Address space', fontsize=labelsize)
        # ax2.set_ylabel('Time index', fontsize=labelsize)
        plot_heatmap(ax2, hhh_grid)

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
    hhhgrid = np.zeros((maxtime, maxaddr + 1))
    for time_index in range(len(episode_blacklist)):
        hhhs = [HHHEntry.from_dict(_) for _ in sorted(episode_blacklist[time_index],
                                                      key=lambda x: x['len'], reverse=True)]
        print(f'======== TIME IDX {time_index} ========')
        print(f'number of rules={len(hhhs)}')
        # if len(hhhs) == 1:
        for r in sorted(hhhs, key=lambda h: h.id):
            print(f'{str(IPv4Address(r.id)).rjust(15)}/{r.len} {r.lo, r.hi}')
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
    BENIGN = 200  # 300
    ATTACK = 50  # 150
    argp.add_argument('--benign', type=int, default=BENIGN, help='Number of benign flows')
    argp.add_argument('--attack', type=int, default=ATTACK, help='Number of attack flows')
    argp.add_argument('--steps', type=int, default=599, help='Number of time steps')
    argp.add_argument('--maxaddr', type=int, default=0xffff, help='Size of address space')
    argp.add_argument('--epsilon', type=float, default=.0001, help='Error bound')
    argp.add_argument('--phi', type=float, default=0.05, help='Query threshold')
    argp.add_argument('--minprefix', type=int, default=22, help='Minimum prefix length')
    argp.add_argument('--interval', type=int, default=10, help='HHH query interval')
    argp.add_argument('--nohhh', action='store_true', help='Skip HHH calculation')

    args = argp.parse_args()

    if (args.flow_file is not None and (args.rate_grid_file is None
                                        or args.attack_grid_file is None)):
        raise ValueError('flow_file requires rate_grid_file and attack_grid_file')

    return args


def main():
    args = cmdline()
    visualize(args.flow_file, args.rate_grid_file, args.attack_grid_file, args.blacklist_file, args.nohhh,
              args.interval, args.epsilon, args.steps, args.phi, args.minprefix)


def visualize(flow_file, rate_grid_file, attack_grid_file, blacklist_file, nohhh, interval, epsilon, steps, phi=None,
              l=None):
    trace = TRandomPatternSwitch(is_eval=True)
    # trace = SSDPTrace()
    trace = TRandomPatternSwitch(is_eval=True, random_toggle_time=True, smooth_transition=True, benign_normal=True,
                                 benign_flows=200)
    trace = THauke5()
    trace = T3()
    # trace = MixedNTPBot()
    # for i in range(0, 9):
    if flow_file is None:
        fgs = trace.get_flow_group_samplers()
        # if i != 8:
        #    continue
        trace_sampler = TraceSampler(fgs, steps)
        trace_sampler.init_flows()
    else:
        trace_sampler = load_tracesampler(flow_file, rate_grid_file,
                                          attack_grid_file)
    if blacklist_file:
        hhh_grid = render_blacklist_history(blacklist_file,
                                            trace_sampler.rate_grid.shape[0], trace_sampler.maxaddr)
    elif not nohhh:
        print('Calculating HHHs...')
        hhh_grid = playthrough(trace_sampler, epsilon, phi, l, interval)
    else:
        hhh_grid = None
    plot(steps, phi, l, trace_sampler.flows, trace_sampler.rate_grid,
         trace_sampler.attack_grid, hhh_grid, squash=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
