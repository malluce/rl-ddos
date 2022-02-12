import argparse
import json
import time
from ipaddress import IPv4Address

import gzip
import numpy as np
from gyms.hhh.cpp.hhhmodule import SketchHHH as HHHAlgo
from matplotlib import pyplot as plt
import matplotlib.gridspec as mgrid

from gyms.hhh.flowgen.distgen import TraceSampler
from gyms.hhh.flowgen.traffic_traces import S1, S2, S3
from gyms.hhh.label import Label
from plotting.plot_results import add_vspan_to, get_vspan_spec_for


# This script is used to visualize scenarios by generating an example episode and showing the active benign and attack
# flows over time. Also the data rate is plotted. (Figures 6.2, 6.19, 6.26)
# Scenarios can be selected in the visualize(..) method below, then simply run the script.

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


def plot(steps, phi, l, flows, rate_grid, attack_grid, hhh_grid=None, squash=False, pattern_id=None):
    fig = plt.figure(figsize=(16, 9))
    gs = mgrid.GridSpec(3, 1)

    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, :])
    ax2 = fig.add_subplot(gs[2, :])
    # ax3 = fig.add_subplot(gs[3, :])

    titlesize = 15
    labelsize = 15
    ticksize = 8
    benign_color = 'dodgerblue'
    attack_color = 'crimson'
    combined_color = 'darkgrey'
    benign_grid = rate_grid - attack_grid

    def scale(grid, num_bins, axis, transpose=False):
        bins2 = np.linspace(0, grid.shape[axis], num_bins + 1).astype(int)[1:]
        split = np.split(grid, bins2[:-1], axis=axis)
        scaled2 = np.array([_.sum(axis=axis) for _ in split])
        if transpose: scaled2 = scaled2.T
        return scaled2, bins2

    def plot_heatmap(axis, grid, vmin=None, vmax=None, colorbar=False, colorbar_label=None):
        scaled_grid, ybins = scale(grid, steps + 1, 0)
        scaled_grid, _ = scale(scaled_grid, 200, 1)
        if squash:
            # squash values together for clearer visuals (e.g. for reflector-botnet switch)
            scaled_grid = np.sqrt(scaled_grid, out=np.zeros_like(scaled_grid, dtype=np.float))
        vmin = vmin if vmin is not None else scaled_grid.min()
        vmax = vmax if vmax is not None else scaled_grid.max()
        mesh = axis.pcolormesh(ybins, np.arange(200) / 2, scaled_grid, cmap='gist_heat_r', shading='nearest', vmin=vmin,
                               vmax=vmax)

        axis.set_yticks(np.arange(0, 101, 50))
        axis.set_yticklabels(['0', '$2^{15}$', '$2^{16}$'])
        axis.set_xticks(np.arange(0, ybins.max() + 1, 100))
        axis.tick_params(labelsize=labelsize)
        if colorbar:
            cb = fig.colorbar(mesh, ax=axis, use_gridspec=True, pad=0.01)
            cb.set_label('Data rate' if colorbar_label is None else colorbar_label, fontsize=labelsize)
            cb.ax.tick_params(labelsize=labelsize)

        return vmin, vmax

    scaled_grid, ybins = scale(rate_grid, steps + 1, 0)
    scaled_grid, _ = scale(scaled_grid, 200, 1, True)
    if squash:
        # squash values together for clearer visuals (e.g. for reflector-botnet switch)
        scaled_grid = np.sqrt(scaled_grid, out=np.zeros_like(scaled_grid, dtype=np.float))
    vmin = scaled_grid.min()
    vmax = scaled_grid.max()

    if hhh_grid is not None:
        ax2.set_title('Applied filter rules', fontsize=titlesize)
        ax2.set_ylabel('Address space', fontsize=labelsize)
        ax2.set_xlabel('Time index', fontsize=labelsize)
        plot_heatmap(ax2, hhh_grid, colorbar=True, colorbar_label='HHH frequency')
    else:
        ax2.set_title('Combined data rate', fontsize=titlesize)
        ax2.set_ylabel('Address space', fontsize=labelsize)
        ax2.set_xlabel('Time index', fontsize=labelsize)
        vmin, vmax = plot_heatmap(ax2, rate_grid, colorbar=True)

    ax1.set_title('Attack data rate', fontsize=titlesize)
    ax1.set_ylabel('Address space', fontsize=labelsize)
    ax1.set_xlabel('Time index', fontsize=labelsize)
    plot_heatmap(ax1, attack_grid, vmin, vmax, colorbar=True)

    ax0.set_title('Benign data rate', fontsize=titlesize)
    ax0.set_ylabel('Address space', fontsize=labelsize)
    ax0.set_xlabel('Time index', fontsize=labelsize)
    plot_heatmap(ax0, benign_grid, vmin, vmax, colorbar=True)

    if pattern_id is not None:
        add_vspan_to(ax0, get_vspan_spec_for(pattern=pattern_id), True, use_time_index=True)
        add_vspan_to(ax1, get_vspan_spec_for(pattern=pattern_id), False, use_time_index=True)
        add_vspan_to(ax2, get_vspan_spec_for(pattern=pattern_id), False, use_time_index=True)
    for ax in [ax0, ax1, ax2]:
        ax.set_xlim(left=0)
    plt.tight_layout()

    plt.savefig(f'test{time.time()}.png', bbox_inches='tight')
    plt.show()

    fig = plt.figure()
    gs = mgrid.GridSpec(1, 1)

    ax0 = fig.add_subplot(gs[0, :])

    # ax1 = fig.add_subplot(gs[1, :])

    def plot_frequencies(axis, x, y, color):
        axis.fill_between(x, y, 0, facecolor=color, alpha=.6)
        axis.tick_params(labelsize=ticksize)

    ax0.set_xlabel('Time index', fontsize=labelsize)
    ax0.set_ylabel('Data rate', fontsize=labelsize)
    x = range(benign_grid.shape[0])
    benign_y = benign_grid.sum(axis=1)
    attack_y = benign_y + attack_grid.sum(axis=1)
    ax0.fill_between(x, 0, benign_y, facecolor=benign_color, label='Benign traffic')
    ax0.fill_between(x, benign_y, attack_y, facecolor=attack_color, label='Attack traffic')
    ax0.tick_params(labelsize=labelsize)
    ax0.set_ylim(bottom=0)
    ax0.set_xlim(left=0, right=x[-1] + 1)
    if pattern_id is not None:
        add_vspan_to(ax0, get_vspan_spec_for(pattern=pattern_id), True, use_time_index=True)

    # ax1.set_title('Data rate distribution', fontsize=titlesize)
    # ax1.set_xlabel('Address space', fontsize=labelsize)
    # ax1.set_ylabel('Data rate', fontsize=labelsize)
    # num_bin = 100
    # benign_y, bins = scale(benign_grid.sum(axis=0), num_bin, 0)
    # x = num_bin * bins / bins[-1]
    # attack_y, _ = scale(attack_grid.sum(axis=0), num_bin, 0)
    # attack_y += benign_y
    # ax1.fill_between(x, 0, benign_y, facecolor=benign_color)
    # ax1.fill_between(x, benign_y, attack_y, facecolor=attack_color)
    # ax1.tick_params(labelsize=ticksize)
    # ax1.set_xticks(np.arange(0, 101, 50))
    # ax1.set_xticklabels(['0', '$2^{15}$', '$2^{16}$'])
    plt.legend(loc='upper left')
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


def main():
    visualize(None, None, None, None, True, 10, .0001, 599)


def visualize(flow_file, rate_grid_file, attack_grid_file, blacklist_file, nohhh, interval, epsilon, steps, phi=None,
              l=None, trace_id=None):
    if flow_file is None:
        trace = S3()  # change this line of code to "S1()", "S2()", or "S3()" visualize a scenario
        fgs = trace.get_flow_group_samplers()
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
         trace_sampler.attack_grid, hhh_grid, squash=True,
         pattern_id=trace.get_rate_pattern_id(0) if trace_id is None else trace_id)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
