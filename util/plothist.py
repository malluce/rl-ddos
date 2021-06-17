#!/usr/bin/env python3

from mpl_toolkits.mplot3d import Axes3D

import argparse
import matplotlib.collections as collections
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import matplotlib.gridspec as mgrid
import numpy as np
import pandas as pd

def cmdline():
	argp = argparse.ArgumentParser(description = 'IP distribution histogram')

	argp.add_argument(metavar = 'histogram file', type = str,
		dest = 'histfile', help = 'CSV file containing histogram data')
	argp.add_argument(metavar = 'imagefile', type = str, dest = 'image',
		nargs = '?', default = None, help = 'output file for the plot')
	argp.add_argument('--noshow', action = 'store_true',
		help = 'suppress screen output')

	return argp.parse_args()

def create_colors(num_colors):
	cmap = plt.get_cmap('viridis')

	my_cmap = cmap(np.arange(cmap.N))
	my_cmap[:,-1] = [0.4] * cmap.N # np.linspace(0.5, 0.5, cmap.N) # alpha
	my_cmap = mcolors.ListedColormap(my_cmap)

	return [my_cmap(i) for i in np.linspace(0, 0.8, num_colors)]

def surface_plot(ax, data):
	piv = data.pivot(index = 'time', columns = 'ip', values = 'ipcount')
	piv.fillna(0)
	piv.iloc[(0,0)] = 0

	x, y = np.meshgrid(piv.index.values, piv.columns.values)
	z = piv.values

	ax.plot_surface(x, y, np.transpose(z), cmap = 'YlOrBr', rstride = 1, cstride = 2)

def polygon_plot(ax, data):
	verts = []
	zs = data.ip.unique()

	for z in zs:
		xs = data.loc[data['ip'] == z, ('time',)]
		# Now this(!) is ugly!
		hi = data.loc[data['ip'] == z, ('ipcount',)]
		xs = pd.concat([pd.DataFrame({'time' : xs.iloc[0] - 1}), xs,
			pd.DataFrame({'time' : xs.iloc[-1] + 1})], ignore_index = True)
		hi = pd.concat([pd.DataFrame([{'ipcount' : 0}]), hi,
			pd.DataFrame([{'ipcount' : 0}])], ignore_index = True)
		verts.append(list(zip(xs.time, hi.ipcount)))

	p = collections.PolyCollection(verts, facecolors = create_colors(len(verts)))

	ax.add_collection3d(p, zs = zs, zdir = 'y')

def plot(data, args):
	fig = plt.figure(figsize = (10.0,6.0))
	gs = mgrid.GridSpec(3, 2, width_ratios = [2, 1], height_ratios = [1, 4, 1])

	ax = fig.add_subplot(gs[:, 0], projection = '3d')

	polygon_plot(ax, data)

	ax.set_xlabel('Time [s]')
	ax.set_ylabel('IP')
	ax.set_zlabel('Packets per time step')
	ax.tick_params(which = 'major', labelsize = 8)
	ax.get_yaxis().labelpad = 10
	ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, _ : '%x' % int(x)))
	ax.get_zaxis().set_major_formatter(ticker.EngFormatter())

	ax.set_xlim3d(data.time.min() - 1, data.time.max() + 1)
	ax.set_ylim3d(data.ip.min(), data.ip.max())
	ax.set_zlim3d(data.ipcount.min(), data.ipcount.max())

	ax = fig.add_subplot(gs[1, 1])

	ax.set_xlabel('IP')
	ax.set_ylabel('Time [s]')
	ax.tick_params(which = 'major', labelsize = 8)
	ax.tick_params(axis = 'x', which = 'major', rotation = 30)
	ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(
		lambda x, pos : '%x' % int(x)
	))

	pcm = ax.scatter(data.ip, data.time, c = data.ipcount, marker = '|',
		s = 8, cmap = 'YlOrBr', norm = mcolors.LogNorm())

	cb = fig.colorbar(pcm, aspect = 50, format = ticker.EngFormatter())
	cb.set_label('Packets per time step')
	
	ax = fig.add_subplot(gs[0, 1])

	ax.text(0.0, 0.0,
		'\n'.join((
			"File  {}".format(args.histfile), '',
			"Total {}".format(data.ipcount.sum()),
			"Peak  {}".format(data.ipcount.max()),
			"MinIP {:x}".format(data.ip.min()),
			"MaxIP {:x}".format(data.ip.max()),
		)),
		verticalalignment = 'bottom',
		family = 'monospace'
	)
	ax.set_clip_on(False)
	ax.axis('off')

	fig.tight_layout()

	plt.gcf().canvas.set_window_title('Histogram {}'.format(args.histfile))

	if args.image is not None:
		plt.savefig(args.image, bbox_inches = 'tight')

	if not args.noshow:
		plt.show()

def main():
	args = cmdline()

	data = pd.read_csv(args.histfile)
	data.columns = ['time', 'ip', 'ipcount']

	plot(data, args)

if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		pass
