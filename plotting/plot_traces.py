from matplotlib import pyplot as plt
import matplotlib as mpl
from numpy.random import default_rng

from gyms.hhh.flowgen.traffic_traces import BotnetSourcePattern, UniformRandomSourcePattern


def plot_botnet_pattern(num_bots=1000, num_benign=200):
    country_colors = ['orange', 'purple', 'navy', 'red']
    fig = plt.figure(figsize=(16, 4))
    ax = fig.gca()
    mpl.rcParams['figure.dpi'] = 300
    linewidth = 0.75
    botnet_src_pattern = BotnetSourcePattern()
    res = botnet_src_pattern.generate_addresses(num_bots=num_bots)

    benign_src_pattern = UniformRandomSourcePattern(16)
    benign_addresses = benign_src_pattern.generate_addresses(num_benign=num_benign)

    attack_addresses = []
    for idx, country in enumerate(res.keys()):
        # plot bots
        country_bots = default_rng().choice(res[country][1], res[country][0], replace=False)
        ax.eventplot(country_bots, color=country_colors[idx], lineoffsets=0,
                     label=f'{country} bots', linewidths=linewidth)
        attack_addresses.extend(country_bots)

    # plot benign
    ax.eventplot(benign_addresses, color='green', lineoffsets=-1, label='Benign hosts', linewidths=linewidth)

    # plot boundaries
    ax.vlines(botnet_src_pattern.subnet_starts, ymin=-2, ymax=1, label='/22 subnet borders', linestyles='dashed')

    ax.set_xlim(left=0, right=2 ** 16)
    ax.set_ylim(top=1, bottom=-2)
    ax.yaxis.set_visible(False)
    plt.title(f'Botnet source address distribution (n_bots={num_bots}, n_benign={num_benign})')
    plt.legend(ncol=6, framealpha=1.0)
    plt.xlabel('Address space')
    plt.xticks([0, 2 ** 15, 2 ** 16], ['0', r'$2^{15}$', r'$2^{16}$'])
    plt.tight_layout()
    fig.show()

    return attack_addresses, benign_addresses
