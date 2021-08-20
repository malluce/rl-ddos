"""
Plots the weighted components of the reward function, i.e. FPR/Precision/Recall/BlacklistSize.
"""
import numpy as np
from matplotlib import pyplot as plt

from gyms.hhh.reward import DefaultRewardCalc, RewardCalc


def plot_contrib_factors(reward_calc: RewardCalc):
    px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
    fig: plt.Figure = plt.figure(figsize=(1920 * px, 1080 * px))
    plt.rcParams.update({'font.size': 18})

    def plot_zero_one():
        ax = fig.add_subplot(211)  # zero-one-axis for fpr, recall, precision
        x = np.linspace(0, 1, 1000)
        weighted_prec = ('precision factor', reward_calc.weighted_precision(x))
        weighted_fpr = ('fpr factor', reward_calc.weighted_fpr(x))
        weighted_rec = ('recall factor', reward_calc.weighted_recall(x))

        for label, y in [weighted_rec, weighted_prec, weighted_fpr]:
            ax.plot(x, y, label=label)

        ax.set_xlabel('precision/recall/fpr')
        ax.set_ylabel('factor in reward function')
        ax.set_ylim(bottom=0.0)
        ax.legend()

    def plot_blacklist():
        ax = fig.add_subplot(212)  # for blacklist size

        x = np.linspace(0, 100, 1000, dtype=np.int)
        weighted_bl = reward_calc.weighted_bl_size(x)
        ax.plot(x, weighted_bl, label='blacklist factor')
        ax.set_xlabel('blacklist size')
        ax.set_ylabel('factor in reward function')
        ax.set_ylim(bottom=0.0)
        ax.legend()

    plot_zero_one()
    plot_blacklist()
    fig.suptitle('Reward Contributing Factors')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    reward_calc = DefaultRewardCalc()
    plot_contrib_factors(reward_calc)
