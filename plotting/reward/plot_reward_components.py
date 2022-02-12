"""
Plots the weighted components of the reward function, i.e. FPR/Precision/Recall/BlacklistSize. (Figure 4.17)
"""
import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from gyms.hhh.reward import MultiplicativeRewardThesis, RewardCalc


def plot_contrib_factors(reward_calc: RewardCalc):
    plt.rcParams.update({'font.size': 15})
    matplotlib.rcParams['figure.dpi'] = 300

    def plot_zero_one():
        x = np.linspace(0, 1, 1000)
        weighted_fpr = ('False positive rate', reward_calc.weighted_fpr(x))
        weighted_rec = ('Recall', reward_calc.weighted_recall(x))

        for label, y in [weighted_rec, weighted_fpr]:
            plt.plot(x, y, label=label)

        plt.xlabel('Raw metric')
        plt.ylabel('Factor in reward function')
        plt.ylim(bottom=0.0)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_blacklist():
        x = np.linspace(0, 1000, 1000, dtype=np.int)
        weighted_bl = reward_calc.weighted_bl_size(x)
        plt.plot(x, weighted_bl)
        plt.xlabel('Number of filter rules')
        plt.ylabel('Factor in reward function')
        plt.ylim(bottom=0.0)
        plt.tight_layout()
        plt.show()

    plot_zero_one()
    plot_blacklist()


if __name__ == '__main__':
    reward_calc = MultiplicativeRewardThesis(precision_weight=0, fpr_weight=2, recall_weight=2, bl_weight=0.25)
    plot_contrib_factors(reward_calc)
