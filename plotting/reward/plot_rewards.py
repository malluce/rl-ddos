"""
Plots the entire reward function, i.e., the multiplication of its components. (Figure 4.18)
"""
from gyms.hhh.reward import MultiplicativeRewardThesis, RewardCalc

import numpy as np
import matplotlib.pyplot as plt

from plotting.reward.util import get_reward
from matplotlib import cm


def plot_reward_3d(reward_calculator: RewardCalc):
    px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
    fig: plt.Figure = plt.figure(figsize=(2000 * px, 1000 * px))
    plt.rcParams.update({'font.size': 18})

    for plt_count, bl_size in enumerate([1, 10, 25, 50, 100, 200, 500, 1000]):
        f = np.linspace(0.0, 1.0, num=100)
        recall = np.linspace(0.0, 1.0, num=100)
        X, Y = np.meshgrid(f, recall)

        num_cols = 4
        num_rows = 2
        ax: plt.Axes = fig.add_subplot(num_rows, num_cols, plt_count + 1, projection='3d')

        Z = get_reward(reward_calculator, f, bl_size, precision=X, recall=Y)
        surf = ax.plot_surface(X, Y, Z, antialiased=True, cmap=cm.inferno)
        surf._facecolors2d = surf._facecolor3d  # required for label
        surf._edgecolors2d = surf._edgecolor3d  # required for label

        ax.set_xlabel('False Positive Rate', labelpad=15)
        ax.set_ylabel('Recall', labelpad=15)
        ax.set_zlabel('Reward', labelpad=10)
        ax.set_zlim(0, 1.0)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_title('$n_{rules}$=' + str(bl_size), y=0.85)
        ax.view_init(5, -60)
    fig.tight_layout(h_pad=0.1, w_pad=3, pad=3)

    plt.show()


if __name__ == '__main__':
    reward_calc = MultiplicativeRewardThesis(precision_weight=0, bl_weight=0.25, recall_weight=2, fpr_weight=2)
    plot_reward_3d(reward_calc)
