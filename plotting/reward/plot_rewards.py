"""
Plots the entire reward function, i.e., the multiplication of its components.
"""
import math

from gyms.hhh.reward import DefaultRewardCalc, RewardCalc

import numpy as np
import matplotlib.pyplot as plt

from plotting.reward.util import get_reward

BLACKLIST_MIN = 1  # start value of blacklist size
BLACKLIST_MAX = 10  # end value of blacklist size
NUM_BLACKLIST = 6  # number of 3D plots to generate (different blacklist sizes)
NUM_FPR = 5  # number of surfaces per 3D plot (different fpr)


def plot_reward(reward_calculator: RewardCalc, bl_min=BLACKLIST_MIN, bl_max=BLACKLIST_MAX, bl_num=NUM_BLACKLIST,
                fpr_num=NUM_FPR):
    px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
    fig: plt.Figure = plt.figure(figsize=(1920 * px, 1080 * px))

    plt.rcParams.update({'font.size': 18})

    for plt_count, bl_size in enumerate(np.linspace(bl_min, bl_max, num=bl_num, dtype=np.int),
                                        start=1):
        precision = np.linspace(0.0, 1.0, num=100)
        recall = np.linspace(0.0, 1.0, num=100)
        X, Y = np.meshgrid(precision, recall)

        num_cols = 1 if bl_num == 1 else (2 if bl_num <= 4 else 3)
        num_rows = math.ceil(bl_num / num_cols)
        ax: plt.Axes = fig.add_subplot(num_rows, num_cols, plt_count, projection='3d')

        for fpr in reversed(np.linspace(0.0, 1.0, num=fpr_num)):
            Z = get_reward(reward_calculator, fpr, bl_size, precision=X, recall=Y)
            surf = ax.plot_surface(X, Y, Z, label=f'fpr={fpr}', antialiased=True)
            surf._facecolors2d = surf._facecolor3d  # required for label
            surf._edgecolors2d = surf._edgecolor3d  # required for label

        ax.set_xlabel('precision', labelpad=15)
        ax.set_ylabel('recall', labelpad=15)
        ax.set_zlabel('reward')
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_zlim(0, 1.0)
        ax.set_title(f'bl_size={bl_size}', y=0.85)
        ax.view_init(5, -60)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(reversed(handles), reversed(labels), ncol=3, loc='upper right', bbox_to_anchor=(0.01, 0.01, 1, 1), )
    fig.suptitle('Reward Function')
    fig.tight_layout(pad=0.01)

    plt.show()


if __name__ == '__main__':
    reward_calc = DefaultRewardCalc()
    plot_reward(reward_calc)
