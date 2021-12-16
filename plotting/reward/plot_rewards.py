"""
Plots the entire reward function, i.e., the multiplication of its components.
"""
import math

from gyms.hhh.reward import DefaultRewardCalc, MultiplicativeRewardSpecificity, RewardCalc

import numpy as np
import matplotlib.pyplot as plt

from plotting.reward.util import get_reward
from matplotlib import cm

BLACKLIST_MIN = 1  # start value of blacklist size
BLACKLIST_MAX = 10  # end value of blacklist size
NUM_BLACKLIST = 6  # number of 3D plots to generate (different blacklist sizes)
NUM_FPR = 5  # number of surfaces per 3D plot (different fpr)


def plot_reward_no_prec_comparison_rec_bl(reward_calculator: RewardCalc):  # compare bl size and recall
    px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
    fig: plt.Figure = plt.figure(figsize=(1920 * px, 1080 * px))

    plt.rcParams.update({'font.size': 18})

    recall = np.linspace(0.0, 1.0, num=100)
    bl_size = np.linspace(1.0, 100, num=100)
    X, Y = np.meshgrid(recall, bl_size)
    print(X)
    print(Y)
    ax: plt.Axes = fig.add_subplot(1, 1, 1, projection='3d')
    Z = np.zeros_like(X)
    for ri, rec in enumerate(recall):
        for fi, f in enumerate(bl_size):
            Z[fi, ri] = get_reward(reward_calculator, fpr=0, bl_size=f, precision=1, recall=rec)
    print(Z)
    surf = ax.plot_surface(X, Y, Z, antialiased=True, cmap=cm.inferno)
    ax.scatter(0.64, 6.6, get_reward(reward_calculator, fpr=0, bl_size=6.6, precision=1, recall=0.64))
    ax.scatter(1, 50, get_reward(reward_calculator, fpr=0, bl_size=50, precision=1, recall=1))
    Z.fill(get_reward(reward_calculator, fpr=0, bl_size=6.6, precision=1, recall=0.64))
    surf = ax.plot_surface(X, Y, Z, antialiased=True, label='should')
    surf._facecolors2d = surf._facecolor3d  # required for label
    surf._edgecolors2d = surf._edgecolor3d  # required for label
    Z.fill(get_reward(reward_calculator, fpr=0, bl_size=50, precision=1, recall=1))
    surf = ax.plot_surface(X, Y, Z, antialiased=True, label='all')
    surf._facecolors2d = surf._facecolor3d  # required for label
    surf._edgecolors2d = surf._edgecolor3d  # required for label
    ax.set_xlabel('Recall', labelpad=15)
    ax.set_ylabel('Number of filter rules', labelpad=15)
    ax.set_zlabel('Reward', labelpad=15)
    # ax.set_xticks([0, 0.5, 1.0])
    # ax.set_yticks([0, 0.5, 1.0])
    ax.view_init(5, -70)
    fig.tight_layout(pad=0.01)
    fig.legend()

    plt.show()


def plot_reward_no_prec_comparison(reward_calculator: RewardCalc):
    px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
    fig: plt.Figure = plt.figure(figsize=(1920 * px, 1080 * px))

    plt.rcParams.update({'font.size': 18})

    recall = np.linspace(0.0, 1.0, num=100)
    fpr = np.linspace(0.0, 1.0, num=100)
    X, Y = np.meshgrid(recall, fpr)
    print(X)
    print(Y)
    ax: plt.Axes = fig.add_subplot(1, 1, 1, projection='3d')
    Z = np.zeros_like(X)
    # for ri, rec in enumerate(reversed(recall)):
    #    for fi, f in enumerate(reversed(fpr)):
    #        Z[ri, fi] = get_reward(reward_calculator, fpr=f, bl_size=10, precision=1, recall=rec)
    Z = get_reward(reward_calculator, fpr=Y, bl_size=10, precision=1, recall=X)
    print(Z)
    surf = ax.plot_surface(X, Y, Z, antialiased=True)
    ax.set_xlabel('Recall', labelpad=15)
    ax.set_ylabel('False positive rate', labelpad=15)
    ax.set_zlabel('Reward', labelpad=15)
    # ax.set_xticks([0, 0.5, 1.0])
    # ax.set_yticks([0, 0.5, 1.0])
    ax.view_init(5, -70)
    fig.tight_layout(pad=0.01)

    plt.show()


def plot_reward_no_prec(reward_calculator: RewardCalc, bl_min=BLACKLIST_MIN, bl_max=BLACKLIST_MAX,
                        bl_num=NUM_BLACKLIST):
    px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
    fig: plt.Figure = plt.figure(figsize=(1920 * px, 1080 * px))

    plt.rcParams.update({'font.size': 18})

    fpr = np.linspace(0.0, 1.0, num=100)
    recall = np.linspace(0.0, 1.0, num=100)
    X, Y = np.meshgrid(recall, fpr)
    ax: plt.Axes = fig.add_subplot(1, 1, 1, projection='3d')
    for bl in reversed(np.linspace(bl_min, bl_max, num=bl_num, dtype=np.int)):
        Z = get_reward(reward_calculator, fpr, bl, precision=X, recall=Y)
        surf = ax.plot_surface(X, Y, Z, label=f'bl={bl}', antialiased=False)
        surf._facecolors2d = surf._facecolor3d  # required for label
        surf._edgecolors2d = surf._edgecolor3d  # required for label
    ax.set_xlabel('false positive rate', labelpad=15)
    ax.set_ylabel('recall', labelpad=15)
    ax.set_zlabel('reward', labelpad=15)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks([0, 0.5, 1.0])
    # ax.set_zlim(0, 4.0)
    ax.view_init(5, -70)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels), ncol=3)  # loc='upper right', bbox_to_anchor=(0.01, 0.01, 1, 1), )
    fig.tight_layout(pad=0.01)

    plt.show()


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
        # ax.set_zlim(0, 4.0)
        ax.set_title(f'bl_size={bl_size}', y=0.85)
        ax.view_init(5, -60)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(reversed(handles), reversed(labels), ncol=3, loc='upper right', bbox_to_anchor=(0.01, 0.01, 1, 1), )
    fig.suptitle('Reward Function')
    fig.tight_layout(pad=0.01)

    plt.show()


def plot_reward2(reward_calculator: RewardCalc):
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
    reward_calc = MultiplicativeRewardSpecificity(precision_weight=0, bl_weight=0.22, recall_weight=0.75, fpr_weight=1)
    plot_reward_no_prec(reward_calc)
