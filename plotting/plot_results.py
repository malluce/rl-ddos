import os
import struct
from collections import defaultdict
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import pandas
import numpy as np
import pandas as pd

from gyms.hhh.env import pattern_ids_to_pattern_sequence


def read_env_csv(env_file_path):
    env_csv = pandas.read_csv(env_file_path)
    cols = list(map(lambda x: str(x).strip(), list(env_csv.columns)))  # strip spaces from col names
    return pandas.read_csv(env_file_path, header=0, names=cols)  # read csv again with stripped col names


def plot_training(environment_file_path: str, pattern):
    """
    Plots the training process of an environment.csv file. Shows episodes on the x-axis
    :param pattern: the pattern for which to show the training progress
    :param environment_file_path: the environment.csv file path
   """
    env_csv = read_env_csv(environment_file_path)

    patterns = get_patterns(env_csv, pattern)

    for pat in patterns:
        plot_train_for_pattern(env_csv, environment_file_path, pat)


def plot_train_for_pattern(env_csv, environment_file_path, pat):
    print('========== TRAIN =========')
    pat_csv = filter_by_pattern(env_csv, pat)
    m = pat_csv.groupby(['Phi']).size().sort_values(ascending=False).reset_index(name='Count')
    print(f'{m}')
    plt.scatter(m['Phi'], m['Count'], marker='x', linewidths=0.5)
    plt.yscale('log')
    plt.show()
    m = pat_csv.groupby(['BlackSize']).size().sort_values(ascending=False)
    print(f'{m}')
    m = pat_csv['Precision'].mean()
    print(f'precision avg={m}')
    m = pat_csv['Recall'].mean()
    print(f'recall avg={m}')
    m = pat_csv['Reward'].mean()
    print(f'reward avg={m}')
    m = pat_csv['Phi'].mean()
    print(f'phi avg={m}')
    m = pat_csv['BlackSize'].mean()
    print(f'blacksize avg={m}')
    grouped_by_episode = pat_csv.groupby(['Episode'])
    mean = grouped_by_episode.mean()
    quantiles = get_quantiles(grouped_by_episode)
    run_id = environment_file_path.split('/')[-4]
    title = f'training (all episodes) \n {run_id}'
    if pat is not None:
        title += f'\n (pattern={pat})'
    fig = create_plots(mean, quantiles, title=title, x_label='episode', data_label='mean')
    if 'UndiscountedReturnSoFar' in pat_csv and 'DiscountedReturnSoFar' in pat_csv:  # plot return
        max_step = pat_csv.loc[:, 'Step'].max()
        last_steps = pat_csv.loc[pat_csv.loc[:, 'Step'] == max_step]
        undiscounted_return = last_steps.loc[:, 'UndiscountedReturnSoFar']
        discounted_return = last_steps.loc[:, 'DiscountedReturnSoFar']
        assert undiscounted_return.shape[0] == discounted_return.shape[0]
        ax = fig.add_subplot(2, 4, 8)
        x = range(undiscounted_return.shape[0])
        ax.plot(x, undiscounted_return, label='undiscounted')
        ax.plot(x, discounted_return, label='discounted')
        ax.set_title('Return')
        ax.set_ylim(bottom=0)
        ax.set_xlabel('episode')
        plt.legend()
    plt.show()


def get_quantiles(data: pandas.DataFrame):
    return [(q, data.quantile(q)) for q in [0.1, 0.2, 0.4, 0.6, 0.8]]


def plot(fig: plt.Figure, data, data_quantiles, cols, x, x_label, data_label, y_max=None, title=None,
         labels=None, mean_for_title=None):
    ax: plt.Axes = fig.add_subplot(2, 4, x)
    x = range(data.shape[0])
    for idx, col in enumerate(cols):
        y = data.loc[:, col]
        color = 'g' if idx == 0 else 'orange'  # lines[-1].get_color()
        for quantile in data_quantiles:
            q, quantile_value = quantile
            alpha = 0.9 - 1.5 * abs(0.5 - q)
            color = (0, 0.5, 0, alpha)  # RGBA
            y_err = quantile_value.loc[:, col]
            if labels is None:
                ax.fill_between(x, y, y_err, edgecolor=color, facecolor=color, label=f'{q} quant')
            else:
                ax.fill_between(x, y, y_err, edgecolor=color,
                                facecolor=color)  # plot without label for discounted and undiscounted

        ax.set_xlabel(x_label)

        base_title = title if title is not None else col

        title = '{} (avg={:3.2f})'.format(base_title, mean_for_title) if mean_for_title is not None else base_title
        ax.set_title(title)

        median_color = 'navy' if idx == 0 else 'red'
        if labels is None:
            ax.plot(x, y, median_color, label=data_label, linewidth=1)
        else:
            ax.plot(x, y, median_color, label=labels[idx], linewidth=1)
    if y_max is not None:
        ax.set_ylim(bottom=0, top=y_max if type(y_max) == int else y_max + 0.1)
    else:
        ax.set_ylim(bottom=0)

    return ax.get_legend_handles_labels()


def get_all_patterns(environment_csv):
    if 'SrcPattern' in environment_csv.columns and pattern is not None:
        episode_patterns = environment_csv.loc[:, ['Episode', 'SrcPattern']].groupby(['Episode'])['SrcPattern'].apply(
            pattern_ids_to_pattern_sequence).apply(lambda x: str.replace(x, ' ', '')).reset_index(name='pat')
        unique_patterns = episode_patterns['pat'].drop_duplicates()
        return list(unique_patterns)
    else:
        raise ValueError('Env CSV does not contain any patterns')


def filter_by_pattern(environment_csv, pattern):
    if 'SrcPattern' in environment_csv.columns and pattern is not None:
        episode_patterns = environment_csv.loc[:, ['Episode', 'SrcPattern']].groupby(['Episode'])['SrcPattern'].apply(
            pattern_ids_to_pattern_sequence).apply(lambda x: str.replace(x, ' ', ''))
        episodes = []
        for x, y in environment_csv.loc[:, ['Episode', 'SrcPattern']].groupby(['Episode'])['Episode']:
            episodes.append(x)
        res = pd.DataFrame(data=zip(episodes, episode_patterns), columns=['ep', 'pat'])
        pattern_episodes = list(res[res['pat'] == pattern]['ep'])  # the episodes which contain the queried pattern
        return environment_csv[environment_csv['Episode'].isin(pattern_episodes)]
    else:
        return environment_csv


def get_rows_with_episode_in(environment_csv, episodes):
    filtered_rows = environment_csv.loc[environment_csv['Episode'].isin(episodes),
                    :]  # get rows where episode in episodes
    return filtered_rows.groupby(['Step'])  # return all columns grouped for each step


def plot_episode_behavior(environment_file_path, pattern, window: Tuple[int, int]):
    """
    Plots the metrics of an environment.csv file.
    :param environment_file_path: the environment.csv file path
    :param window: the episodes to plot (relative to the last episode)
    """
    env_csv = read_env_csv(environment_file_path)

    patterns = get_patterns(env_csv, pattern)

    for pat in patterns:
        plot_ep_behav_for_pattern(env_csv, environment_file_path, pat, window)


def get_patterns(env_csv, pattern):
    if pattern == 'all':
        patterns = get_all_patterns(env_csv)
    else:
        patterns = [pattern]  # might be [None] for non-pattern traces
    return patterns


def plot_ep_behav_for_pattern(env_csv, environment_file_path, pat, window):
    pat_csv = filter_by_pattern(env_csv, pat)
    print('========== EVAL =========')
    pat_csv = filter_by_pattern(env_csv, pat)
    m = pat_csv.groupby(['Phi']).size().sort_values(ascending=False).reset_index(name='Count')
    print(f'{m}')
    plt.scatter(m['Phi'], m['Count'], marker='x', linewidths=0.5)
    plt.yscale('log')
    plt.show()
    m = pat_csv.groupby(['BlackSize']).size().sort_values(ascending=False)
    print(f'{m}')
    m = pat_csv['Precision'].mean()
    print(f'precision avg={m}')
    m = pat_csv['Recall'].mean()
    print(f'recall avg={m}')
    m = pat_csv['Reward'].mean()
    print(f'reward avg={m}')
    m = pat_csv['Phi'].mean()
    print(f'phi avg={m}')
    m = pat_csv['BlackSize'].mean()
    print(f'blacksize avg={m}')
    eps = pat_csv.loc[:, 'Episode']
    unique_eps = sorted(list(set(eps)))  # get all unique episode numbers (some are not included in csv files)
    eps_to_show = unique_eps[len(unique_eps) - window[0]:len(unique_eps) - window[1]]
    last = get_rows_with_episode_in(pat_csv, eps_to_show)
    last_median = last.median()
    last_means = last.mean()
    last_quantiles = get_quantiles(last)
    run_id = environment_file_path.split('/')[-4]
    title = f'{window[0] - window[1]} eval episodes (first:{unique_eps[len(unique_eps) - window[0]]}, last:{unique_eps[len(unique_eps) - window[1] - 1]}) \n {run_id}'
    if pat is not None:
        title += f'\n (pattern={pat})'
    # plot common data
    fig = create_plots(last_median, last_quantiles,
                       title=title,
                       x_label='step',
                       data_label='median', means_for_title=last_means)
    if 'UndiscountedReturnSoFar' in pat_csv and 'DiscountedReturnSoFar' in pat_csv:
        # plot return until step
        handles, labels = plot(fig, last_median, last_quantiles,
                               ['UndiscountedReturnSoFar', 'DiscountedReturnSoFar'],
                               8, 'step',
                               'median', title='Return until step', labels=['undiscounted', 'discounted'])
        plt.legend(handles, labels)
    plt.show()


def create_plots(data, quantiles, title, x_label, data_label, means_for_title=None):
    fig: plt.Figure = plt.figure(figsize=(16, 8))
    title_means = defaultdict(lambda: None)
    if means_for_title is not None:
        for col in ['Precision', 'Recall', 'BlackSize', 'FPR', 'Reward']:
            title_means[col] = means_for_title[col].mean()
    plot(fig, data, quantiles, ['Phi'], 1, y_max=1.0, x_label=x_label, data_label=data_label)
    if data['MinPrefix'].max() > 1:
        y_max_pref = 32  # L column
        title = 'MinPrefix'
    else:
        y_max_pref = 1.0  # Thresh Column
        title = 'Performance Threshold'
    plot(fig, data, quantiles, ['MinPrefix'], 2, y_max=y_max_pref, x_label=x_label, data_label=data_label, title=title)
    plot(fig, data, quantiles, ['Precision'], 3, y_max=1.0, x_label=x_label, data_label=data_label,
         mean_for_title=title_means['Precision'])
    plot(fig, data, quantiles, ['Recall'], 4, y_max=1.0, x_label=x_label, data_label=data_label,
         mean_for_title=title_means['Recall'])
    plot(fig, data, quantiles, ['BlackSize'], 5, x_label=x_label, data_label=data_label,
         mean_for_title=title_means['BlackSize'])
    plot(fig, data, quantiles, ['FPR'], 6, y_max=1.0, x_label=x_label, data_label=data_label,
         mean_for_title=title_means['FPR'])
    reward_max = data['Reward'].max() if data['Reward'].max() > 1.0 else 1.0
    # reward_max = 400
    handles, labels = plot(fig, data, quantiles, ['Reward'], 7, x_label=x_label, data_label=data_label,
                           y_max=reward_max, mean_for_title=title_means['Reward'])

    fig.suptitle(title)

    fig.tight_layout()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.01, 0.01, 1, 1), ncol=3)
    return fig


def plot_kickoff(fig: plt.Figure, data, data_quantiles, cols, x, x_label, data_label, y_max=None, title=None,
                 labels=None, mean_for_title=None):
    ax: plt.Axes = fig.add_subplot(1, 4, x)
    x = range(data.shape[0])
    for idx, col in enumerate(cols):
        y = data.loc[:, col]
        color = 'g' if idx == 0 else 'orange'  # lines[-1].get_color()
        for quantile in data_quantiles:
            q, quantile_value = quantile
            alpha = 0.9 - 1.5 * abs(0.5 - q)
            y_err = quantile_value.loc[:, col]
            if labels is None:
                if q == 0.2 or q == 0.4:
                    ax.plot(x, y_err, color=color, alpha=alpha, label=f'{q}/{0.5 + abs(0.5 - q)} quantile')
                else:
                    ax.plot(x, y_err, color=color, alpha=alpha)
            else:
                ax.plot(x, y_err, color=color, alpha=alpha)  # plot without label for discounted and undiscounted
            ax.fill_between(x, y, y_err, color=color, alpha=alpha)

        ax.set_xlabel(x_label)

        base_title = title if title is not None else col

        title = '{} (avg={:3.2f})'.format(base_title, mean_for_title) if mean_for_title is not None else base_title
        ax.set_title(title)

        median_color = 'navy' if idx == 0 else 'red'
        if labels is None:
            ax.plot(x, y, median_color, label=data_label)
        else:
            ax.plot(x, y, median_color, label=labels[idx])
    if y_max is not None:
        ax.set_ylim(bottom=0, top=y_max if type(y_max) == int else y_max + 0.1)
    else:
        ax.set_ylim(bottom=0)

    return ax.get_legend_handles_labels()


def plot_training_kickoff(environment_file_path: str):
    env_csv = read_env_csv(environment_file_path)
    grouped_by_episode = env_csv.groupby(['Episode'])
    mean = grouped_by_episode.median()
    quantiles = get_quantiles(grouped_by_episode)

    data = mean
    data_label = 'median'
    x_label = 'training episode'

    fig: plt.Figure = plt.figure(figsize=(16, 4))
    plot_kickoff(fig, data, quantiles, ['Precision'], 1, y_max=1.0, x_label=x_label, data_label=data_label)
    handles, labels = plot_kickoff(fig, data, quantiles, ['FPR'], 2, y_max=1.0, x_label=x_label, data_label=data_label,
                                   title='False positive rate')
    plot_kickoff(fig, data, quantiles, ['Recall'], 3, y_max=1.0, x_label=x_label, data_label=data_label)
    plot_kickoff(fig, data, quantiles, ['BlackSize'], 4, x_label=x_label, data_label=data_label,
                 title='Number of filter rules')

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0, 0, 1, 1), ncol=5)
    plt.suptitle(' ')
    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    matplotlib.rcParams.update({'font.size': 15})

    ds_base = '/srv/bachmann/data/hafner/dqn_20210930-072309/datastore'
    # ds_base = '/srv/bachmann/data/ppo/ppo_20211006-073532/datastore'
    # ds_base = '/srv/bachmann/data/dqn/dqn_20211001-080033/datastore'
    if ds_base.split('/')[-2].startswith('ppo'):
        train_dir = 'train1'
    else:
        train_dir = 'train'
    train_path = os.path.join(ds_base, train_dir, 'environment.csv')
    eval_path = os.path.join(ds_base, 'eval', 'environment.csv')
    paths_exist = os.path.exists(train_path) and os.path.exists(eval_path)
    if not paths_exist:
        raise ValueError('Paths do not exist')

    pattern = None
    plot_training(environment_file_path=train_path, pattern=pattern)
    plot_episode_behavior(environment_file_path=eval_path, pattern=pattern, window=(10, 0))
