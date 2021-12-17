import os
import struct
from collections import defaultdict
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import pandas
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb

from gyms.hhh.env import pattern_ids_to_pattern_sequence


# TODO delete, for inter slides

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


def print_csv_stats(csv):
    # m = csv.groupby(['Phi']).size().sort_values(ascending=False).reset_index(name='Count')
    # print(f'{m}')
    # plt.scatter(m['Phi'], m['Count'], marker='x', linewidths=0.5)
    # plt.yscale('log')
    # plt.show()
    # m = csv.groupby(['BlackSize']).size().sort_values(ascending=False)
    # print(f'{m}')
    m = csv['Precision'].mean()
    print(f'precision avg={m}')
    m = csv['Recall'].mean()
    print(f'recall avg={m}')
    m = csv['Reward'].mean()
    print(f'reward avg={m}')
    m = csv['Phi'].mean()
    print(f'phi avg={m}')
    m = csv['BlackSize'].mean()
    print(f'blacksize avg={m}')
    m = csv['FPR'].mean()
    print(f'fpr avg={m}')


def plot_train_for_pattern(env_csv, environment_file_path, pat):
    print('========== TRAIN =========')
    pat_csv = filter_by_pattern(env_csv, pat)

    print(pat_csv[:1000].loc[:, 'Phi'].mean(), pat_csv[:1000].loc[:, 'Phi'].std())
    print(pat_csv[-1000:].loc[:, 'Phi'].mean(), pat_csv[-1000:].loc[:, 'Phi'].std())
    print(pat_csv[:1000].loc[:, 'Thresh'].mean(), pat_csv[:1000].loc[:, 'Thresh'].std())
    print(pat_csv[-1000:].loc[:, 'Thresh'].mean(), pat_csv[-1000:].loc[:, 'Thresh'].std())
    plt.boxplot([pat_csv[:1000].loc[:, 'Phi'], pat_csv[-1000:].loc[:, 'Phi']])
    plt.show()

    # print_csv_stats(pat_csv)
    grouped_by_episode = pat_csv.groupby(['Episode'])

    mean = grouped_by_episode.mean()
    quantiles = get_quantiles(grouped_by_episode)
    run_id = environment_file_path.split('/')[-4]
    title = f'training (all episodes) \n {run_id}'
    if pat is not None:
        title += f'\n (pattern={pat})'
    create_plots(mean, quantiles, title=title, x_label='episode', data_label='mean')
    plt.show()


def get_quantiles(data: pandas.DataFrame):
    return [(q, data.quantile(q)) for q in [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]]


def plot(fig: plt.Figure, data, data_quantiles, cols, x, x_label, data_label, y_max=None, title=None,
         labels=None, mean_for_title=None):
    ax: plt.Axes = fig.add_subplot(2, 3, x)
    x = range(data.shape[0])
    x = list(map(lambda y: y * 10, x))
    for idx, col in enumerate(cols):
        y = data.loc[:, col]
        for quantile in data_quantiles:
            q, quantile_value = quantile
            alpha = 0.9 - 1.5 * abs(0.5 - q)
            color = (0, 0.5, 0, alpha)  # RGBA
            y_err = quantile_value.loc[:, col]
            if labels is None:
                if q in [0.1, 0.2, 0.4]:
                    ax.fill_between(x, y, y_err, edgecolor=color, facecolor=color,
                                    label=f'{q}/{0.5 + abs(0.5 - q)} quantile')
                else:
                    ax.fill_between(x, y, y_err, edgecolor=color, facecolor=color)
            else:
                ax.fill_between(x, y, y_err, edgecolor=color,
                                facecolor=color)  # plot without label for discounted and undiscounted

        ax.set_xlabel(x_label)

        base_title = title if title is not None else col

        title = '{} (mean={:3.3f})'.format(base_title, mean_for_title) if mean_for_title is not None else base_title
        ax.set_title(title)

        median_color = 'navy' if idx == 0 else 'red'
        if labels is None:
            ax.plot(x, y, median_color, label=data_label, linewidth=1)
        else:
            ax.plot(x, y, median_color, label=labels[idx], linewidth=1)
    if y_max is not None:
        ax.set_ylim(bottom=-0.05, top=y_max if type(y_max) == int else y_max + 0.1)
    else:
        ax.set_ylim(bottom=-0.05)

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
    return environment_csv.loc[environment_csv['Episode'].isin(episodes), :]  # get rows where episode in episodes


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


def plot_time_index(last, metric, ax, label, y_top=None):
    new_df = last.loc[:, ['Episode', 'Step', metric]]

    new_df = pd.DataFrame(
        pd.DataFrame(data=new_df[metric].str.split(';').tolist(),
                     index=[last.Episode, last.Step]).stack(), columns=[metric])
    new_df.index = new_df.index.set_names(['Episode', 'Step', 'Index'])
    new_df.reset_index(inplace=True)
    if new_df[metric][0] == ' ':
        return False  # nothing to plot here (if rejection unused)

    if label == 'rule-gran':
        # number of rules per prefixlength
        def get_occuring_prefix_lengths(entry):
            return list(map(lambda x: x.split(':')[0].strip(), entry.split('-')))

        rule_prefix_lengths = new_df[metric].apply(get_occuring_prefix_lengths)
        unique_prefix_lengths = rule_prefix_lengths.explode().unique()
        unique_prefix_lengths = sorted(np.delete(unique_prefix_lengths, np.where(unique_prefix_lengths == '')))
        for pref_len in unique_prefix_lengths:
            new_df[f'{pref_len}'] = 0

        def fill_pref_len_column(entry):
            len_cnt = list(
                map(lambda x: (x.split(':')[0].strip(), int(x.split(':')[1].strip())) if len(
                    x.split(':')) == 2 else None, entry[metric].split('-')))
            for l in len_cnt:
                if l is None:
                    continue
                length, count = l
                entry[length] += count
            return entry

        new_df = new_df.apply(fill_pref_len_column, axis=1)

    if metric == 'CacheIdx':
        # total cache rules
        new_df[metric] = new_df[metric].apply(
            lambda x: sum(
                map(lambda z: int(z[1]) if len(z) == 2 else 0, map(lambda y: y.split(':'), x.split('-')))))

    new_df = new_df.astype(float)
    new_df['Index'] = new_df['Index'] + new_df['Step'] * 10
    new_df['Index'] = new_df['Index'] + 20  # offset due to pre-sampling and first random action

    if label == 'rule-gran':
        new_df_grouped = new_df.groupby(['Index'])
        mean = new_df_grouped.mean().iloc[:, 3:]
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        linestyle = ['-'] * len(colors) + ['--'] * len(colors)
        plt.rcParams["axes.prop_cycle"] += plt.cycler("linestyle", linestyle)
        ax.set_prop_cycle(plt.rcParams["axes.prop_cycle"])
        for i, col in enumerate(mean.columns):
            ax.plot(mean[col], linewidth=1.5, label=f'/{col}', linestyle=linestyle[i])
        ax.legend(ncol=int(len(mean.columns) / 2), loc='best')
        ax.set_ylabel('Cache Utilization')
    else:
        new_df_grouped = new_df.loc[:, ['Index', metric]].groupby(['Index'])
        new_df_mean = new_df_grouped.mean()

        new_df_quantiles = get_quantiles(new_df_grouped)

        lines = ax.plot(new_df_mean, linewidth=1.5, label=label)
        plot_color = to_rgb(lines[0].get_color())
        x = new_df['Index'].unique()
        for quantile in new_df_quantiles:
            q, quantile_value = quantile
            alpha = 0.7 - 1.5 * abs(0.5 - q)

            color = (plot_color[0], plot_color[1], plot_color[2], alpha)  # RGBA
            y = new_df_mean.loc[:, metric]
            y_err = quantile_value.loc[:, metric]
            ax.fill_between(x, y, y_err,
                            edgecolor=color,
                            facecolor=color)
        ax.legend()

    ax.set_ylim(bottom=0)
    if y_top is not None:
        ax.set_ylim(bottom=0, top=y_top)
    ax.set_xlim(left=0, right=600)
    ax.vlines(range(0, int(new_df['Index'].max()), 10), ymin=0, ymax=ax.get_ylim()[1], linestyles='--',
              linewidth=1, alpha=0.6)
    return True


def plot_ep_behav_for_pattern(env_csv, environment_file_path, pat, window):
    print('========== EVAL =========')
    pat_csv = filter_by_pattern(env_csv, pat)
    print_csv_stats(pat_csv)
    eps = pat_csv.loc[:, 'Episode']
    unique_eps = sorted(list(set(eps)))  # get all unique episode numbers (some are not included in csv files)
    eps_to_show = unique_eps[len(unique_eps) - window[0]:len(unique_eps) - window[1]]
    last = get_rows_with_episode_in(pat_csv, eps_to_show)

    # title
    title = None

    # plot per time idx
    if all(map(lambda x: x in last.columns, ['PrecisionIdx', 'FPRIdx', 'RecallIdx', 'BlackSizeIdx', 'CacheIdx'])):
        fig: plt.Figure = plt.figure(figsize=(16, 9))
        plotted_at_least_one = False
        for idx, val in enumerate(
                zip(['CacheIdx', 'CacheIdx', 'BlackSizeIdx'],
                    ['Cache Utilization (total)', 'rule-gran', 'Number of Filter Rules'])):
            metric, label = val
            ax: plt.Axes = fig.add_subplot(3, 1, idx + 1)
            ax.set_xlabel('time index')
            plotted = plot_time_index(last, metric, ax, label=label)
            plotted_at_least_one = plotted_at_least_one or plotted
        if plotted_at_least_one:
            fig.suptitle(title)
            plt.tight_layout()
            plt.show()
        else:
            plt.clf()

        fig: plt.Figure = plt.figure(figsize=(16, 9))
        plotted_at_least_one = False
        for idx, val in enumerate(
                zip(['PrecisionIdx', 'RecallIdx', 'FPRIdx'], ['Precision', 'Recall', 'False positive rate'])):
            metric, label = val
            ax: plt.Axes = fig.add_subplot(3, 1, idx + 1)
            ax.set_xlabel('time index')
            plotted = plot_time_index(last, metric, ax, label=label, y_top=1.1)
            plotted_at_least_one = plotted_at_least_one or plotted
        if plotted_at_least_one:
            fig.suptitle(title)
            plt.tight_layout()
            plt.show()
        else:
            plt.clf()
    last.loc[:, 'Thresh'] = 1 - last.loc[:, 'Thresh']
    last = last.groupby(['Step'])  # return all columns grouped for each step

    last_median = last.median()
    last_means = last.mean()
    last_quantiles = get_quantiles(last)

    # plot per time step
    create_plots(last_median, last_quantiles,
                 title=title,
                 x_label='Time index',
                 data_label='median', means_for_title=last_means)
    plt.show()


def create_plots(data, quantiles, title, x_label, data_label, means_for_title=None):
    fig: plt.Figure = plt.figure(figsize=(16, 8))
    title_means = defaultdict(lambda: None)
    if means_for_title is not None:
        for col in ['Precision', 'Recall', 'BlackSize', 'FPR', 'Reward', 'Phi', 'Thresh', 'MinPrefix']:
            if col in data:
                title_means[col] = means_for_title[col].mean()
    plot(fig, data, quantiles, ['Phi'], 1, y_max=1.0, x_label=x_label, data_label=data_label, title='$\phi$')
    plot(fig, data, quantiles, ['MinPrefix'], 2, y_max=32, x_label=x_label, data_label=data_label, title='$L$')
    if 'Thresh' in data and data['Thresh'].max() > -1:
        plot(fig, data, quantiles, ['Thresh'], 3, y_max=1.0, x_label=x_label, data_label=data_label,
             title='FPR Threshold')
        offset = 1
    else:
        offset = 0
    plot(fig, data, quantiles, ['Recall'], 3 + offset, y_max=1.0, x_label=x_label, data_label=data_label,
         mean_for_title=title_means['Recall'])
    plot(fig, data, quantiles, ['BlackSize'], 4 + offset, x_label=x_label, data_label=data_label,
         mean_for_title=title_means['BlackSize'], title='Number of rules')
    handles, labels = plot(fig, data, quantiles, ['FPR'], 5 + offset, y_max=1.0, x_label=x_label, data_label=data_label,
                           mean_for_title=title_means['FPR'])

    fig.tight_layout()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.95, 0.375), ncol=1)
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


def plot_training_kickoff(environment_file_path: str, pat):
    env_csv = read_env_csv(environment_file_path)
    env_csv = filter_by_pattern(env_csv, pat)
    grouped_by_episode = env_csv.groupby(['Episode'])
    mean = grouped_by_episode.mean()
    quantiles = get_quantiles(grouped_by_episode)

    data = mean
    data_label = 'mean'
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
    plt.rc('axes', labelsize=18)
    plt.rc('figure', titlesize=18)


    def plot_hl(ds_base, pattern, window):
        if ds_base.split('/')[-2].startswith('ppo'):
            train_dir = 'train1'
        else:
            train_dir = 'train'
        train_path = os.path.join(ds_base, train_dir, 'environment.csv')
        eval_path = os.path.join(ds_base, 'eval', 'environment.csv')
        paths_exist = os.path.exists(train_path) and os.path.exists(eval_path)
        if not paths_exist:
            raise ValueError('Paths do not exist')

        # plot_training_kickoff(train_path, pattern)
        # plot_training(environment_file_path=train_path, pattern=pattern)
        plot_episode_behavior(environment_file_path=eval_path, pattern=pattern, window=window)


    first_pattern = 'ssdp'
    second_pattern = 'ntp'

    pattern = f'{first_pattern}->{first_pattern}+{second_pattern}->{second_pattern}'
    window = (30, 0)
    # pattern = 'ntp;bot'
    # pattern = 'T3'
    pattern = 'T3WithoutPause'
    # ds_base = '/srv/bachmann/data/td3/td3_20211109-154010/datastore'
    # ds_base = '/home/bachmann/test-pycharm/data/dqn_20210911-132807/datastore'
    ds_base = '/srv/bachmann/data/dqn/dqn_20211117-075247/datastore'  # rej
    # ds_base = '/srv/bachmann/data/dqn/dqn_20211117-075336/datastore'  # phi,l
    # ds_base = '/srv/bachmann/data/dqn/dqn_20211109-070809/datastore'

    ds_base = '/srv/bachmann/data/ppo/ppo_20211110-083716/datastore'
    ds_base = '/srv/bachmann/data/ppo/ppo_20211122-081003/datastore'  # T3 rej
    ds_base = '/srv/bachmann/data/ppo/ppo_20211122-095112/datastore'  # T3 phi/l correct ub
    # ds_base = '/srv/bachmann/data/sac/sac_20211112-072533/datastore'
    plot_hl(ds_base, pattern, window)
