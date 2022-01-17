import datetime
import os
import struct
import sys
from collections import defaultdict
from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import pandas
import numpy as np
import pandas as pd
from matplotlib import ticker
from matplotlib.colors import to_rgb

from gyms.hhh.env import pattern_ids_to_pattern_sequence
from plotting.plot_tb_csv import smooth

TITLESIZE = 18
LABELSIZE = 18
LEGENDSIZE = 11
TICKSIZE = 8


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


def plot(fig: plt.Figure, data, data_quantiles, cols, x, x_label, data_label, y_min=None, y_max=None, title=None,
         labels=None, mean_for_title=None, nrow=None, ncol=None):
    ax: plt.Axes = fig.add_subplot(nrow, ncol, x)
    x = range(data.shape[0])
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
                                    label=f'{q}/{0.5 + abs(0.5 - q)} Quantile')
                else:
                    ax.fill_between(x, y, y_err, edgecolor=color, facecolor=color)
            else:
                ax.fill_between(x, y, y_err, edgecolor=color,
                                facecolor=color)  # plot without label for discounted and undiscounted

        ax.set_xlabel(x_label, fontsize=LABELSIZE)

        base_title = title if title is not None else col

        title = '{} (avg={:3.3f})'.format(base_title, mean_for_title) if mean_for_title is not None else base_title
        ax.set_title(title, fontsize=TITLESIZE)

        median_color = 'navy' if idx == 0 else 'red'
        if labels is None:
            ax.plot(x, y, median_color, label=data_label, linewidth=1)
        else:
            ax.plot(x, y, median_color, label=labels[idx], linewidth=1)
    if y_max is not None:
        ax.set_ylim(bottom=y_min if y_min is not None else 0, top=y_max if type(y_max) == int else y_max + 0.1)
    else:
        ax.set_ylim(bottom=y_min if y_min is not None else 0)
    ax.tick_params(labelsize=LABELSIZE)
    ax.grid()
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


def plot_episode_behavior(environment_file_path, pattern, window: Tuple[int, int], fancy):
    """
    Plots the metrics of an environment.csv file.
    :param environment_file_path: the environment.csv file path
    :param window: the episodes to plot (relative to the last episode)
    :param fancy: whether to plot it fancy (for thesis/presentation) or with more info and ugly
    """
    env_csv = read_env_csv(environment_file_path)

    patterns = get_patterns(env_csv, pattern)

    for pat in patterns:
        plot_ep_behav_for_pattern(env_csv, environment_file_path, pat, window, fancy)


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
    if new_df[metric].max() == ' ':
        return False  # nothing to plot here (if rejection unused)

    if label == 'rule-gran':
        # number of rules per prefixlength
        def get_occuring_prefix_lengths(entry):
            return list(map(lambda x: x.split(':')[0].strip(), entry.split('-')))

        rule_prefix_lengths = new_df[metric].apply(get_occuring_prefix_lengths)
        unique_prefix_lengths = rule_prefix_lengths.explode().unique()
        unique_prefix_lengths = sorted(np.delete(unique_prefix_lengths, np.where(unique_prefix_lengths == '')))
        for pref_len in unique_prefix_lengths:
            # if pref_len in ['17', '18', '19']:  # TODO remove
            new_df[f'{pref_len}'] = 0

        def fill_pref_len_column(entry):
            len_cnt = list(
                map(lambda x: (x.split(':')[0].strip(), int(x.split(':')[1].strip())) if len(
                    x.split(':')) == 2 else None, entry[metric].split('-')))
            for l in len_cnt:
                if l is None:
                    continue
                length, count = l
                # if length in ['17', '18', '19']:  # TODO remove
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


def plot_ep_behav_for_pattern(env_csv, environment_file_path, pat, window, fancy):
    print('========== EVAL =========')
    pat_csv = filter_by_pattern(env_csv, pat)
    print_csv_stats(pat_csv)
    eps = pat_csv.loc[:, 'Episode']
    unique_eps = sorted(list(set(eps)))  # get all unique episode numbers (some are not included in csv files)
    eps_to_show = unique_eps[len(unique_eps) - window[0]:len(unique_eps) - window[1]]
    last = get_rows_with_episode_in(pat_csv, eps_to_show)

    # title
    if not fancy:
        run_id = environment_file_path.split('/')[-4]
        title = f'{window[0] - window[1]} eval episodes (first:{unique_eps[len(unique_eps) - window[0]]}, last:{unique_eps[len(unique_eps) - window[1] - 1]}) \n {run_id}'
        if pat is not None:
            title += f'\n (pattern={pat})'
    else:
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

    last = last.groupby(['Step'])  # return all columns grouped for each step
    last_median = last.median()
    last_means = last.mean()
    last_quantiles = get_quantiles(last)

    # plot per time step
    create_plots(last_median, last_quantiles,
                 title=title,
                 x_label='Adaptation step',
                 data_label='Median', means_for_title=last_means, fancy=fancy)


def create_plots(data, quantiles, title, x_label, data_label, means_for_title=None, fancy=False):
    def finalize():
        if title is not None:
            fig.suptitle(title)

        fig.tight_layout()
        fig.gca().legend(handles, labels, fontsize=LEGENDSIZE, ncol=1)
        # fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.95, 0.375), ncol=1)
        plt.show()

    print(data.loc[:, ['Precision', 'Recall', 'BlackSize', 'FPR', 'Reward', 'Phi', 'Thresh', 'MinPrefix']])
    print('=====direct attack:\n' + str(
        data.loc[0:22, ['Precision', 'Recall', 'BlackSize', 'FPR', 'Reward', 'Phi', 'Thresh', 'MinPrefix']].mean()))
    print('=====no attack:\n' + str(
        data.loc[28:34, ['Precision', 'Recall', 'BlackSize', 'FPR', 'Reward', 'Phi', 'Thresh', 'MinPrefix']].mean()))
    print('=====reflector attack:\n' + str(
        data.loc[37:, ['Precision', 'Recall', 'BlackSize', 'FPR', 'Reward', 'Phi', 'Thresh', 'MinPrefix']].mean()))
    print('=====total:\n' + str(
        data.loc[:, ['Precision', 'Recall', 'BlackSize', 'FPR', 'Reward', 'Phi', 'Thresh', 'MinPrefix']].mean()))

    if fancy:
        nrow = 1
        ncol = 3
    else:
        nrow = 2
        ncol = 4

    fig: plt.Figure = plt.figure(figsize=(15, 5) if fancy else (16, 9))
    title_means = defaultdict(lambda: None)
    if means_for_title is not None:
        for col in ['Precision', 'Recall', 'BlackSize', 'FPR', 'Reward', 'Phi', 'Thresh', 'MinPrefix']:
            if col in data:
                title_means[col] = means_for_title[col].mean() if not fancy else None
    plot(fig, data, quantiles, ['Phi'], 1, y_max=0.25, x_label=x_label, data_label=data_label,
         mean_for_title=title_means['Phi'], title='Frequency threshold $\phi$', nrow=nrow, ncol=ncol)
    handles, labels = plot(fig, data, quantiles, ['MinPrefix'], 2, y_max=32, x_label=x_label, data_label=data_label,
                           mean_for_title=title_means['MinPrefix'], title='Minimum prefix length $L$', nrow=nrow,
                           ncol=ncol)
    if 'Thresh' in data and data['Thresh'].max() > -1:
        plot(fig, data, quantiles, ['Thresh'], 3, y_min=0.84, y_max=0.95, x_label=x_label, data_label=data_label,
             mean_for_title=title_means['Thresh'], title='Performance threshold pthresh', nrow=nrow, ncol=ncol)
        offset = 1
    else:
        offset = 0

    if fancy:
        ncol = 4
        offset = -2
        finalize()
        fig: plt.Figure = plt.figure(figsize=(15, 5) if fancy else (16, 9))

    plot(fig, data, quantiles, ['Precision'], 3 + offset, y_max=1.0, x_label=x_label, data_label=data_label,
         mean_for_title=title_means['Precision'], nrow=nrow, ncol=ncol)
    plot(fig, data, quantiles, ['Recall'], 4 + offset, y_max=1.0, x_label=x_label, data_label=data_label,
         mean_for_title=title_means['Recall'], nrow=nrow, ncol=ncol)
    plot(fig, data, quantiles, ['BlackSize'], 5 + offset, x_label=x_label, data_label=data_label,
         mean_for_title=title_means['BlackSize'], title='Number of filter rules', nrow=nrow, ncol=ncol)
    handles, labels = plot(fig, data, quantiles, ['FPR'], 6 + offset, y_max=1.0, x_label=x_label, data_label=data_label,
                           mean_for_title=title_means['FPR'], title='False positive rate', nrow=nrow, ncol=ncol)
    if not fancy:
        reward_max = data['Reward'].max() if data['Reward'].max() > 1.0 else 1.0
        plot(fig, data, quantiles, ['Reward'], 7 + offset, x_label=x_label, data_label=data_label,
             y_max=reward_max, mean_for_title=title_means['Reward'], nrow=nrow, ncol=ncol)
    finalize()


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


def plot_training_curves(ep_paths: List[str], pattern, show_all=False, label_agent_suffixes=None):
    matplotlib.rcParams.update({'font.size': 13})
    plt.rc('axes', labelsize=18)
    plt.rc('figure', titlesize=18)

    STEPS_PER_EPISODE = 58
    NUM_EVAL_EPISODES = 9
    EVAL_PERIODICITY = 3000
    plt.figure(dpi=200)
    for i, base_path in enumerate(ep_paths):
        agent_name = base_path.split('/')[-2].split('_')[0].upper()
        if label_agent_suffixes is not None:
            agent_name += label_agent_suffixes[i]

        if agent_name == 'PPO':
            INITIAL_STEPS = 0
            date = base_path.split('/')[-2].split('_')[1].split('-')[0]
            date = datetime.datetime.strptime(date, '%Y%m%d')
            if date <= datetime.datetime(2022, 1, 11):
                NUM_EVAL_EPISODES = 3
                EVAL_PERIODICITY = 6000
        else:
            INITIAL_STEPS = 1200

        train_path, eval_path = get_paths(base_path, env=False)
        train_csv = filter_by_pattern(read_env_csv(train_path), pattern)
        eval_csv = filter_by_pattern(read_env_csv(eval_path), pattern)
        # replace buggy episode counter by pandas index
        train_csv.iloc[:, 0] = train_csv.index.values
        eval_csv.iloc[:, 0] = eval_csv.index.values

        train_csv.iloc[:, 1] = (STEPS_PER_EPISODE - 1) + STEPS_PER_EPISODE * train_csv.iloc[:, 0]
        print(train_csv.iloc[:, 1])

        eval_csv.iloc[:, 0] = (eval_csv.iloc[:, 0]) // NUM_EVAL_EPISODES  # evaluation round
        eval_csv = eval_csv.groupby(['Episode'], as_index=False).mean()  # mean return per evaluation round
        eval_csv.iloc[0, 1] = INITIAL_STEPS
        eval_csv.iloc[:, 1] = eval_csv.iloc[:, 0] * EVAL_PERIODICITY
        print(f'mean over last 10 eval: {agent_name, eval_csv.iloc[-10:, -1].mean()}')
        print(f'std over all eval: {agent_name, eval_csv.iloc[:, -1].std()}')

        if len(ep_paths) == 1 or show_all:
            plt.plot(train_csv.iloc[:, 1], smooth(np.array(train_csv.iloc[:, -1])),
                     label='training' if not show_all else f'{agent_name} training')
            plt.plot(eval_csv.iloc[:, 1], eval_csv.iloc[:, -1],
                     label='evaluation (no exploration)' if not show_all else f'{agent_name} evaluation (no exploration)')
            if not show_all:
                plt.title(f'{agent_name} agent')
        else:
            plt.plot(eval_csv.iloc[:, 1], eval_csv.iloc[:, -1],
                     label=f'{agent_name} evaluation (no exploration)')

    plt.xlabel('Adaptation step in training environment')
    plt.ylabel('Undiscounted return')
    plt.ylim(bottom=0)
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{}'.format(int(x / 1000)) + 'K'))
    # plt.xlim(left=0)
    plt.tight_layout()
    plt.legend(fontsize=10)
    plt.show()


def get_paths(base_path, env=True):
    if base_path.split('/')[-2].startswith('ppo'):
        train_dir = 'train1'
    else:
        train_dir = 'train'
    train_path = os.path.join(base_path, train_dir, 'environment.csv' if env else 'episodes.csv')
    eval_path = os.path.join(base_path, 'eval', 'environment.csv' if env else 'episodes.csv')
    if 'eval-baseline' in base_path:
        if not os.path.exists(train_path):
            raise ValueError('Paths do not exist')
    else:
        paths_exist = os.path.exists(train_path) and os.path.exists(eval_path)
        if not paths_exist:
            raise ValueError('Paths do not exist')
    return train_path, eval_path


if __name__ == '__main__':
    # matplotlib.rcParams.update({'font.size': 15})
    # plt.rc('axes', labelsize=18)
    # plt.rc('figure', titlesize=18)

    def plot_hl(ds_base, pattern, window, fancy=False):
        # if not isinstance(ds_base, List):
        #    ds_base = [ds_base]
        #
        # train_paths = []
        # eval_paths = []
        # for ds in ds_base:
        #    train_path, eval_path = get_paths(ds)
        #    train_paths.append(train_path)
        #    eval_paths.append(eval_path)
        train_path, eval_path = get_paths(ds_base)
        # plot_training_kickoff(train_paths, pattern)
        # plot_training(environment_file_path=train_paths, pattern=pattern)
        # plot_training(environment_file_path=eval_paths, pattern=pattern)
        # plot_episode_behavior(environment_file_path=eval_paths, pattern=pattern, window=window, fancy=fancy)
        plot_episode_behavior(environment_file_path=train_path, pattern=pattern, window=window, fancy=fancy)


    first_pattern = 'ssdp'
    second_pattern = 'ssdp'

    pattern = f'{first_pattern}->{first_pattern}+{second_pattern}->{second_pattern}'
    window = (100, 0)
    # pattern = 'ntp;bot'
    pattern = 'T4'

    ##### individual training curves #####
    # plot_training_curves(['/srv/bachmann/data/dqn/dqn_20220108-133951/datastore'], 'T4')
    # plot_training_curves(['/srv/bachmann/data/ddpg/ddpg_20220114-070345/datastore'], 'T4')
    # plot_training_curves(['/srv/bachmann/data/ppo/ppo_20220112-071400/datastore'], 'T4')

    ##### eval curves comparison #####
    # plot_training_curves(['/srv/bachmann/data/ddpg/ddpg_20220114-070345/datastore',
    #                      '/srv/bachmann/data/ppo/ppo_20220112-071400/datastore',
    #                      '/srv/bachmann/data/dqn/dqn_20220108-133951/datastore'], 'T4')

    # DDPG BatchNorm
    # plot_training_curves(['/srv/bachmann/data/ddpg/ddpg_20220114-070345/datastore',
    #                      '/srv/bachmann/data/ddpg/ddpg_20220114-073804/datastore'], 'T4', show_all=True,
    #                     label_agent_suffixes=[' (BatchNorm)', ' (no BatchNorm)'])

    # ds_base = '/srv/bachmann/data/dqn/dqn_20220108-133951/datastore'  # DQN-S1 in thesis
    ds_base = '/srv/bachmann/data/ddpg/ddpg_20220114-070345/datastore'  # DDPG-S1 in thesis
    # ds_base = '/srv/bachmann/data/ppo/ppo_20220112-071400/datastore'  # PPO-S1 in thesis
    ds_base = '/srv/bachmann/data/ddpg/ddpg_20220114-073804/datastore'  # DDPG-NoBN in thesis
    # ds_base = '/srv/bachmann/data/dqn/dqn_20220115-141343/datastore'  # DQN L
    ds_base = '/srv/bachmann/data/dqn/dqn_20220115-141541/datastore'  # DQN no WOC
    ds_base = '/home/bachmann/test-pycharm/data/eval-baseline_20220117-073045/datastore'  # fix params no WOC
    ds_base = '/home/bachmann/test-pycharm/data/eval-baseline_20220117-073141/datastore'  # fix params WOC
    plot_hl(ds_base, 'T4', (100, 0), fancy=True)

    # plot_hl(ds_base, pattern, window, fancy=True)
