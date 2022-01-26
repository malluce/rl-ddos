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
from plotting.util import smooth

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


def plot(fig: plt.Figure, data, data_quantiles, cols, ax_id, x_label, data_label, y_min=None, y_max=None, title=None,
         labels=None, mean_for_title=None, nrow=None, ncol=None, ax=None, line_color=None, vspan_spec=None,
         show_vspan_label=False):
    ax: plt.Axes = fig.add_subplot(nrow, ncol, ax_id) if ax is None else ax
    x = range(data.shape[0])
    for idx, col in enumerate(cols):
        y = data.loc[:, col]
        if data_quantiles is not None:
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
        if line_color is None:
            median_color = 'navy' if idx == 0 else 'red'
        else:
            median_color = line_color
        if labels is None:
            ax.plot(x, y, color=median_color, label=data_label, linewidth=1)
        else:
            ax.plot(x, y, color=median_color, label=labels[idx], linewidth=1)
    if y_max is not None:
        ax.set_ylim(bottom=y_min if y_min is not None else 0, top=y_max if type(y_max) == int else y_max + 0.1)
    else:
        ax.set_ylim(bottom=y_min if y_min is not None else 0)
    ax.tick_params(labelsize=LABELSIZE)
    ax.grid()
    add_vspan_to(ax, vspan_spec, show_vspan_label)
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


def plot_time_index(last, metric, ax, label, y_top=None, make_legend=True, vspan_spec=None, show_vspan_label=False):
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
        ax.set_ylabel('Mean #rules in WOC')
    else:
        new_df_grouped = new_df.loc[:, ['Index', metric]].groupby(['Index'])
        new_df_mean = new_df_grouped.median()

        new_df_quantiles = get_quantiles(new_df_grouped)

        lines = ax.plot(new_df_mean, linewidth=1.5, label='Median')
        plot_color = to_rgb(lines[0].get_color())
        x = new_df['Index'].unique()
        for quantile in new_df_quantiles:
            q, quantile_value = quantile
            alpha = 0.7 - 1.5 * abs(0.5 - q)

            color = (plot_color[0], plot_color[1], plot_color[2], alpha)  # RGBA
            y = new_df_mean.loc[:, metric]
            y_err = quantile_value.loc[:, metric]
            if q in [0.1, 0.2, 0.4]:
                ax.fill_between(x, y, y_err, edgecolor=color,
                                facecolor=color, label=f'{q}/{0.5 + abs(0.5 - q)} Quantile')
            else:
                ax.fill_between(x, y, y_err, edgecolor=color, facecolor=color)
        ax.set_ylabel(label)
        if make_legend:
            ax.legend(ncol=2)

    ax.set_ylim(bottom=0)
    if y_top is not None:
        ax.set_ylim(bottom=0, top=y_top)
    ax.set_xlim(left=0, right=600)
    ax.vlines(range(0, int(new_df['Index'].max()), 10), ymin=0, ymax=ax.get_ylim()[1], linestyles='--',
              linewidth=1, alpha=0.6)
    add_vspan_to(ax, vspan_spec, show_vspan_label, use_time_index=True)
    return True


def add_vspan_to(ax, vspan_spec, show_vspan_label, use_time_index=False):
    if vspan_spec is not None:
        for lo, hi, color, phase_id in vspan_spec:
            lo = lo if not use_time_index else lo * 10
            hi = hi if not use_time_index else hi * 10 - 1
            rect = ax.axvspan(lo, hi, facecolor=color, alpha=0.15)
            if show_vspan_label:
                mid_x = rect.xy[0, 0]
                ax.text(mid_x, 0.5 * ax.get_ylim()[1], phase_id)


def plot_ep_behav_for_pattern(env_csv, environment_file_path, pat, window, fancy):
    vspan_spec = get_vspan_spec_for(pattern=pat)
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

    matplotlib.rcParams.update({'font.size': 13})
    plt.rc('axes', labelsize=18)
    plt.rc('figure', titlesize=18)
    # plot per time idx
    if all(map(lambda x: x in last.columns, ['PrecisionIdx', 'FPRIdx', 'RecallIdx', 'BlackSizeIdx', 'CacheIdx'])):
        fig: plt.Figure = plt.figure(figsize=(16, 9))
        plotted_at_least_one = False
        for idx, val in enumerate(
                zip(['CacheIdx', 'CacheIdx'],
                    ['#Rules in WOC (total)', 'rule-gran'])):
            metric, label = val
            ax: plt.Axes = fig.add_subplot(2, 1, idx + 1)
            ax.set_xlabel('Time index')
            plotted = plot_time_index(last, metric, ax, label=label, make_legend=True, vspan_spec=vspan_spec,
                                      show_vspan_label=True if idx == 0 else False)
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
                zip(['PrecisionIdx', 'RecallIdx', 'BlackSizeIdx', 'FPRIdx'],
                    ['Precision', 'Recall', '#Rules', 'FPR'])):
            metric, label = val
            ax: plt.Axes = fig.add_subplot(4, 1, idx + 1)
            ax.set_xlabel('Time index')
            plotted = plot_time_index(last, metric, ax, label=label, y_top=1.1 if metric != 'BlackSizeIdx' else None,
                                      make_legend=idx == 3, vspan_spec=vspan_spec,
                                      show_vspan_label=True if idx == 0 else False)
            plotted_at_least_one = plotted_at_least_one or plotted
        if plotted_at_least_one:
            fig.suptitle(title)
            plt.tight_layout()
            plt.show()
        else:
            plt.clf()

    last = last.groupby(['Step'])  # return all columns grouped for each step
    last_median = last.median()
    last_quantiles = get_quantiles(last)

    print(
        last.median().loc[:, ['Precision', 'Recall', 'BlackSize', 'FPR', 'Reward', 'Phi', 'Thresh', 'MinPrefix']])
    print('=====direct attack:\n' + str(
        last.mean().loc[0:21,
        ['Precision', 'Recall', 'BlackSize', 'FPR', 'Reward', 'Phi', 'Thresh', 'MinPrefix']].mean()))
    print('=====no attack:\n' + str(
        last.mean().loc[28:32,
        ['Precision', 'Recall', 'BlackSize', 'FPR', 'Reward', 'Phi', 'Thresh', 'MinPrefix']].mean()))
    print('=====reflector attack:\n' + str(
        last.mean().loc[37:,
        ['Precision', 'Recall', 'BlackSize', 'FPR', 'Reward', 'Phi', 'Thresh', 'MinPrefix']].mean()))
    print('=====total:\n' + str(
        last.mean().loc[:, ['Precision', 'Recall', 'BlackSize', 'FPR', 'Reward', 'Phi', 'Thresh', 'MinPrefix']].mean()))

    # plot per time step
    create_plots(last_median, last_quantiles,
                 title=title,
                 x_label='Adaptation step',
                 data_label='Median', means_for_title=last.mean(), fancy=fancy, vspan_spec=vspan_spec)


def get_vspan_spec_for(pattern):
    if pattern == 'T5':
        return [(0, 8, 'yellow', 'LO1'), (13, 19, 'grey', 'MO'), (24, 30, 'purple', 'SO'), (35, 47, 'blue', 'TO'),
                (55, 57, 'orange', 'LO2')]
    elif pattern == 'T6':
        raise NotImplementedError()
    elif pattern == 'T4':
        return [(0, 21, 'yellow', 'Direct'), (28, 32, 'grey', 'No'), (37, 57, 'purple', 'Reflection')]
    else:
        return None


def create_plots(data, quantiles, title, x_label, data_label, means_for_title=None, fancy=False, vspan_spec=None):
    if vspan_spec is None:
        vspan_spec = []

    def finalize():
        if title is not None:
            fig.suptitle(title)

        fig.tight_layout()
        fig.gca().legend(handles, labels, fontsize=LEGENDSIZE, ncol=1)
        plt.show()

    l_used = data['MinPrefix'].max() > 16
    pthresh_used = 'Thresh' in data and data['Thresh'].max() > -1

    if fancy:
        nrow = 1
        ncol = 3 if l_used and pthresh_used else 2
    else:
        nrow = 2
        ncol = 4

    fig: plt.Figure = plt.figure(figsize=(15, 5) if fancy else (16, 9))
    title_means = defaultdict(lambda: None)
    if means_for_title is not None:
        for col in ['Precision', 'Recall', 'BlackSize', 'FPR', 'Reward', 'Phi', 'Thresh', 'MinPrefix']:
            if col in data:
                title_means[col] = means_for_title[col].mean() if not fancy else None
    handles, labels = plot(fig, data, quantiles, ['Phi'], 1, y_max=0.25, x_label=x_label, data_label=data_label,
                           mean_for_title=title_means['Phi'], title='Frequency threshold $\phi$', nrow=nrow, ncol=ncol,
                           vspan_spec=vspan_spec, show_vspan_label=True)
    if l_used:
        plot(fig, data, quantiles, ['MinPrefix'], 2, y_max=32, x_label=x_label, data_label=data_label,
             mean_for_title=title_means['MinPrefix'], title='Minimum prefix length $L$', nrow=nrow,
             ncol=ncol, vspan_spec=vspan_spec)
    if pthresh_used:
        plot(fig, data, quantiles, ['Thresh'], 3 if l_used else 2, y_min=0.84, y_max=0.95, x_label=x_label,
             data_label=data_label,
             mean_for_title=title_means['Thresh'], title='Performance threshold pthresh', nrow=nrow, ncol=ncol,
             vspan_spec=vspan_spec)
        offset = 1 if l_used else 0
    else:
        offset = 0 if l_used else -1

    if fancy:
        ncol = 4
        offset = -2
        finalize()
        fig: plt.Figure = plt.figure(figsize=(15, 5) if fancy else (16, 9))

    plot(fig, data, quantiles, ['Precision'], 3 + offset, y_max=1.0, x_label=x_label, data_label=data_label,
         mean_for_title=title_means['Precision'], nrow=nrow, ncol=ncol, vspan_spec=vspan_spec, show_vspan_label=True)
    plot(fig, data, quantiles, ['Recall'], 4 + offset, y_max=1.0, x_label=x_label, data_label=data_label,
         mean_for_title=title_means['Recall'], nrow=nrow, ncol=ncol, vspan_spec=vspan_spec)
    plot(fig, data, quantiles, ['BlackSize'], 5 + offset, x_label=x_label, data_label=data_label,
         mean_for_title=title_means['BlackSize'], title='Number of filter rules', nrow=nrow, ncol=ncol,
         vspan_spec=vspan_spec)
    handles, labels = plot(fig, data, quantiles, ['FPR'], 6 + offset, y_max=1.0, x_label=x_label, data_label=data_label,
                           mean_for_title=title_means['FPR'], title='False positive rate', nrow=nrow, ncol=ncol,
                           vspan_spec=vspan_spec)
    if not fancy:
        reward_max = data['Reward'].max() if data['Reward'].max() > 1.0 else 1.0
        plot(fig, data, quantiles, ['Reward'], 7 + offset, x_label=x_label, data_label=data_label,
             y_max=reward_max, mean_for_title=title_means['Reward'], nrow=nrow, ncol=ncol, vspan_spec=vspan_spec)
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
    plt.gca().grid(visible=True)
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


def plot_agent_comparison(ds_bases, data_labels, pattern, use_train_path=True, window=None):
    paths = []
    for ds in ds_bases:
        train_path, eval_path = get_paths(ds)
        paths.append(train_path if use_train_path else eval_path)

    def finalize():
        fig.tight_layout()
        fig.gca().legend(handles, labels, fontsize=15, ncol=1)
        plt.show()

    _vspan_spec = get_vspan_spec_for(pattern)

    x_label = 'Adaptation step'
    colors = ['navy', 'red', 'green', 'orange', 'black', 'purple', 'yellow']
    title_means = defaultdict(lambda: None)
    nrow = 1
    ncol = 4
    fig: plt.Figure = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(nrow, ncol, 1)
    ax2 = fig.add_subplot(nrow, ncol, 2)
    ax3 = fig.add_subplot(nrow, ncol, 3)
    ax4 = fig.add_subplot(nrow, ncol, 4)
    for i, path in enumerate(paths):
        vspan_spec = _vspan_spec if i == 0 else None
        print(f'------- {data_labels[i]} -------')
        env_csv = read_env_csv(path)
        if window is not None:
            eps = env_csv.loc[:, 'Episode']
            unique_eps = sorted(list(set(eps)))  # get all unique episode numbers (some are not included in csv files)
            eps_to_show = unique_eps[len(unique_eps) - window[0]:len(unique_eps) - window[1]]
            print(f'showing eps {eps_to_show}')
            env_csv = get_rows_with_episode_in(env_csv, eps_to_show)

        last = env_csv.groupby(['Step'])  # return all columns grouped for each step
        data = last.mean()
        # print(
        #    last.mean().loc[:,
        #    ['Precision', 'Recall', 'BlackSize', 'FPR', 'Reward', 'Phi', 'Thresh', 'MinPrefix']])
        print('=====direct attack:\n' + str(
            last.mean().loc[0:21,
            ['Precision', 'Recall', 'BlackSize', 'FPR']].mean()))
        print('=====no attack:\n' + str(
            last.mean().loc[28:32,
            ['Precision', 'Recall', 'BlackSize', 'FPR']].mean()))
        print('=====reflector attack:\n' + str(
            last.mean().loc[37:,
            ['Precision', 'Recall', 'BlackSize', 'FPR']].mean()))
        print('=====total:\n' + str(
            env_csv.loc[:, ['Precision', 'Recall', 'BlackSize', 'FPR']].mean()))
        data_label = f'Mean ({data_labels[i]})'
        plot(fig, data, None, ['Precision'], 1, y_max=1.0, x_label=x_label, data_label=data_label,
             mean_for_title=title_means['Precision'], ax=ax1, line_color=colors[i], vspan_spec=vspan_spec,
             show_vspan_label=True)
        plot(fig, data, None, ['Recall'], 2, y_max=1.0, x_label=x_label, data_label=data_label,
             mean_for_title=title_means['Recall'], ax=ax2, line_color=colors[i],
             vspan_spec=vspan_spec)
        plot(fig, data, None, ['BlackSize'], 3, x_label=x_label, data_label=data_label,
             mean_for_title=title_means['BlackSize'], title='Number of filter rules', ax=ax3, line_color=colors[i],
             vspan_spec=vspan_spec)
        handles, labels = plot(fig, data, None, ['FPR'], 4, y_max=1.0, x_label=x_label,
                               data_label=data_label,
                               mean_for_title=title_means['FPR'], title='False positive rate', ax=ax4,
                               line_color=colors[i],
                               vspan_spec=vspan_spec)
    ax1.grid(visible=True)
    ax2.grid(visible=True)
    ax3.grid(visible=True)
    ax4.grid(visible=True)
    finalize()

    l_used = False
    pthresh_used = False
    for i, path in enumerate(paths):
        env_csv = read_env_csv(path)
        l_used = l_used or env_csv.loc[:, 'MinPrefix'].max() > 16
        pthresh_used = pthresh_used or env_csv.loc[:, 'Thresh'].min() > -1

    nrow = 1
    ncol = 3 if l_used and pthresh_used else 2
    fig: plt.Figure = plt.figure(figsize=(15, 5))
    axes = []
    handles, labels = [], []
    for i in range(ncol):
        axes.append(fig.add_subplot(nrow, ncol, i + 1))

    plotted_l, plotted_pthresh = False, False
    for i, path in enumerate(paths):
        vspan_spec = _vspan_spec if i == 0 else None
        print(f'------- {data_labels[i]} -------')
        env_csv = read_env_csv(path)
        if window is not None:
            eps = env_csv.loc[:, 'Episode']
            unique_eps = sorted(list(set(eps)))  # get all unique episode numbers (some are not included in csv files)
            eps_to_show = unique_eps[len(unique_eps) - window[0]:len(unique_eps) - window[1]]
            env_csv = get_rows_with_episode_in(env_csv, eps_to_show)

        last = env_csv.groupby(['Step'])  # return all columns grouped for each step
        data = last.mean()
        print('=====direct attack:\n' + str(last.mean().loc[0:21, ['Phi', 'MinPrefix', 'Thresh']].mean()))
        print('=====no attack:\n' + str(last.mean().loc[28:32, ['Phi', 'MinPrefix', 'Thresh']].mean()))
        print('=====reflector attack:\n' + str(last.mean().loc[37:, ['Phi', 'MinPrefix', 'Thresh']].mean()))
        print('=====total:\n' + str(env_csv.loc[:, ['Phi', 'MinPrefix', 'Thresh']].mean()))
        data_label = f'Mean ({data_labels[i]})'
        handles, labels = plot(fig, data, None, ['Phi'], 1, y_max=0.25, x_label=x_label, data_label=data_label,
                               mean_for_title=title_means['Phi'], ax=axes[0], line_color=colors[i],
                               title='Frequency threshold $\phi$',
                               vspan_spec=vspan_spec, show_vspan_label=True)

        if env_csv.loc[:, 'MinPrefix'].max() > 16:
            plot(fig, data, None, ['MinPrefix'], 2, y_max=32, x_label=x_label,
                 data_label=data_label,
                 mean_for_title=title_means['MinPrefix'], ax=axes[1], line_color=colors[i],
                 title='Minimum prefix length $L$',
                 vspan_spec=_vspan_spec if i == 0 or not plotted_l else None)
            plotted_l = True
        else:
            assert env_csv.loc[:, 'Thresh'].min() > -1
            plot(fig, data, None, ['Thresh'], 2, y_min=0.84, y_max=0.95, x_label=x_label,
                 data_label=data_label,
                 mean_for_title=title_means['Thresh'], ax=axes[-1], line_color=colors[i],
                 title='Performance threshold pthresh',
                 vspan_spec=vspan_spec if i == 0 or not plotted_pthresh else None)
            plotted_pthresh = True
    for ax in axes:
        ax.grid(visible=True)
    finalize()


if __name__ == '__main__':
    def plot_hl(ds_base, pattern, window, fancy=False, use_train_path=False):
        train_path, eval_path = get_paths(ds_base)
        # plot_training(environment_file_path=train_path, pattern=pattern)
        # plot_training(environment_file_path=eval_path, pattern=pattern)
        path = train_path if use_train_path else eval_path
        plot_episode_behavior(environment_file_path=path, pattern=pattern, window=window, fancy=fancy)


    old_dqn_s1 = '/srv/bachmann/data/dqn/dqn_20220108-133951/datastore'  # (old) DQN-S1 in thesis
    old_ddpg_s1 = '/srv/bachmann/data/ddpg/ddpg_20220114-070345/datastore'  # (old) DDPG-S1 in thesis
    old_ppo_s1 = '/srv/bachmann/data/ppo/ppo_20220112-071400/datastore'  # (old) PPO-S1 in thesis

    dqn_s1 = '/srv/bachmann/data/dqn/dqn_20220119-070755/datastore'  # DQN-S1 in thesis
    ddpg_s1 = '/srv/bachmann/data/ddpg/ddpg_20220120-202020/datastore'  # DDPG-S1 in thesis
    ppo_s1 = '/srv/bachmann/data/ppo/ppo_20220120-202121/datastore'  # PPO-S1 in thesis

    dddpg_rej_s2 = '/srv/bachmann/data/ddpg/ddpg_20220124-132957/datastore'  # DDPG S2 test (not in thesis)
    dqn_rej_s2 = '/srv/bachmann/data/dqn/dqn_20220124-125147/datastore'  # DQN-pthresh S2 in thesis
    dqn_l_s2 = '/srv/bachmann/data/dqn/dqn_20220124-204351/datastore'  # DQN-L S2 in thesis

    # plot_hl(dqn_rej_s2, 'T5', (10, 0), fancy=True)
    plot_hl(dqn_l_s2, 'T5', (10, 0), fancy=True)
    # plot_agent_comparison([dqn_rej_s2, dqn_l_s2], ['DQN-pthresh', 'DQN-L'], 'T5',
    #                      use_train_path=False,
    #                      window=(10, 0))

    dqn_rej_s3 = '/srv/bachmann/data/dqn/dqn_20220125-105514/datastore'  # DQN-pthresh S3
    dqn_l_s3 = '/srv/bachmann/data/dqn/dqn_20220125-105606/datastore'  # DQN-L S3

    # plot_hl(dqn_rej_s3, 'T6', (10, 0), fancy=True)
    # plot_hl(dqn_l_s3, 'T6', (10, 0), fancy=True)
    # plot_agent_comparison([dqn_rej_s3, dqn_l_s3], ['DQN-pthresh', 'DQN-L'],'T6', use_train_path=False,
    #                      window=(10, 0))

    ##### params and metrics #####
    # plot_hl(dqn_s1, 'T4', (10, 0), fancy=True)
    # plot_hl(ddpg_s1, 'T4', (10, 0), fancy=True)
    # plot_hl(ppo_s1, 'T4', (10, 0), fancy=True)

    ##### individual training curves #####
    # plot_training_curves([dqn_s1], 'T4')
    # plot_training_curves([ddpg_s1], 'T4')
    # plot_training_curves([ppo_s1], 'T4')

    ##### eval curves comparison #####
    # plot_training_curves([dqn_s1, ddpg_s1, ppo_s1], 'T4')
    ##### direct metric and param comp #####
    # plot_agent_comparison([dqn_s1, ddpg_s1, ppo_s1], ['DQN', 'DDPG', 'PPO'], 'T4', use_train_path=False, window=(10, 0))

    ##### 6.5 DDPG BatchNorm #####
    ddpg_s1_no_bn = '/srv/bachmann/data/ddpg/ddpg_20220121-203113/datastore'
    # plot_training_curves([ddpg_s1, ddpg_s1_no_bn], 'T4', show_all=True,
    #                     label_agent_suffixes=[' (BatchNorm)', ' (no BatchNorm)'])
    # plot_hl(ddpg_s1_no_bn, 'T4', (10, 0), fancy=True)

    ##### 6.6.1 WOC vs no WOC #####
    fix_params_no_woc = '/home/bachmann/test-pycharm/data/eval-baseline_20220122-160758/datastore'  # fix params no WOC
    fix_params_woc = '/home/bachmann/test-pycharm/data/eval-baseline_20220122-160733/datastore'  # fix params WOC
    # plot_hl(fix_params_woc, 'T4', (100, 0), fancy=True, use_train_path=True)
    # plot_agent_comparison([fix_params_no_woc, fix_params_woc], ['No WOC', 'WOC'], 'T4')
