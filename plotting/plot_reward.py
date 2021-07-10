import os
import matplotlib.pyplot as plt
import pandas


def plot_episode_behavior(environment_file_path, last_x_episodes: int = None, show_in_episode_behaviour: bool = False):
    """
    Plots the metrics of an environment.csv file.
    :param environment_file_path: the environment.csv file path
    :param last_x_episodes: the number of episodes to plot (taken from the end), only relevant if show_in_episode_behavior is True
    :param show_in_episode_behaviour: whether to show the steps inside the episodes on the x axis
    """

    if not ((last_x_episodes is not None and show_in_episode_behaviour) or (
            last_x_episodes is None and not show_in_episode_behaviour)):
        print('Not currently supported, aborting')
        return 1

    env_csv = pandas.read_csv(environment_file_path)
    cols = list(map(lambda x: str(x).strip(), list(env_csv.columns)))  # strip spaces from col names
    env_csv = pandas.read_csv(environment_file_path, header=0, names=cols)  # read csv again with stripped col names

    if show_in_episode_behaviour:  # show steps on x-axis, for last_x_episodes only
        def get_rows_with_episode_in(episodes):
            filtered_rows = env_csv.loc[env_csv['Episode'].isin(episodes), :]  # get rows where episode in episodes
            return filtered_rows.groupby(['Step']).mean(), filtered_rows.groupby(
                ['Step']).std()  # return mean and std of all columns for each step

        eps = env_csv.loc[:, 'Episode']
        unique_eps = sorted(list(set(eps)))  # get all unique episode numbers (some are not included in csv files)
        eps_to_show = unique_eps[len(unique_eps) - last_x_episodes:]
        last, last_std = get_rows_with_episode_in(eps_to_show)
    else:  # show all episodes on x-axis
        grouped_by_episode = env_csv.groupby(['Episode'])
        last, last_std = grouped_by_episode.mean(), grouped_by_episode.std()

    def plot(fig: plt.Figure, data_frame, data_frame_std, cols, x, is_one_bounded=False):
        ax: plt.Axes = fig.add_subplot(2, 4, x)
        x = range(data_frame.shape[0])
        for col in cols:
            y = data_frame.loc[:, col]
            y_err = data_frame_std.loc[:, col]
            ax.plot(x, y, label=col)
            ax.fill_between(x, y - y_err, y + y_err, alpha=0.2)
            if show_in_episode_behaviour:
                ax.set_xlabel('step')
            else:
                ax.set_xlabel('episode')
        if is_one_bounded:
            ax.set_ylim(bottom=0, top=1.1)
        else:
            ax.set_ylim(bottom=0)
        ax.legend()

    fig: plt.Figure = plt.figure(figsize=(16, 8))
    plot(fig, last, last_std, ['Phi'], 1, is_one_bounded=True)
    plot(fig, last, last_std, ['MinPrefix'], 2)
    plot(fig, last, last_std, ['Precision', 'EstPrecision'], 3, is_one_bounded=True)
    plot(fig, last, last_std, ['Recall', 'EstRecall'], 4, is_one_bounded=True)
    plot(fig, last, last_std, ['BlackSize'], 5)
    plot(fig, last, last_std, ['FPR', 'EstFPR'], 6, is_one_bounded=True)
    plot(fig, last, last_std, ['Reward'], 7)
    if last_x_episodes is not None:
        fig.suptitle(f'last {last_x_episodes} eval episodes')
    else:
        fig.suptitle(f'all episodes')
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    ds_base = '/home/bachmann/test-pycharm/data/td3_20210710-105917/datastore'
    train_path = os.path.join(ds_base, 'train', 'environment.csv')
    eval_path = os.path.join(ds_base, 'eval', 'environment.csv')
    plot_episode_behavior(environment_file_path=train_path, show_in_episode_behaviour=False)
    plot_episode_behavior(environment_file_path=eval_path, last_x_episodes=5, show_in_episode_behaviour=True)
