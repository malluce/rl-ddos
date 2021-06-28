import matplotlib.pyplot as plt
import pandas


def plot_results(episode_file, label):
    episodes = pandas.read_csv(episode_file)
    rewards = episodes.iloc[:, -1]
    precisions = episodes.iloc[:, 3]
    recalls = episodes.iloc[:, 4]
    rules = episodes.iloc[:, 2]
    fig: plt.Figure = plt.figure()

    rew_ax = fig.add_subplot(2, 2, 1)
    rew_ax.plot(range(rewards.shape[0]), rewards, label=label)
    rew_ax.set_xlabel('episodes')
    rew_ax.set_ylabel('reward')
    rew_ax.set_ylim(bottom=0)
    rew_ax.legend()

    prec_ax = fig.add_subplot(2, 2, 2)
    prec_ax.plot(range(precisions.shape[0]), precisions, label=label)
    prec_ax.set_xlabel('episodes')
    prec_ax.set_ylabel('precision')
    prec_ax.set_ylim(bottom=0, top=1)
    prec_ax.legend()

    rec_ax = fig.add_subplot(2, 2, 3)
    rec_ax.plot(range(recalls.shape[0]), recalls, label=label)
    rec_ax.set_xlabel('episodes')
    rec_ax.set_ylabel('recall')
    rec_ax.set_ylim(bottom=0, top=1)
    rec_ax.legend()

    rules_ax = fig.add_subplot(2, 2, 4)
    rules_ax.plot(range(rules.shape[0]), rules, label=label)
    rules_ax.set_xlabel('episodes')
    rules_ax.set_ylabel('number of rules')
    rules_ax.set_ylim(bottom=0)
    rules_ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    episode_file = 'D:\\home\\bachmann\\test-pycharm\\data\\sb-dqn_20210625-090803\\datastore\\train\\episodes.csv'
    plot_results(episode_file, label='')
