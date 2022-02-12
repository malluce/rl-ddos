import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from gyms.hhh.action import DqnRejectionActionSpace, agent_action_to_resolved_linear, agent_action_to_resolved_phi


# This script is used to visualize the action spaces and resolving (Figures 4.12 and 4.13)

def agent_action_to_resolved_l(agent_action, lower_bound, upper_bound):  # re-defined to handle numpy arrays as input
    bin_size = 2 / 17  # range[-1,1], 17 L values (16,17,...,32)
    return np.minimum((agent_action + 1) / bin_size + lower_bound, upper_bound).astype('int')


def finalize_resolve_plot():
    plt.legend()
    plt.ylabel('Resolved mitigation parameter')
    plt.xlabel('Action chosen by agent')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    matplotlib.rcParams.update({'font.size': 15})

    # DDPG L (Fig 4.13b)
    agent_action = np.linspace(-1.0, 1.0, 1000)
    resolved = agent_action_to_resolved_l(agent_action=agent_action, lower_bound=16, upper_bound=32)
    plt.plot(agent_action, resolved, label='$L$')
    finalize_resolve_plot()

    # PPO+DDPG pthresh and phi (Fig 4.13a)
    resolved = agent_action_to_resolved_linear(agent_action=agent_action, lower_bound=0.85, upper_bound=1.0)
    plt.plot(agent_action, resolved, label='pthresh')
    resolved = agent_action_to_resolved_phi(agent_action=agent_action, lower_bound=0.001, upper_bound=0.3)
    plt.plot(agent_action, resolved, label='$\phi$')
    plt.yticks([0, 0.25, 0.5, 0.75, 1.0])
    finalize_resolve_plot()

    # DQN continuous action space (Fig 4.12a)
    plt.fill_between([0, 1], [0], [1])
    plt.xlim(left=0.0, right=1.0)
    plt.ylim(bottom=0.0, top=1.0)
    plt.xlabel('$\phi$')
    plt.ylabel('pthresh')
    plt.tight_layout()
    plt.show()

    # DQN pruned action space (Fig 4.12b)
    plt.fill_between([0.001, 0.3], [0.85], [1])
    plt.xlim(left=0.0, right=1.0)
    plt.ylim(bottom=0.0, top=1.0)
    plt.yticks([0, 0.25, 0.5, 0.75, 1.0])
    plt.xticks([0, 0.25, 0.5, 0.75, 1.0])
    plt.xlabel('$\phi$')
    plt.ylabel('pthresh')
    plt.tight_layout()
    plt.show()

    # DQN pruned action space with discretized actions (Fig 4.12c)
    a = DqnRejectionActionSpace()
    x = list(map(lambda act: act[0], a.actions))
    y = list(map(lambda act: act[1], a.actions))
    plt.scatter(x, y, linewidths=1, marker='x')
    plt.xlabel('$\phi$')
    plt.yticks([0.85, 0.9, 0.95, 1.0])
    plt.ylabel('pthresh')
    plt.tight_layout()
    plt.show()
