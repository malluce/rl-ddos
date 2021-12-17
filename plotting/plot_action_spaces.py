import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from gyms.hhh.actionset import EvenSmallerDiscreteRejectionActionSet, EvenSmallerDiscreteRejectionActionSet2, \
    NotSoHugeDiscreteActionSet, \
    SmallDiscreteRejectionActionSet

matplotlib.rcParams.update({'font.size': 15})


def agent_action_to_resolved_phi(agent_action, lower_bound, upper_bound):
    return np.maximum((1 / (1 + np.exp(-3.25 * (agent_action - 1)))) * 2 * upper_bound, lower_bound)


def agent_action_to_resolved_pthresh(agent_action, lower_bound, upper_bound):
    middle = (upper_bound + lower_bound) / 2
    middle_to_bound = np.abs(middle - upper_bound)
    return middle + middle_to_bound * agent_action


def agent_action_to_resolved_l(agent_action, lower_bound, upper_bound):
    bin_size = 2 / 17  # range[-1,1], 17 L values (16,17,...,32)
    return np.minimum((agent_action + 1) / bin_size + lower_bound, upper_bound).astype('int')


a = NotSoHugeDiscreteActionSet()
print(len(a.actions))

agent_action = np.linspace(-1.0, 1.0, 1000)
resolved = agent_action_to_resolved_l(agent_action=agent_action, lower_bound=16, upper_bound=32)
print(min(resolved), max(resolved))
plt.plot(agent_action, resolved, label='$L$')
resolved = agent_action_to_resolved_pthresh(agent_action=agent_action, lower_bound=0.85, upper_bound=1.0)
print(min(resolved), max(resolved))
# plt.plot(agent_action, resolved, label='pthresh')
resolved = agent_action_to_resolved_pthresh(agent_action=agent_action, lower_bound=0.001, upper_bound=0.3)
print(min(resolved), max(resolved))
# print(resolved)
# plt.plot(agent_action, resolved, label='$\phi$')
plt.xlabel('Action chosen by agent')
plt.ylabel('Resolved mitigation parameter')
# plt.ylim(bottom=0.85, top=1.0)
# plt.yticks([0, 0.25, 0.5, 0.75, 1.0])
plt.legend()
plt.tight_layout()
plt.show()
sys.exit(0)

plt.fill_between([0, 1], [0], [1])
plt.xlim(left=0.0, right=1.0)
plt.ylim(bottom=0.0, top=1.0)
plt.xlabel('$\phi$')
plt.ylabel('pthresh')
plt.tight_layout()
plt.show()

plt.fill_between([0.001, 0.3], [0.85], [1])
plt.xlim(left=0.0, right=1.0)
plt.ylim(bottom=0.0, top=1.0)
plt.yticks([0, 0.25, 0.5, 0.75, 1.0])
plt.xticks([0, 0.25, 0.5, 0.75, 1.0])
plt.xlabel('$\phi$')
plt.ylabel('pthresh')
plt.tight_layout()
plt.show()

a = EvenSmallerDiscreteRejectionActionSet2()
x = list(map(lambda act: act[0], a.actions))
y = list(map(lambda act: act[1], a.actions))
plt.scatter(x, y, linewidths=1, marker='x')
plt.xlabel('$\phi$')
plt.yticks([0.85, 0.9, 0.95, 1.0])
plt.ylabel('pthresh')
# plt.xscale('log')
plt.tight_layout()
plt.show()
