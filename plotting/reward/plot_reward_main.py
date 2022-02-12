from gyms.hhh.reward import MultiplicativeRewardThesis
from plotting.reward.plot_reward_components import plot_contrib_factors
from plotting.reward.plot_rewards import plot_reward_3d

# Simple script that plots the contributing weighted factors and 3D plot of a reward function. (Figure 4.17 and 4.18)

if __name__ == '__main__':
    reward = MultiplicativeRewardThesis(precision_weight=0, bl_weight=0.25, recall_weight=2, fpr_weight=2)
    plot_reward_3d(reward)
    plot_contrib_factors(reward)
