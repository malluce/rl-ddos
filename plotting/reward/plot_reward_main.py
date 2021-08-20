from gyms.hhh.reward import DefaultRewardCalc, MultiplicativeReward, RewardCalc
from plotting.reward.plot_reward_components import plot_contrib_factors
from plotting.reward.plot_rewards import plot_reward


def plot_reward_function(reward_func: RewardCalc):
    plot_contrib_factors(reward_func)
    plot_reward(reward_func)


if __name__ == '__main__':
    reward = MultiplicativeReward()
    plot_reward_function(reward)
