from collections import namedtuple

import numpy as np

from gyms.hhh.reward import AdditiveRewardCalc, AdditiveRewardExponentialWeights, DefaultRewardCalc, \
    MultiplicativeReward, \
    MultiplicativeRewardNew, MultiplicativeRewardSpecificity, RewardCalc
from plotting.reward.plot_reward_components import plot_contrib_factors
from plotting.reward.plot_rewards import plot_reward, plot_reward2, plot_reward_no_prec, plot_reward_no_prec_comparison, \
    plot_reward_no_prec_comparison_rec_bl


def plot_reward_function(reward_func: RewardCalc):
    # plot_reward_no_prec_comparison(reward_func)
    # plot_reward_no_prec_comparison_rec_bl(reward_func)
    # plot_contrib_factors(reward_func)
    # plot_reward_no_prec(reward_func, bl_max=100)
    plot_reward2(reward_func)


if __name__ == '__main__':
    # reward = MultiplicativeRewardSpecificity(precision_weight=0, bl_weight=0.3, recall_weight=1.35, fpr_weight=1
    #                                        )

    class DummyState:
        def __init__(self, recall, precision, blacklist_size, fpr):
            self.recall = recall
            self.precision = precision
            self.blacklist_size = blacklist_size
            self.fpr = fpr


    def comp_reward(recall, precision, bl_size, fpr):
        state = DummyState(recall, precision, bl_size, fpr)
        return reward.calc_reward(state)


    metrics = namedtuple('Metrics', ['recall', 'precision', 'fpr', 'bl_size'])

    bot_should = metrics(0.64, 0.973, 0.065, 6.627)
    ssdp_should = metrics(0.994, 0.976, 0.104, 100)

    ssdp_bad_cases = [
        # too few rules
        metrics(0.461, 0.888, 0.444, 0.953),
        metrics(0.321, 0.942, 0.028, 1.222),
        metrics(recall=0.872, precision=0.993, fpr=0.11, bl_size=83.892),
        metrics(recall=0.494, precision=0.979, fpr=0.187, bl_size=4.617),
        metrics(recall=0.005, precision=0.162, fpr=0.005, bl_size=0.051),
        metrics(recall=1, precision=0.947, fpr=1, bl_size=1),
        metrics(recall=0, precision=1, fpr=0, bl_size=0),
        metrics(recall=0.659, precision=0.988, fpr=0.142, bl_size=17.439),
        metrics(recall=0.624, precision=0.99, fpr=0.114, bl_size=17.333),
        metrics(recall=0.643, precision=0.99, fpr=0.124, bl_size=20.090),
        metrics(recall=0.309, precision=0.959, fpr=0.03, bl_size=1.148),
        metrics(recall=0.489, precision=0.995, fpr=0.047, bl_size=5.593),
    ]

    bot_bad_cases = [
        # too many rules
        metrics(recall=0.962, precision=0.980, fpr=0.093, bl_size=53.103),
        metrics(recall=0.988, precision=0.977, fpr=0.103, bl_size=54.047),
        metrics(recall=0.94, precision=0.981, fpr=0.083, bl_size=45.268),
        metrics(recall=0.958, precision=0.982, fpr=0.081, bl_size=47.708),
        metrics(recall=0.899, precision=0.974, fpr=0.114, bl_size=68.128),
        # too few rules
        metrics(recall=0.409, precision=0.973, fpr=0.027, bl_size=1.367),
        metrics(recall=0.384, precision=0.957, fpr=0.07, bl_size=1.369),
        metrics(recall=0.363, precision=0.944, fpr=0.024, bl_size=1.231),
        metrics(recall=0.413, precision=0.964, fpr=0.061, bl_size=1.623),
        metrics(recall=0.278, precision=0.859, fpr=0.038, bl_size=0.845),
        metrics(recall=0.005, precision=0.124, fpr=0.002, bl_size=0.03),
        metrics(recall=0.42, precision=0.965, fpr=0.041, bl_size=2.545),
        metrics(recall=0.592, precision=0.868, fpr=0.436, bl_size=1.959),
        metrics(recall=0, precision=1, fpr=0, bl_size=0),
    ]

    reward = AdditiveRewardCalc(precision_weight=0, bl_weight=0.1, recall_weight=1.9595, fpr_weight=2.0)
    reward = MultiplicativeRewardSpecificity(precision_weight=0, bl_weight=0.22, recall_weight=0.75, fpr_weight=1)
    reward = MultiplicativeRewardSpecificity(precision_weight=0, bl_weight=0.344, recall_weight=2.03, fpr_weight=3.4)
    reward = MultiplicativeRewardSpecificity(precision_weight=0, bl_weight=0.295, recall_weight=1.4, fpr_weight=3)
    reward = MultiplicativeRewardNew(precision_weight=0, bl_weight=0.263, recall_weight=1.784, fpr_weight=1.432)
    bot_should_reward = comp_reward(bot_should.recall, bot_should.precision, bot_should.bl_size,
                                    bot_should.fpr)
    ssdp_should_reward = comp_reward(ssdp_should.recall, ssdp_should.precision, ssdp_should.bl_size,
                                     ssdp_should.fpr)
    print(f'bot should={bot_should_reward}')

    print(f'bot bad cases:')
    for case in bot_bad_cases:
        case_rew = comp_reward(case.recall, case.precision, case.bl_size, case.fpr)
        print(f'{case_rew}, dif={abs(case_rew - bot_should_reward)}')
        if case_rew >= bot_should_reward:
            print(f'fail! ({case.recall, case.precision, case.bl_size, case.fpr})')
            break

    print(f'ssdp should={ssdp_should_reward}')
    print(f'ssdp bad cases:')
    for case in ssdp_bad_cases:
        case_rew = comp_reward(case.recall, case.precision, case.bl_size, case.fpr)
        print(f'{case_rew}, dif={abs(case_rew - ssdp_should_reward)}')
        if case_rew >= ssdp_should_reward:
            print(f'fail! ({case.recall, case.precision, case.bl_size, case.fpr})')
            break

    reward = MultiplicativeRewardNew(precision_weight=0, bl_weight=0.25, recall_weight=2, fpr_weight=2)
    plot_reward_function(reward)

    exit(1)

    for bl in np.random.permutation(np.linspace(0.1, 1, num=1000)):
        for rec in np.random.permutation(np.linspace(5, 0.5, num=1000)):
            for fpr in np.random.permutation(np.linspace(5, 0.5, num=1000)):
                # reward = AdditiveRewardCalc(precision_weight=0, bl_weight=bl, recall_weight=rec, fpr_weight=fpr)
                # reward = AdditiveRewardExponentialWeights(precision_weight=0, bl_weight=bl, recall_weight=rec,
                #                                          fpr_weight=fpr)
                # reward = MultiplicativeRewardSpecificity(precision_weight=0, bl_weight=bl, recall_weight=rec,
                #                                         fpr_weight=fpr)
                reward = MultiplicativeRewardNew(precision_weight=0, bl_weight=bl, recall_weight=rec,
                                                 fpr_weight=fpr)
                bot_should_reward = comp_reward(bot_should.recall, bot_should.precision, bot_should.bl_size,
                                                bot_should.fpr)
                ssdp_should_reward = comp_reward(ssdp_should.recall, ssdp_should.precision, ssdp_should.bl_size,
                                                 ssdp_should.fpr)
                bot_pass = True
                bot_dif = np.inf
                for case in bot_bad_cases:
                    case_rew = comp_reward(case.recall, case.precision, case.bl_size, case.fpr)

                    if case_rew >= bot_should_reward:
                        bot_pass = False
                        break
                    else:
                        bot_dif = min(bot_should_reward - case_rew, bot_dif)
                ssdp_pass = True
                ssdp_dif = np.inf
                for case in ssdp_bad_cases:
                    case_rew = comp_reward(case.recall, case.precision, case.bl_size, case.fpr)
                    if case_rew >= ssdp_should_reward:
                        ssdp_pass = False
                        break
                    else:
                        ssdp_dif = min(ssdp_should_reward - case_rew, ssdp_dif)
                if bot_pass and ssdp_pass:
                    if bot_dif > 0.01 and ssdp_dif > 0.01:
                        print('=' * 100)

                    print(f'passed (bot dif={bot_dif}, ssdp dif={ssdp_dif}): bl={bl}, rec={rec}, fpr={fpr}')

                    if bot_dif > 0.01 and ssdp_dif > 0.01:
                        print('=' * 100)
