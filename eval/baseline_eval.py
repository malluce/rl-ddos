import time

import gym
from eval_agents import FixedEvalAgent, RandomEvalAgent
from gyms.hhh.actionset import ContinuousRejectionActionSet, DiscreteRejectionActionSet, HugeDiscreteActionSet, \
    LargeDiscreteActionSet, \
    TupleActionSet
from gyms.hhh.env import register_hhh_gym
from gyms.hhh.flowgen.traffic_traces import BotSSDPTrace, BotTrace, MixedSSDPBot, NTPTrace, SSDPTrace, T3, T4, THauke, \
    THauke5, TRandomPatternSwitch
from gyms.hhh.images import ImageGenerator
from gyms.hhh.loop import TimedWorstOffenderCache, TrackingWorstOffenderCache
from gyms.hhh.reward import AdditiveRewardCalc, MultiplicativeReward, MultiplicativeRewardSpecificity
from gyms.hhh.obs import BaseObservations, DistVol, DistVolStd, FalsePositiveRate, MinMaxBlockedAddress
from lib.datastore import Datastore
from agents.util import get_dirs
from plotting.plot_results import plot_episode_behavior
import gin
from absl import logging

env_name = register_hhh_gym()
# gin.bind_parameter('DistributionTrace.traffic_trace_construct', THauke)
# gin.bind_parameter('THauke.benign_flows', 500)
# gin.bind_parameter('THauke.attack_flows', 1000)
# gin.bind_parameter('THauke.maxtime', 599)
# gin.bind_parameter('THauke.maxaddr', 0xffff)


gin.bind_parameter('DistributionTrace.traffic_trace_construct', SSDPTrace)
gin.bind_parameter('TRandomPatternSwitch.benign_normal', True)
gin.bind_parameter('TRandomPatternSwitch.random_toggle_time', True)
gin.bind_parameter('TRandomPatternSwitch.smooth_transition', True)
gin.bind_parameter('RulePerformanceTable.use_cache', True)
gin.bind_parameter('RulePerformanceTable.cache_class', TrackingWorstOffenderCache)
gin.bind_parameter('TrackingWorstOffenderCache.capacity', 'inf')
gin.bind_parameter('RulePerformanceTable.metric', 'fpr')
gin.bind_parameter('TrackingWorstOffenderCache.metric', 'fpr')
gin.bind_parameter('Loop.sampling_rate', 0.25)


# gin.bind_parameter('DistributionTrace.traffic_trace_construct', MixedSSDPBot)


# gin.bind_parameter('DistributionTrace.traffic_trace_construct', TRandomPatternSwitch)
# gin.bind_parameter('TRandomPatternSwitch.random_toggle_time', True)


def eval(eval_agent, action_set):
    dirs = get_dirs('/home/bachmann/test-pycharm/data', Datastore.get_timestamp(), 'eval-baseline')
    ds_train, env = make_env(dirs, action_set)
    # eval_agent = RandomEvalAgent(env.loop.actionset.actionspace, env.loop.actionset)
    start = time.time()
    run_eval_episodes(env, eval_agent)
    print(f'exec time={time.time() - start}')
    # plot_episode_behavior(ds_train.environment_file.name, pattern='ssdp->ssdp+ssdp->ssdp', window=(5, 0))
    # plot_episode_behavior(ds_train.environment_file.name, pattern='bot->bot+bot->bot', window=(5, 0))
    # plot_episode_behavior(ds_train.environment_file.name, pattern='ntp->ntp+ntp->ntp', window=(5, 0))
    plot_episode_behavior(ds_train.environment_file.name, pattern=None, window=(2, 0))
    # plot_episode_behavior(ds_train.environment_file.name, pattern=None, window=(1, 0))
    # plot_episode_behavior(ds_train.environment_file.name, pattern='ssdp', window=(1, 0))
    # plot_episode_behavior(ds_train.environment_file.name, pattern='ntp->ssdp', window=(1, 0))
    # plot_episode_behavior(ds_train.environment_file.name, pattern='bot->ssdp', window=(1, 0))


def run_eval_episodes(env, eval_agent, num_episodes=3):
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        step_cnt = 0
        while not done:
            print(obs)
            action = eval_agent.action(obs)
            obs, rew, done, info = env.step(action)
            step_cnt += 1


def make_env(dirs, actionset):
    gym_kwargs = {
        'state_obs_selection': [BaseObservations()],
        'use_prev_action_as_obs': True,
        'actionset': actionset,
        'gamma': 0,
        'reward_calc': MultiplicativeRewardSpecificity(precision_weight=0, bl_weight=0.344, recall_weight=2.03,
                                                       fpr_weight=3.4),
        'image_gen': None,
        'is_eval': True
    }
    ds_train = Datastore(dirs['root'], 'train', collect_raw=True)
    return ds_train, gym.make(env_name, **{'data_store': ds_train, **gym_kwargs})


logging.set_verbosity('debug')

# Bot
# phi = 0.091
phi = 0.007
# phi = 0.068
thresh = 0.96
L = None
actionset = ContinuousRejectionActionSet()
# actionset = HugeDiscreteActionSet()
# actionset = TupleActionSet()
eval_agent = FixedEvalAgent(phi, L, thresh, actionset)
eval(eval_agent, actionset)
