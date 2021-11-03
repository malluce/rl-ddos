import time

import gym
from eval_agents import FixedEvalAgent, RandomEvalAgent
from gyms.hhh.actionset import ContinuousRejectionActionSet, DirectResolveActionSet, \
    DirectResolveRejectionActionSet, \
    LargeDiscreteActionSet
from gyms.hhh.env import register_hhh_gym
from gyms.hhh.flowgen.traffic_traces import MixedSSDPBot, SSDPTrace, T4, THauke, TRandomPatternSwitch
from gyms.hhh.images import ImageGenerator
from gyms.hhh.reward import MultiplicativeReward
from gyms.hhh.obs import BaseObservations, DistVol, DistVolStd, FalsePositiveRate, MinMaxBlockedAddress
from lib.datastore import Datastore
from agents.util import get_dirs
from plotting.plot_results import plot_episode_behavior
import gin

env_name = register_hhh_gym()
# gin.bind_parameter('DistributionTrace.traffic_trace_construct', THauke)
# gin.bind_parameter('THauke.benign_flows', 500)
# gin.bind_parameter('THauke.attack_flows', 1000)
# gin.bind_parameter('THauke.maxtime', 599)
# gin.bind_parameter('THauke.maxaddr', 0xffff)


gin.bind_parameter('DistributionTrace.traffic_trace_construct', SSDPTrace)
gin.bind_parameter('RulePerformanceTable.use_cache', True)
gin.bind_parameter('RulePerformanceTable.cache_capacity', 100)


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
    plot_episode_behavior(ds_train.environment_file.name, pattern=None, window=(2, 1))
    # plot_episode_behavior(ds_train.environment_file.name, pattern='ssdp', window=(1, 0))
    # plot_episode_behavior(ds_train.environment_file.name, pattern='ntp->ssdp', window=(1, 0))
    # plot_episode_behavior(ds_train.environment_file.name, pattern='bot->ssdp', window=(1, 0))


def run_eval_episodes(env, eval_agent, num_episodes=2):
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        step_cnt = 0
        while not done:
            print(obs)
            if type(eval_agent) == ProgressDependentEvalAgent:
                action = eval_agent.action(obs, step=None, new_episode=step_cnt == 0)
            else:
                action = eval_agent.action(obs)
            obs, rew, done, info = env.step(action)
            step_cnt += 1


def make_env(dirs, actionset):
    gym_kwargs = {
        'state_obs_selection': [BaseObservations()],
        'use_prev_action_as_obs': True,
        'actionset': actionset,
        'gamma': 0,
        'reward_calc': MultiplicativeReward(precision_weight=6, bl_weight=0.3, recall_weight=1.5, fpr_weight=3),
        'image_gen': None,
        'is_eval': True
    }
    ds_train = Datastore(dirs['root'], 'train', collect_raw=True)
    return ds_train, gym.make(env_name, **{'data_store': ds_train, **gym_kwargs})


# phi = 0.045
phi = 0.002
thresh = 0.9
L = None
actionset = DirectResolveActionSet() if thresh is None else DirectResolveRejectionActionSet()
actionset = ContinuousRejectionActionSet()
eval_agent = FixedEvalAgent(phi, L, thresh, actionset)
eval(eval_agent, actionset)
