import time

import gym
from basic_agents import FixedAgent, FixedSequenceAgent
from gyms.hhh.action import DqnMinPrefLenActionSpace, DqnRejectionActionSpace
from gyms.hhh.env import register_hhh_gym
from gyms.hhh.flowgen.traffic_traces import S2, S3
from gyms.hhh.loop import PerformanceTrackingWorstOffenderCache
from gyms.hhh.reward import MultiplicativeRewardThesis
from gyms.hhh.obs import BaseObservations
from lib.datastore import Datastore
from training.util import get_dirs
from plotting.plot_results import plot_episode_behavior
import gin
from absl import logging

env_name = register_hhh_gym()

gin.bind_parameter('S1.num_benign', 500)
gin.bind_parameter('S1.num_attack', 300)
gin.bind_parameter('DistributionTrace.traffic_trace_construct', S3)
gin.bind_parameter('RulePerformanceTable.use_cache', False)
gin.bind_parameter('RulePerformanceTable.cache_class', PerformanceTrackingWorstOffenderCache)
gin.bind_parameter('PerformanceTrackingWorstOffenderCache.capacity', 'inf')
gin.bind_parameter('RulePerformanceTable.metric', 'fpr')
gin.bind_parameter('PerformanceTrackingWorstOffenderCache.metric', 'fpr')
gin.bind_parameter('Loop.sampling_rate', 0.25)


def eval(eval_agent, action_space, episodes):
    dirs = get_dirs('/home/bachmann/test-pycharm/data', Datastore.get_timestamp(), 'eval-baseline')
    ds_train, env = make_env(dirs, action_space)
    start = time.time()
    run_eval_episodes(env, eval_agent, num_episodes=episodes)
    print(f'exec time={time.time() - start}')
    plot_episode_behavior(ds_train.environment_file.name, pattern='S3', window=(2, 0), fancy=True)


def run_eval_episodes(env, eval_agent, num_episodes=10):
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        step_cnt = 0
        while not done:
            action = eval_agent.action(obs)
            obs, rew, done, info = env.step(action)
            step_cnt += 1


def make_env(dirs, action_space):
    gym_kwargs = {
        'state_obs_selection': [BaseObservations()],
        'use_prev_action_as_obs': False,
        'action_space': action_space,
        'gamma': 0,
        'reward_calc': MultiplicativeRewardThesis(precision_weight=0, bl_weight=0.26, recall_weight=1.78,
                                                  fpr_weight=1.43),
        'image_gen': None,
        'is_eval': True
    }
    ds_train = Datastore(dirs['root'], 'train', collect_raw=True)
    return ds_train, gym.make(env_name, **{'data_store': ds_train, **gym_kwargs})


logging.set_verbosity('debug')

# DQN S1
phi = [0.09] + [0.06] * 26 + [0.07] + [0.3] * 5 + [0.008, 0.009, 0.09, 0.1] + [0.08] * 21
thresh = [0.95] + [0.925] * 26 + [0.9, 0.925, 0.85, 0.85, 0.85, 0.925, 0.95, 0.95, 0.95, 0.85, 0.95, 0.95, 0.95] + [
    0.85] * 18

# DQN S2
phi = [0.09, 0.08] + [0.06] * 4 + [0.07] + [0.07] * 3 + [0.06, 0.06] + [0.07, 0.08] + [0.07] * 7 + [0.08] + [0.09] + [
    0.1, 0.1] + [0.2] * 8 + [0.3] * 13 + [0.2] * 3 + [0.3] * 2 + [0.2, 0.08] + [0.07] * 4 + [0.3]
thresh = [0.9, 0.85] + [0.925] * 7 + [0.9, 0.925, 1, 0.975, 0.95] + [0.85] * 7 + [0.95, 0.9, 0.95, 0.95, 0.9] + [
    0.85] * 7 + [0.95] + [0.9] * 4 + [0.95] * 7 + [0.85] * 4 + [0.95, 0.9, 0.925, 0.95] + [0.85] * 5

# DQN S3
phi = [0.08] * 3 + [0.08] + [0.08] * 2 + [0.07, 0.06, 0.06, 0.05, 0.05] + [0.06] * 7 + [0.08, 0.08] + [0.09] * 6 + [
    0.08, 0.07, 0.07, 0.07, 0.08] + [0.08] * 7 + [0.08] + [0.08] * 6 + [0.07] + [0.06] * 8 + [0.07] + [0.07] * 2 + [
          0.08]
thresh = [0.975] * 3 + [0.925] * 3 + [0.95] * 3 + [0.9] + [0.95] * 8 + [0.925] * 3 + [0.95] + [0.975] + [0.925] * 4 + [
    0.9, 0.925, 0.95, 0.95, 0.95] + [0.975] * 4 + [0.95] * 3 + [0.9] * 2 + [0.925] * 4 + [0.9] * 2 + [0.925] * 2 + [
             0.95] * 9
thresh = [1.0] * 58

L = None
action_space = DqnRejectionActionSpace()
eval_agent = FixedSequenceAgent(58, phi, L, thresh, action_space)
# action_space = DqnMinPrefLenActionSpace()
# eval_agent = FixedAgent(0.05, 19, action_space=action_space)
eval(eval_agent, action_space, episodes=100)
