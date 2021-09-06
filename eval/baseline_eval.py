import gym
from eval_agents import FixedEvalAgent, ProgressDependentEvalAgent, RandomEvalAgent
from gyms.hhh.actionset import ActionSet, DirectResolveActionSet, LargeDiscreteActionSet
from gyms.hhh.env import register_hhh_gym
from gyms.hhh.flowgen.traffic_traces import T4
from gyms.hhh.reward import DefaultRewardCalc
from gyms.hhh.state import BaseObservations, BlocklistDistribution, DistVol, DistVolStd, FalsePositiveRate, \
    MinMaxBlockedAddress
from lib.datastore import Datastore
from agents.util import get_dirs
from plotting.plot_results import plot_episode_behavior
import gin

env_name = register_hhh_gym()
gin.bind_parameter('DistributionTrace.traffic_trace', T4(maxaddr=0xffff))


def eval(eval_agent, action_set, label):
    dirs = get_dirs('/home/bachmann/test-pycharm/data', Datastore.get_timestamp(), 'eval-baseline')
    ds_train, env = make_env(dirs, action_set)
    # eval_agent = RandomEvalAgent(env.loop.actionset.actionspace, env.loop.actionset)
    run_eval_episodes(env, eval_agent)
    plot_episode_behavior(ds_train.environment_file.name, last_x_episodes=1)


def run_eval_episodes(env, eval_agent, num_episodes=5):
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        step_cnt = 0
        while not done:
            if type(eval_agent) == ProgressDependentEvalAgent:
                action = eval_agent.action(obs, step=None, new_episode=step_cnt == 0)
            else:
                action = eval_agent.action(obs)
            obs, rew, done, info = env.step(action)
            step_cnt += 1


def make_env(dirs, actionset):
    gym_kwargs = {
        'state_obs_selection': [BaseObservations(), FalsePositiveRate(), DistVol(), MinMaxBlockedAddress(),
                                DistVolStd(),
                                BlocklistDistribution()],
        'use_prev_action_as_obs': False,
        'actionset': actionset,
        'gamma': 0,
        'reward_calc': DefaultRewardCalc(),
        'image_gen': None
    }
    ds_train = Datastore(dirs['root'], 'train')
    return ds_train, gym.make(env_name, **{'data_store': ds_train, **gym_kwargs})


phi = 0.57
L = 21
eval_agent = FixedEvalAgent(phi, L)
eval_agent = ProgressDependentEvalAgent(
    phis=[0.25, 0.15],
    min_prefix_lengths=[18, 20],
    step_bounds=[30]
)
eval(eval_agent, DirectResolveActionSet(), f'$\phi={phi}$, $L={L}$')
