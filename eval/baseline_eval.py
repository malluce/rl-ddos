import gym
from eval_agents import FixedEvalAgent, RandomEvalAgent
from gyms.hhh.actionset import ActionSet, DirectResolveActionSet, LargeDiscreteActionSet
from gyms.hhh.env import register_hhh_gym
from gyms.hhh.state import BaseObservations, BlocklistDistribution, DistVol, DistVolStd, FalsePositiveRate, \
    MinMaxBlockedAddress
from lib.datastore import Datastore
from agents.util import get_dirs
from plotting.plot_results import plot_episode_behavior

env_name = register_hhh_gym()


def eval(eval_agent, action_set, label):
    dirs = get_dirs('/home/bachmann/test-pycharm/data', Datastore.get_timestamp(), 'eval-baseline')
    ds_train, env = make_env(dirs, action_set)
    run_eval_episodes(env, eval_agent)
    plot_episode_behavior(ds_train.environment_file.name)


def run_eval_episodes(env, eval_agent, num_episodes=5):
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = eval_agent.action(obs)
            obs, rew, done, info = env.step(action)


def make_env(dirs, actionset):
    gym_kwargs = {
        'state_obs_selection': [BaseObservations(), FalsePositiveRate(), DistVol(), MinMaxBlockedAddress(),
                                DistVolStd(),
                                BlocklistDistribution()],
        'use_prev_action_as_obs': False,
        'actionset': actionset
    }
    ds_train = Datastore(dirs['root'], 'train')
    return ds_train, gym.make(env_name, **{'data_store': ds_train, **gym_kwargs})


# action_set = LargeDiscreteActionSet()
# eval_agent = RandomEvalAgent(action_set.actionspace, action_set)
# eval(eval_agent, action_set, 'random(LargeDiscrete)')

phi = 0.57
L = 21
eval_agent = FixedEvalAgent(phi, L)
eval(eval_agent, DirectResolveActionSet(), f'$\phi={phi}$, $L={L}$')
