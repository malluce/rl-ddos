import gin
import gym
import numpy as np

from gym import spaces
from gym.utils import seeding
from math import log2, sqrt

from gym.envs.registration import register

from .actionset import ActionSet
from .loop import Loop
from .obs import Observation
from .reward import RewardCalc
from .state import State
from gyms.hhh.flowgen.disttrace import DistributionTrace

from .util import maybe_cast_to_arr


def register_hhh_gym(env_name='HHHGym-v0'):
    register(
        id=env_name,
        entry_point='gyms.hhh.env:HHHEnv',
        max_episode_steps=1000,
        reward_threshold=1000.0,
        nondeterministic=True,
    )
    return env_name


@gin.configurable
class HHHEnv(gym.Env):

    def __init__(self, data_store, state_obs_selection: [Observation], use_prev_action_as_obs: bool,
                 actionset: ActionSet, gamma: float, reward_calc: RewardCalc):

        self.use_prev_action_as_obs = use_prev_action_as_obs
        self.ds = data_store
        self.trace = DistributionTrace()
        self.loop = Loop(self.trace, lambda: State(state_obs_selection), actionset)
        self.episode = 0
        self.current_step = 0

        self.action_space = self.loop.actionset.actionspace
        self.observation_space = self._observation_spec()

        self.gamma = gamma
        self.reward_calc = reward_calc

        self.seed()
        self.obs_from_state = None
        self.terminated = False
        self.rewards = []
        self.discounted_return_so_far = 0.0
        self.undiscounted_return_so_far = 0.0
        self.rules = []
        self.precisions = []
        self.recalls = []
        self.fprs = []
        self.hhh_distance_sums = []

        if self.ds is not None:
            self.ds.set_config('state_selection', str(state_obs_selection))
            self.ds.set_config('actions', str(self.loop.actionset))
            self.ds.set_config('sampling_rate', self.loop.SAMPLING_RATE)
            self.ds.set_config('hhh_epsilon', self.loop.HHH_EPSILON)
            self.ds.set_config('sampling_rate', self.loop.SAMPLING_RATE)
            self.ds.set_config('hhh_epsilon', self.loop.HHH_EPSILON)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # call loop.step
        trace_ended, state = self.loop.step(action)

        # reward
        reward = self.reward_calc.calc_reward(state)
        self.rewards.append(reward)

        # data logging
        self.rules.append(state.blacklist_size)
        self.precisions.append(state.precision)
        self.recalls.append(state.recall)
        self.fprs.append(state.fpr)
        self.hhh_distance_sums.append(state.hhh_distance_sum)
        self._log_to_datastore(trace_ended, reward, state)

        # we did 1 more step
        self.current_step += 1

        # get numpy observation array
        observation = self._build_observation(previous_action=action)

        return observation, reward, trace_ended, {}

    def _log_to_datastore(self, trace_ended, reward, state):
        if self.ds is not None:
            self.discounted_return_so_far += self.gamma ** self.current_step * reward
            self.undiscounted_return_so_far += reward
            self.ds.add_step(self.episode, self.current_step, reward, self.discounted_return_so_far,
                             self.undiscounted_return_so_far, state)

        if trace_ended and self.ds is not None:
            # compute return
            rewards = np.array(self.rewards)
            discounts = np.array([self.gamma ** step for step in np.arange(0, len(rewards))])
            return_discounted = np.dot(discounts.T, rewards)
            return_undiscounted = np.sum(rewards)

            self.ds.add_episode(self.episode + 1, 0,
                                np.mean(self.rules), np.mean(self.precisions),
                                np.mean(self.recalls), np.mean(self.fprs),
                                np.mean(self.hhh_distance_sums), np.mean(self.rewards), return_discounted,
                                return_undiscounted)

        if trace_ended:
            self.ds.flush()

    def _build_observation(self, previous_action=None):
        action_observation, state_observation = (None, None)
        if previous_action is None:
            state_observation = self.loop.state.get_initialization()
            if self.use_prev_action_as_obs:
                action_observation = maybe_cast_to_arr(self.loop.actionset.get_initialization())
        else:
            state_observation = self.loop.state.get_features()
            if self.use_prev_action_as_obs:
                action_observation = maybe_cast_to_arr(self.loop.actionset.get_observation(previous_action))

        return state_observation if not self.use_prev_action_as_obs else np.concatenate((state_observation,
                                                                                         action_observation))

    def _observation_spec(self):
        if self.use_prev_action_as_obs:
            lb = np.concatenate(
                (self.loop.state.get_lower_bounds(), maybe_cast_to_arr(self.loop.actionset.get_lower_bound())))
            ub = np.concatenate(
                (self.loop.state.get_upper_bounds(), maybe_cast_to_arr(self.loop.actionset.get_upper_bound())))
        else:
            lb = self.loop.state.get_lower_bounds()
            ub = self.loop.state.get_upper_bounds()

        return spaces.Box(
            lb, ub,
            dtype=np.float32
        )

    def reset(self):
        self.episode += 1
        self.current_step = 0
        self.trace.rewind()
        self.loop.reset()
        self.rewards = []
        self.rules = []
        self.precisions = []
        self.recalls = []
        self.fprs = []
        self.hhh_distance_sums = []
        self.discounted_return_so_far = 0.0
        self.undiscounted_return_so_far = 0.0

        return self._build_observation(previous_action=None)

    def render(self, mode):
        pass

    def close(self):
        pass
