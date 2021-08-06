import gym
import numpy as np

from gym import spaces
from gym.utils import seeding
from math import log2, sqrt

from gym.envs.registration import register

from .actionset import ActionSet
from .loop import Loop
from .obs import Observation
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


class HHHEnv(gym.Env):

    def __init__(self, data_store, state_obs_selection: [Observation], use_prev_action_as_obs: bool,
                 actionset: ActionSet):

        self.use_prev_action_as_obs = use_prev_action_as_obs
        self.ds = data_store
        self.trace = DistributionTrace()
        self.loop = Loop(self.trace, lambda: State(state_obs_selection), actionset)
        self.episode = 0
        self.current_step = 0

        self.action_space = self.loop.actionset.actionspace
        self.observation_space = self._observation_spec()

        self.seed()
        self.figure = None
        self.obs_from_state = None
        self.terminated = False
        self.rewards = []
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
        reward = self._calc_reward(state)
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

    def _calc_reward(self, state):
        if state.blacklist_size == 0:
            # fpr is 0 (nothing blocked, so there are no false positives)
            # precision and recall are 1 if there is no malicious packet, 0 otherwise
            reward = state.precision * state.recall
        else:
            reward = (
                    (state.precision ** 6) * sqrt(state.recall) * (1 - sqrt(state.fpr))
                    * (1.0 - 0.2 * sqrt(log2(state.blacklist_size)))
            )
        return reward

    def _log_to_datastore(self, trace_ended, reward, state):
        if self.ds is not None:
            self.ds.add_step(self.episode, self.current_step, reward, state)

        if trace_ended and self.ds is not None:
            self.ds.add_episode(self.episode + 1, 0,
                                np.mean(self.rules), np.mean(self.precisions),
                                np.mean(self.recalls), np.mean(self.fprs),
                                np.mean(self.hhh_distance_sums), np.mean(self.rewards))

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

        return self._build_observation(previous_action=None)

    def render(self, mode):
        pass

    def close(self):
        pass
