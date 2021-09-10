import gin
import gym
import numpy as np

from gym import spaces
from gym.utils import seeding
from math import log2, sqrt

from gym.envs.registration import register

from .actionset import ActionSet
from .images import ImageGenerator
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
                 actionset: ActionSet, gamma: float, reward_calc: RewardCalc, image_gen: ImageGenerator):

        self.use_prev_action_as_obs = use_prev_action_as_obs
        self.ds = data_store
        self.trace = DistributionTrace()
        self.loop = Loop(self.trace, lambda: State(state_obs_selection), actionset, image_gen=image_gen)
        self.episode = 0
        self.current_step = 0
        self.image_gen = image_gen

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
        self.blacklists = []
        self.rules = []
        self.precisions = []
        self.recalls = []
        self.fprs = []
        self.hhh_distance_sums = []

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # call loop.step
        trace_ended, state, blacklist_history = self.loop.step(action)

        # reward
        reward = self.reward_calc.calc_reward(state)
        self.rewards.append(reward)

        # data logging
        self.blacklists += blacklist_history
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

            self.ds.add_numpy_data(self.loop.trace.trace_sampler.flows,
                                   'flows_{}'.format(self.episode + 1))
            self.ds.add_numpy_data(self.loop.trace.trace_sampler.rate_grid,
                                   'combined_{}'.format(self.episode + 1))
            self.ds.add_numpy_data(self.loop.trace.trace_sampler.attack_grid,
                                   'attack_{}'.format(self.episode + 1))
            self.ds.add_blacklist([b.to_serializable() for b in self.blacklists],
                                  'blacklist_{}'.format(self.episode + 1))

            self.ds.add_episode(self.episode + 1, 0,
                                np.mean(self.rules), np.mean(self.precisions),
                                np.mean(self.recalls), np.mean(self.fprs),
                                np.mean(self.hhh_distance_sums), np.mean(self.rewards), return_discounted,
                                return_undiscounted)

            if trace_ended:
                self.ds.flush()

    def _build_observation(self, previous_action=None):
        action_observation, state_observation = (None, None)
        img_obs, hhh_img_obs = (None, None)
        use_images = self.image_gen is not None
        if previous_action is None:
            state_observation = self.loop.state.get_initialization()
            if self.use_prev_action_as_obs:
                action_observation = maybe_cast_to_arr(self.loop.actionset.get_initialization())
            if use_images:
                img_obs = np.zeros(self.observation_space['image'].shape, dtype=self.observation_space['image'].dtype)
                hhh_img_obs = np.zeros(self.observation_space['hhh_image'].shape,
                                       dtype=self.observation_space['hhh_image'].dtype)
        else:
            state_observation = self.loop.state.get_features()
            if self.use_prev_action_as_obs:
                action_observation = maybe_cast_to_arr(self.loop.actionset.get_observation(previous_action))
            if use_images:
                img_obs = self.loop.state.image
                hhh_img_obs = self.loop.state.hhh_image
        if not self.use_prev_action_as_obs:
            vector_obs = state_observation
        else:
            vector_obs = np.concatenate((state_observation, action_observation))

        if use_images:
            assert img_obs is not None
            return {
                'vector': vector_obs,
                'image': img_obs,
                'hhh_image': hhh_img_obs
            }
        else:
            return vector_obs

    def _observation_spec(self):
        if self.use_prev_action_as_obs:
            lb = np.concatenate(
                (self.loop.state.get_lower_bounds(), maybe_cast_to_arr(self.loop.actionset.get_lower_bound())))
            ub = np.concatenate(
                (self.loop.state.get_upper_bounds(), maybe_cast_to_arr(self.loop.actionset.get_upper_bound())))
        else:
            lb = self.loop.state.get_lower_bounds()
            ub = self.loop.state.get_upper_bounds()

        # all non-image observations
        vector_spec = spaces.Box(lb, ub, dtype=np.float32)

        if self.image_gen is not None:
            complete_img_spec = self.image_gen.get_img_spec()
            hhh_img_spec = self.image_gen.get_hhh_img_spec()
            return spaces.Dict({'vector': vector_spec,
                                'image': complete_img_spec,
                                'hhh_image': hhh_img_spec
                                })
        else:
            return vector_spec

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
        self.blacklists = []
        self.discounted_return_so_far = 0.0
        self.undiscounted_return_so_far = 0.0

        return self._build_observation(previous_action=None)

    def render(self, mode):
        pass

    def close(self):
        pass
