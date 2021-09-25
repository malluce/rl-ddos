import gin
import gym
import numpy as np

from gym import spaces
from gym.utils import seeding
from math import log2, sqrt

from gym.envs.registration import register

from .actionset import ActionSet, HafnerActionSet
from .flowgen.traffic_traces import SamplerTrafficTrace
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


def pattern_ids_to_pattern_sequence(pattern_ids):
    # remove redundant patterns, retain sequence of used patterns (e.g [p1,p1,p2,p3,p2]->[p1,p2,p3,p2])
    pattern_sequence = []
    for pat in pattern_ids:
        if len(pattern_sequence) == 0 or pattern_sequence[-1] != pat:
            pattern_sequence.append(pat)

    if len(pattern_sequence) == 1:
        return pattern_sequence[0]
    else:
        seq_str = pattern_sequence[0]
        for pat in pattern_sequence[1:]:
            seq_str += f'->{pat}'
        return seq_str


@gin.configurable
class HHHEnv(gym.Env):

    def __init__(self, data_store, state_obs_selection: [Observation], use_prev_action_as_obs: bool,
                 actionset: ActionSet, gamma: float, reward_calc: RewardCalc, image_gen: ImageGenerator, is_eval: bool):

        self.use_prev_action_as_obs = use_prev_action_as_obs
        self.ds = data_store
        self.trace = DistributionTrace(is_eval=is_eval)
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
        self.source_pattern_ids = []
        self.rate_pattern_ids = []
        self.change_pattern_ids = []

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

        # print(f'observation={observation}')
        # print(f'reward={reward}')
        # print(f'trace_ended={trace_ended}')

        return observation, reward, trace_ended, {}

    def _log_to_datastore(self, trace_ended, reward, state):
        if self.ds is not None:
            self.discounted_return_so_far += self.gamma ** self.current_step * reward
            self.undiscounted_return_so_far += reward

            traffic_trace: SamplerTrafficTrace = self.trace.traffic_trace
            source_id = traffic_trace.get_source_pattern_id(self.current_step)
            rate_id = traffic_trace.get_rate_pattern_id(self.current_step)
            change_id = traffic_trace.get_change_pattern_id()

            self.ds.add_step(self.episode, self.current_step, reward, source_id, rate_id, change_id,
                             self.discounted_return_so_far,
                             self.undiscounted_return_so_far, state)

            self.source_pattern_ids.append(source_id)
            self.rate_pattern_ids.append(rate_id)
            self.change_pattern_ids.append(change_id)

        if trace_ended and self.ds is not None:
            # trace pattern sequences
            source_seq = pattern_ids_to_pattern_sequence(self.source_pattern_ids)
            rate_seq = pattern_ids_to_pattern_sequence(self.rate_pattern_ids)
            unique_change_pattern_ids = set(self.change_pattern_ids)
            if len(unique_change_pattern_ids) != 1:
                raise ValueError(f'Encountered {len(unique_change_pattern_ids)} change pattern IDs, expected 1.')
            change_id = unique_change_pattern_ids.pop()

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

            self.ds.add_episode(self.episode + 1, 0, source_seq, rate_seq, change_id,
                                np.mean(self.rules), np.mean(self.precisions),
                                np.mean(self.recalls), np.mean(self.fprs),
                                np.mean(self.hhh_distance_sums), np.mean(self.rewards), return_discounted,
                                return_undiscounted)

            if trace_ended:
                self.ds.flush()

    def _build_observation(self, previous_action=None):
        # print(f'building observation, previous={previous_action}')
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
        self.source_pattern_ids = []
        self.rate_pattern_ids = []
        self.change_pattern_ids = []

        if isinstance(self.loop.actionset, HafnerActionSet):
            return self._build_observation(previous_action=0)
        else:
            return self._build_observation(previous_action=None)

    def render(self, mode):
        pass

    def close(self):
        pass
