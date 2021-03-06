import gin
import gym
import numpy as np
from absl import logging

from gym import spaces
from gym.utils import seeding

from gym.envs.registration import register

from .action import ActionSpace
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
    """Implementation of the simulated DDoS mitigation environment, following the OpenAI Gym interface."""

    def __init__(self, data_store, state_obs_selection: [Observation], use_prev_action_as_obs: bool,
                 action_space: ActionSpace, gamma: float, reward_calc: RewardCalc, image_gen: ImageGenerator,
                 is_eval: bool):

        self.use_prev_action_as_obs = use_prev_action_as_obs
        self.ds = data_store
        self.trace = DistributionTrace(is_eval=is_eval)
        self.loop = Loop(self.trace, lambda: State(state_obs_selection), action_space, image_gen=image_gen)
        self.episode = 0
        self.current_step = 0
        self.image_gen = image_gen

        self.action_space = self.loop.action_space.actionspace
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
        # call loop.step (performs the mitigation system simulation)
        logging.debug(self.trace.traffic_trace.get_source_pattern_id(self.current_step))
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
        logging.debug(
            f'reward={reward} (prec={self.loop.state.precision}, rec={self.loop.state.recall}, fpr={self.loop.state.fpr}, bl size={self.loop.state.blacklist_size})')

        if trace_ended:
            logging.debug('==========================================================')
        else:
            logging.debug('===================')
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
                                   'flows_{}'.format(self.episode))
            self.ds.add_numpy_data(self.loop.trace.trace_sampler.rate_grid,
                                   'combined_{}'.format(self.episode))
            self.ds.add_numpy_data(self.loop.trace.trace_sampler.attack_grid,
                                   'attack_{}'.format(self.episode))
            self.ds.add_blacklist(self.blacklists,
                                  'blacklist_{}'.format(self.episode))

            self.ds.add_episode(self.episode, 0, source_seq, rate_seq, change_id,
                                np.mean(self.rules), np.mean(self.precisions),
                                np.mean(self.recalls), np.mean(self.fprs),
                                np.mean(self.hhh_distance_sums), np.mean(self.rewards), return_discounted,
                                return_undiscounted)

            if trace_ended:
                self.ds.flush()

    def _build_observation(self, previous_action):
        assert previous_action is not None
        use_images = self.image_gen is not None
        state_observation = self.loop.state.get_features()
        if self.use_prev_action_as_obs:
            action_observation = maybe_cast_to_arr(self.loop.action_space.get_observation(previous_action))
            vector_obs = np.concatenate((state_observation, action_observation))
        else:
            vector_obs = state_observation

        if use_images:
            return {
                'vector': vector_obs,
                'hhh_image': self.loop.state.hhh_image
            }
        else:
            return vector_obs

    def _observation_spec(self):
        if self.use_prev_action_as_obs:
            lb = np.concatenate(
                (self.loop.state.get_lower_bounds(), maybe_cast_to_arr(self.loop.action_space.get_lower_bound())))
            ub = np.concatenate(
                (self.loop.state.get_upper_bounds(), maybe_cast_to_arr(self.loop.action_space.get_upper_bound())))
        else:
            lb = self.loop.state.get_lower_bounds()
            ub = self.loop.state.get_upper_bounds()

        # all non-image observations
        vector_spec = spaces.Box(lb, ub, dtype=np.float32)

        if self.image_gen is not None:
            hhh_img_spec = self.image_gen.get_hhh_img_spec()
            return spaces.Dict({'vector': vector_spec,
                                'hhh_image': hhh_img_spec
                                })
        else:
            return vector_spec

    def reset(self):
        self.episode += 1
        self.current_step = 0
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
        self.trace.rewind()
        # make one step with randomly chosen action (for first observation)
        first_action, blacklist_history = self.loop.reset()
        self.blacklists += blacklist_history
        return self._build_observation(previous_action=first_action)

    def render(self, mode):
        pass

    def close(self):
        pass
