import gin
import gym
import numpy as np

from datetime import datetime
from gym import spaces, logger
from gym.utils import seeding
from importlib import import_module
from math import log, log2, sqrt
from random import choice, random, randint, shuffle
from matplotlib import pyplot as plt
from matplotlib import cm

from gym.envs.registration import register

from .loop import Loop
from .label import Label
from .obs import Observation
from .state import State
from .disttrace import DistributionTrace

from lib.datastore import Datastore


def register_hhh_gym(env_name='HHHGym-v0'):
    register(
        id=env_name,
        entry_point='gyms.hhh.env:HHHEnv',
        max_episode_steps=1000,
        reward_threshold=1000.0,
        nondeterministic=True,
    )
    return env_name


class Packet(object):

    def __init__(self, ip, malicious):
        self.ip = ip
        self.malicious = malicious


@gin.configurable
class PacketGenerator(object):
    _SPARSE_RANGES = 20

    def __init__(self, baselen=16, prob_benign=0.2):
        self.baselen = baselen
        self.prob_benign = prob_benign
        self.split = None
        self.benign = None
        self.malicious = None
        self.generate_ranges = None
        self.sampling_function = None
        self.select_sampling_method()

    def select_sampling_method(self):
        if True:  # random() > 1.0/3:
            self.generate_ranges = lambda: self.generate_compact_ranges
            self.sampling_function = lambda: self.sample_compact
        else:
            self.generate_ranges = lambda: self.generate_sparse_ranges
            self.sampling_function = lambda: self.sample_sparse

    # Compact range sampling

    def generate_compact_ranges(self):
        # Split range into even sized chunks and select
        # half of them as malicious address ranges.
        # Select entire range for benign addesses.
        self.split = choice([2, 4, 6])

        lo = 0xffffffff & Label.PREFIXMASK[self.baselen]
        hi = 0xffffffff
        m = [(lo, hi)]

        self.benign = [(lo, hi)]

        # Split address range in even sized chunks
        for i in range(self.split):
            l = []

            for lo, hi in m:
                s = (lo + hi) // 2
                l.append((lo, s))
                l.append((s + 1, hi))

            m = l

        shuffle(m)

        self.malicious = m[:len(m) // 2]

    def sample_compact(self):
        m = random() > self.prob_benign

        if m:
            range = choice(self.malicious)
        else:
            range = self.benign[0]

        return Packet(randint(range[0], range[1]), m)

    # Sparse sampling

    def generate_sparse_ranges(self):
        self.split = 4  # log2(20)

        lo = 0xffffffff & Label._prefixmask(self.baselen, 32)
        hi = 0xffffffff

        self.benign = (lo, hi)
        self.malicious = [randint(lo, hi)
                          for _ in range(PacketGenerator._SPARSE_RANGES)]

    def sample_sparse(self):
        m = random() > self.prob_benign

        if m:
            ip = choice(self.malicious)
        else:
            ip = randint(self.benign[0], self.benign[1])

        return Packet(ip, m)

    def __next__(self):
        return self.next()

    def next(self):
        if self.malicious is None:
            self.generate_ranges()()

        return self.sampling_function()()


class Trace(object):

    def __init__(self, trace_length):
        self.N = trace_length
        self.i = 0
        self.g = PacketGenerator()

    def __next__(self):
        return self.next()

    def sample(self):
        return self.g.next()

    def next(self):
        if self.i == self.N - 1:
            raise StopIteration()

        self.i += 1

        return self.sample()

    def rewind(self):
        self.i = 0
        self.g = PacketGenerator()

    def __len__(self):
        return self.N


class HHHEnv(gym.Env):

    def __init__(self, data_store, state_obs_selection: [Observation], use_prev_action_as_obs: bool,
                 actionset_selection,
                 trace_length):
        actionset = actionset_selection()

        self.use_prev_action_as_obs = use_prev_action_as_obs
        self.ds = data_store
        self.trace = Trace(trace_length)
        # self.trace = DistributionTrace(trace_length) ## TODO revert commit
        self.loop = Loop(self.trace, lambda: State(state_obs_selection), actionset)
        self.episode = 0
        # self.current_step = 0 ## TODO revert commit

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
            self.ds.set_config('benign_probability', self.trace.g.prob_benign)  ## TODO revert commit
            self.ds.set_config('trace_length', len(self.trace))

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
        # self.current_step += 1 ## TODO revert commit

        # get numpy observation array
        observation = self._build_observation(previous_action=action)

        return observation, reward, trace_ended, {}

    def _calc_reward(self, state):
        if state.blacklist_size == 0:
            reward = 0.0
        else:
            reward = ((state.precision ** 4) * state.recall * (1 - sqrt(state.fpr)))
            # reward = ( ## TODO revert commit
            #        (state.precision ** 6) * sqrt(state.recall) * (1 - sqrt(state.fpr))
            #        * (1.0 - 0.2 * sqrt(log2(state.blacklist_size)))
            # )
        return reward

    def _log_to_datastore(self, trace_ended, reward, state):
        if self.ds is not None:
            self.ds.add_step(self.episode, self.trace.g.split, reward, state)
            # self.ds.add_step(self.episode, self.current_step, reward, state)

        if trace_ended and self.ds is not None:
            self.ds.add_episode(self.episode + 1, self.trace.g.split,
                                np.mean(self.rules), np.mean(self.precisions),
                                np.mean(self.recalls), np.mean(self.fprs),
                                np.mean(self.hhh_distance_sums), np.mean(self.rewards))
            # self.ds.add_episode(self.episode + 1, 0,
            #                    np.mean(self.rules), np.mean(self.precisions),
            #                    np.mean(self.recalls), np.mean(self.fprs),
            #                    np.mean(self.hhh_distance_sums), np.mean(self.rewards))

    def _build_observation(self, previous_action=None):
        action_observation, state_observation = (None, None)
        if previous_action is None:
            state_observation = self.loop.state.get_initialization()
            if self.use_prev_action_as_obs:
                action_observation = self.loop.actionset.get_initialization()
        else:
            state_observation = self.loop.state.get_features()
            if self.use_prev_action_as_obs:
                action_observation = self.loop.actionset.get_observation(previous_action)

        return state_observation if not self.use_prev_action_as_obs else np.concatenate((state_observation,
                                                                                         action_observation))

    def _observation_spec(self):
        if self.use_prev_action_as_obs:
            lb = np.concatenate((self.loop.state.get_lower_bounds(), self.loop.actionset.get_lower_bound()))
            ub = np.concatenate((self.loop.state.get_upper_bounds(), self.loop.actionset.get_upper_bound()))
        else:
            lb = self.loop.state.get_lower_bounds()
            ub = self.loop.state.get_upper_bounds()

        return spaces.Box(
            lb, ub,
            dtype=np.float32
        )

    def reset(self):
        print('Resetting env')
        self.episode += 1
        # self.current_step = 0 ## TODO revert commit
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
        return
        width = 800
        height = 512

        def on_close(ev):
            self.terminated = True

        if self.figure is None:
            self.figure = plt.figure(figsize=(8, 4), dpi=200)
            self.img_data = np.ones((height, width, 4))
            self.xcoord = 0
            self.xticks = ([], [])

            axes = plt.axes()
            axes.spines['top'].set_visible(False)
            axes.spines['left'].set_visible(False)
            axes.spines['right'].set_visible(False)

            self.figure.canvas.mpl_connect('close_event', on_close)

            self.image = plt.imshow(self.img_data, interpolation='nearest')

            plt.ion()
            plt.yticks([], [])
            plt.tick_params('x', labelsize=4)
            plt.show()

        if self.xcoord == width - 2:
            self.xcoord = 0

        if not self.xcoord:
            self.img_data = np.ones((height, width, 4))

        self.xcoord += 2

        if self.trace.i == 0:
            self.xticks[0].append(self.xcoord)
            self.xticks[1].append(self.episode)
            plt.xticks(self.xticks[0], self.xticks[1])

        for p in self.loop.packets:
            idx = (p.ip >> 7) & 0x1ff
            rgba = self.img_data[idx][self.xcoord]

            if p.malicious:
                rgba[1] *= 0.98
                rgba[2] *= 0.98
            else:
                rgba[0] *= 0.98
                rgba[1] *= 0.98

        for b in self.loop.blacklist.hhhs:
            id = b.id
            length = b.len

            for idx in range(id >> 7, (id + (0x1 << (32 - length))) >> 7):
                idx &= 0x1ff

                rgba = self.img_data[idx][self.xcoord + 1]

                rgba[0] = 0.0
                rgba[1] = 0.0
                rgba[2] = 0.0

        if not self.terminated:
            self.image.set_data(self.img_data)

            try:
                plt.draw()
                plt.pause(0.025)
            except TclError:
                return True

        return self.terminated

    def close(self):
        pass
