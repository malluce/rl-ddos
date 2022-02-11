import gym
from basic_agents import FixedSequenceAgent
from gyms.hhh.action import DqnRejectionActionSpace
from gyms.hhh.env import register_hhh_gym
from gyms.hhh.flowgen.traffic_traces import S1, S2, S3
from gyms.hhh.tables import PerformanceTrackingWorstOffenderCache
from gyms.hhh.reward import MultiplicativeRewardThesis
from gyms.hhh.obs import BaseObservations
from lib.datastore import Datastore
from training.util import get_dirs
from plotting.plot_results import plot_episode_behavior
import gin
from absl import logging

# median chosen phi values per adaptation step by trained DQN-pthresh agent in different scenarios
MEDIAN_CHOSEN_PHI = {
    'S1': [0.09] + [0.06] * 26 + [0.07] + [0.3] * 5 + [0.008, 0.009, 0.09, 0.1] + [0.08] * 21,
    'S2': [0.09, 0.08] + [0.06] * 4 + [0.07] + [0.07] * 3 + [0.06, 0.06] + [0.07, 0.08] + [0.07] * 7 + [0.08] + [
        0.09] + [0.1, 0.1] + [0.2] * 8 + [0.3] * 13 + [0.2] * 3 + [0.3] * 2 + [0.2, 0.08] + [0.07] * 4 + [0.3],
    'S3:': [0.08] * 3 + [0.08] + [0.08] * 2 + [0.07, 0.06, 0.06, 0.05, 0.05] + [0.06] * 7 + [0.08, 0.08] + [
        0.09] * 6 + [0.08, 0.07, 0.07, 0.07, 0.08] + [0.08] * 7 + [0.08] + [0.08] * 6 + [0.07] + [0.06] * 8 + [0.07] + [
               0.07] * 2 + [0.08]
}

# median chosen pthresh values per adaptation step by trained DQN-pthresh agent in different scenarios
MEDIAN_CHOSEN_PTHRESH = {
    'S1': [0.95] + [0.925] * 26 + [0.9, 0.925, 0.85, 0.85, 0.85, 0.925, 0.95, 0.95, 0.95, 0.85, 0.95, 0.95, 0.95] + [
        0.85] * 18,
    'S2': [0.9, 0.85] + [0.925] * 7 + [0.9, 0.925, 1, 0.975, 0.95] + [0.85] * 7 + [0.95, 0.9, 0.95, 0.95, 0.9] + [
        0.85] * 7 + [0.95] + [0.9] * 4 + [0.95] * 7 + [0.85] * 4 + [0.95, 0.9, 0.925, 0.95] + [0.85] * 5,
    'S3': [0.975] * 3 + [0.925] * 3 + [0.95] * 3 + [0.9] + [0.95] * 8 + [0.925] * 3 + [0.95] + [0.975] + [
        0.925] * 4 + [0.9, 0.925, 0.95, 0.95, 0.95] + [0.975] * 4 + [0.95] * 3 + [0.9] * 2 + [0.925] * 4 + [0.9] * 2 + [
              0.925] * 2 + [0.95] * 9
}


def evaluate(eval_agent, action_space, episodes, base_dir, scenario_id):
    dirs = get_dirs(base_dir, Datastore.get_timestamp(), 'eval-baseline')
    ds_train, env = make_env(dirs, action_space)
    run_eval_episodes(env, eval_agent, num_episodes=episodes)
    plot_episode_behavior(ds_train.environment_file.name, pattern=scenario_id, window=(100, 0), fancy=True)


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
        'reward_calc': MultiplicativeRewardThesis(precision_weight=0, bl_weight=0.26, recall_weight=1.5, fpr_weight=3),
        'image_gen': None,  # don't compute images to save computation time, state is not used here anyway
        'is_eval': True
    }
    ds_train = Datastore(dirs['root'], 'train', collect_raw=True)
    return ds_train, gym.make(env_name, **{'data_store': ds_train, **gym_kwargs})


if __name__ == '__main__':
    ##### MODIFY THESE PARAMETERS FOR EXPERIMENTATION #####
    base_dir = '/home/bachmann/test-pycharm/data'  # where to store the result files (sub-dir is created therein)
    gin.bind_parameter('RulePerformanceTable.use_cache', False)  # toggle to enable/disable WOC
    gin.bind_parameter('Loop.sampling_rate', 0.25)  # sets TCAM sampling rate
    scenario = S1  # select scenario (S1,S2,S3) to re-play median parameter selections as learned by DQN-pthresh agent
    episodes_to_run = 100  # number of episodes tu run the environment for
    #####

    env_name = register_hhh_gym()
    logging.set_verbosity('debug')
    gin.bind_parameter('S1.num_benign', 500)
    gin.bind_parameter('S1.num_attack', 300)
    gin.bind_parameter('DistributionTrace.traffic_trace_construct', scenario)
    gin.bind_parameter('RulePerformanceTable.cache_class', PerformanceTrackingWorstOffenderCache)
    gin.bind_parameter('PerformanceTrackingWorstOffenderCache.capacity', 'inf')
    gin.bind_parameter('RulePerformanceTable.metric', 'fpr')
    gin.bind_parameter('PerformanceTrackingWorstOffenderCache.metric', 'fpr')

    scenario_id = scenario.__name__
    action_space = DqnRejectionActionSpace()
    agent = FixedSequenceAgent(
        episode_length=58,
        phis=MEDIAN_CHOSEN_PHI[scenario_id],
        min_prefix_lengths=None,
        pthreshs=MEDIAN_CHOSEN_PTHRESH[scenario_id],
        action_space=action_space
    )
    evaluate(agent, action_space, episodes=episodes_to_run, base_dir=base_dir, scenario_id=scenario_id)
