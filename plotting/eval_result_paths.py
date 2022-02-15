import os
from collections import namedtuple

# This script contains the paths to training results that are used in the evaluation.
# Currently, path to the results shown in the thesis are defined.
# If you reproduce some results, you must change the paths below to
# point to the new results directories. For raw paths, episode numbers have to be changed.
# The results can be plotted with plot_results.py and rawdata_vis.py

project_dir = '/home/bachmann/test-pycharm'  # TODO change this line to project root dir location

# for raw data ("Generated rules" plots)
RawPath = namedtuple('raw_path', ['path', 'episode_number'])


def to_raw_path(datastore_path, train=False):
    return os.path.join(datastore_path, 'eval/rawdata' if not train else 'train/rawdata')


# 6.4 Adaptivity of three agents in scenario S1
dqn_s1 = os.path.join(project_dir, 'data/dqn/dqn_20220119-070755/datastore')  # DQN S1
raw_dqn_s1 = RawPath(path=to_raw_path(dqn_s1), episode_number=503)
ddpg_s1 = os.path.join(project_dir, 'data/ddpg/ddpg_20220120-202020/datastore')  # DDPG S1
raw_ddpg_s1 = RawPath(path=to_raw_path(ddpg_s1), episode_number=1662)
ppo_s1 = os.path.join(project_dir, 'data/ppo/ppo_20220120-202121/datastore')  # PPO S1
raw_ppo_s1 = RawPath(path=to_raw_path(ppo_s1), episode_number=1664)

# 6.5 DDPG without BatchNorm (scenario S1)
ddpg_s1_no_bn = os.path.join(project_dir, 'data/ddpg/ddpg_20220121-203113/datastore')
raw_ddpg_s1_no_bn = RawPath(path=to_raw_path(ddpg_s1_no_bn), episode_number=1663)

# 6.6.1 WOC vs no WOC
fix_params_no_woc = os.path.join(project_dir, 'data/sim/eval-baseline_20220122-160758/datastore')
raw_fix_params_no_woc = RawPath(path=to_raw_path(fix_params_no_woc, train=True), episode_number=1)
fix_params_woc = os.path.join(project_dir, 'data/sim/eval-baseline_20220122-160733/datastore')
raw_fix_params_woc = RawPath(path=to_raw_path(fix_params_woc, train=True), episode_number=1)  # not shown in thesis

# 6.6.2.2 Scenario S2: DQN-pthresh vs DQN-L
dqn_rej_s2 = os.path.join(project_dir, 'data/dqn/dqn_20220124-125147/datastore')  # DQN-pthresh
raw_dqn_rej_s2 = RawPath(path=to_raw_path(dqn_rej_s2), episode_number=506)
dqn_l_s2 = os.path.join(project_dir, 'data/dqn/dqn_20220124-204351/datastore')  # DQN-L
raw_dqn_l_s2 = RawPath(path=to_raw_path(dqn_l_s2), episode_number=507)

# 6.6.2.4 Scenario S3: DQN-pthresh vs DQN-L
dqn_rej_s3 = os.path.join(project_dir,
                          'data/dqn/dqn_20220131-143520/datastore')  # DQN-pthresh without BatchNorm - not used!
dqn_l_s3 = os.path.join(project_dir, 'data/dqn/dqn_20220131-143427/datastore')  # DQN-L without BatchNorm - not used!

dqn_rej_s3_bn = os.path.join(project_dir, 'data/dqn/dqn_20220201-071304/datastore')  # DQN-pthresh
raw_dqn_rej_s3_bn = RawPath(path=to_raw_path(dqn_rej_s3_bn), episode_number=509)
dqn_l_s3_bn = os.path.join(project_dir, 'data/dqn/dqn_20220201-071412/datastore')  # DQN-L
raw_dqn_l_s3_bn = RawPath(path=to_raw_path(dqn_l_s3_bn), episode_number=505)

# Below file paths contain results of additional experiments that did not make it into the thesis,
# even though the results might still be interesting.

# Comparison between disabled WOC and enabled WOC in scenario S2 (similar to 6.6.1, just for scenario S2)
s2_fix_params_no_woc = os.path.join(project_dir, 'data/sim/eval-baseline_20220209-105133/datastore')
s2_fix_params_woc = os.path.join(project_dir, 'data/sim/eval-baseline_20220209-105117/datastore')

# Comparison between disabled WOC and enabled WOC in scenario S3 (similar to 6.6.1, just for scenario S3)
s3_fix_params_no_woc = os.path.join(project_dir, 'data/sim/eval-baseline_20220209-105204/datastore')
s3_fix_params_woc = os.path.join(project_dir, 'data/sim/eval-baseline_20220209-105154/datastore')

# S3 influence of TCAM sampling rate
dqn_rej_s3_ten_percent = os.path.join(project_dir, 'data/dqn/dqn_20220208-105402/datastore')  # 0.1
dqn_rej_s3_five_percent = os.path.join(project_dir, 'data/dqn/dqn_20220208-105444/datastore')  # 0.05
dqn_rej_s3_one_percent = os.path.join(project_dir, 'data/dqn/dqn_20220208-105545/datastore')  # 0.01

# S3 comparison between different agents (in thesis only DQN is used for scenario S3)
ppo_rej_s3 = os.path.join(project_dir, 'data/ppo/ppo_20220202-070950/datastore')  # PPO-pthresh
ppo_rej_s3_bn = os.path.join(project_dir,
                             'data/ppo/ppo_20220202-072037/datastore')  # PPO-pthresh with BatchNorm - didn't work well
ddpg_rej_s3 = os.path.join(project_dir, 'data/ddpg/ddpg_20220207-102818/datastore')  # DDPG-pthresh

# S3 influence of pthresh on metrics?
chosen_thresh = os.path.join(project_dir, 'data/sim/eval-baseline_20220205-090240/datastore')  # training behavior
# minimum pthresh(0.85) chosen throughout complete episode
min_thresh = os.path.join(project_dir, 'data/sim/eval-baseline_20220205-090332/datastore')
# maximum pthresh(1.0) chosen throughout complete episode
max_thresh = os.path.join(project_dir, 'data/sim/eval-baseline_20220205-090407/datastore')
