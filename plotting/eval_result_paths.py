# This script contains the paths to training results that are used in the evaluation.
# Currently, the results used in the thesis are used. If you reproduce some results, you must change the paths below to
# point to the new result directory (i.e., datastore root directory).
# The results can be plotted with plot_results.py

# 6.4 Adaptivity of three agents in scenario S1
dqn_s1 = '/srv/bachmann/data/dqn/dqn_20220119-070755/datastore'  # DQN S1
ddpg_s1 = '/srv/bachmann/data/ddpg/ddpg_20220120-202020/datastore'  # DDPG S1
ppo_s1 = '/srv/bachmann/data/ppo/ppo_20220120-202121/datastore'  # PPO S1

# 6.5 DDPG without BatchNorm (scenario S1)
ddpg_s1_no_bn = '/srv/bachmann/data/ddpg/ddpg_20220121-203113/datastore'

# 6.6.1 WOC vs no WOC
fix_params_no_woc = '/home/bachmann/test-pycharm/data/eval-baseline_20220122-160758/datastore'
fix_params_woc = '/home/bachmann/test-pycharm/data/eval-baseline_20220122-160733/datastore'

# 6.6.2.2 Scenario S2: DQN-pthresh vs DQN-L
ddpg_rej_s2 = '/srv/bachmann/data/ddpg/ddpg_20220124-132957/datastore'  # DDPG (not shown in thesis)
dqn_rej_s2 = '/srv/bachmann/data/dqn/dqn_20220124-125147/datastore'  # DQN-pthresh
dqn_l_s2 = '/srv/bachmann/data/dqn/dqn_20220124-204351/datastore'  # DQN-L

# 6.6.2.4 Scenario S3: DQN-pthresh vs DQN-L
dqn_rej_s3 = '/srv/bachmann/data/dqn/dqn_20220131-143520/datastore'  # DQN-pthresh without BatchNorm - not used!
dqn_l_s3 = '/srv/bachmann/data/dqn/dqn_20220131-143427/datastore'  # DQN-L without BatchNorm - not used!

dqn_rej_s3_bn = '/srv/bachmann/data/dqn/dqn_20220201-071304/datastore'  # DQN-pthresh
dqn_l_s3_bn = '/srv/bachmann/data/dqn/dqn_20220201-071412/datastore'  # DQN-L

# Below file paths contain results of additional experiments that did not make it into the thesis,
# even though the results might still be interesting.

# Comparison between disabled WOC and enabled WOC in scenario S2 (similar to 6.6.1, just for scenario S2)
s2_fix_params_no_woc = '/home/bachmann/test-pycharm/data/eval-baseline_20220209-105133/datastore'
s2_fix_params_woc = '/home/bachmann/test-pycharm/data/eval-baseline_20220209-105117/datastore'

# Comparison between disabled WOC and enabled WOC in scenario S3 (similar to 6.6.1, just for scenario S3)
s3_fix_params_no_woc = '/home/bachmann/test-pycharm/data/eval-baseline_20220209-105204/datastore'
s3_fix_params_woc = '/home/bachmann/test-pycharm/data/eval-baseline_20220209-105154/datastore'

# S3 influence of TCAM sampling rate
dqn_rej_s3_ten_percent = '/srv/bachmann/data/dqn/dqn_20220208-105402/datastore'  # 0.1
dqn_rej_s3_five_percent = '/srv/bachmann/data/dqn/dqn_20220208-105444/datastore'  # 0.05
dqn_rej_s3_one_percent = '/srv/bachmann/data/dqn/dqn_20220208-105545/datastore'  # 0.01

# S3 comparison between different agents (in thesis only DQN is used for scenario S3)
ppo_rej_s3 = '/srv/bachmann/data/ppo/ppo_20220202-070950/datastore'  # PPO-pthresh
ppo_rej_s3_bn = '/srv/bachmann/data/ppo/ppo_20220202-072037/datastore'  # PPO-pthresh with BatchNorm - didn't work well
ddpg_rej_s3 = '/srv/bachmann/data/ddpg/ddpg_20220207-102818/datastore'  # DDPG-pthresh

# S3 influence of pthresh on metrics?
chosen_thresh = '/home/bachmann/test-pycharm/data/eval-baseline_20220205-090240/datastore'  # training behavior
# minimum pthresh(0.85) chosen throughout complete episode
min_thresh = '/home/bachmann/test-pycharm/data/eval-baseline_20220205-090332/datastore'
# maximum pthresh(1.0) chosen throughout complete episode
max_thresh = '/home/bachmann/test-pycharm/data/eval-baseline_20220205-090407/datastore'
