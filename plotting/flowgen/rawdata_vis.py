import os

from scenario_vis import visualize

# This script is used to plot the "Generate filter rules" plots from the eval chapter.
# TODO

# rawdata_base = '/home/bachmann/test-pycharm/data/eval-baseline_20220122-160758/datastore/train/rawdata'
rawdata_base = "/srv/bachmann/data/dqn/dqn_20220201-071412/datastore/eval/rawdata"
# rawdata_base = "/srv/bachmann/data/ddpg/ddpg_20220121-203113/datastore/eval/rawdata"
# rawdata_base = "/srv/bachmann/data/ppo/ppo_20220120-202121/datastore/eval/rawdata"
episode = 188
# range(1660, 1670) # DDPG/PPO
# range(500,510) # DQN
for e in range(505, 510):
    flows = os.path.join(rawdata_base, f'trace_flows_{e}.npy.gz')
    combi = os.path.join(rawdata_base, f'trace_combined_{e}.npy.gz')
    attack = os.path.join(rawdata_base, f'trace_attack_{e}.npy.gz')
    blacklist = os.path.join(rawdata_base, f'trace_blacklist_{e}.json')

    try:
        visualize(flows, combi, attack, blacklist, None, None, .0001, 599, trace_id='S3')
    except FileNotFoundError:
        pass
