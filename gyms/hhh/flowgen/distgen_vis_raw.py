import os

import matplotlib

from distgen_vis2 import visualize

# rawdata_base = "/srv/bachmann/data/dqn/dqn_20211122-071838/datastore/eval/rawdata"
rawdata_base = "/srv/bachmann/data/ddpg/ddpg_20211206-081343/datastore/eval/rawdata"
# rawdata_base = "/srv/bachmann/data/ppo/ppo_20211122-095112/datastore/eval/rawdata"

# rawdata_base = '/home/bachmann/test-pycharm/data/eval-baseline_20211129-083730/datastore/train/rawdata'
# episode = 2015
# episode = 3337
episode = 188
for e in range(3339, 3340):
    flows = os.path.join(rawdata_base, f'trace_flows_{e}.npy.gz')
    combi = os.path.join(rawdata_base, f'trace_combined_{e}.npy.gz')
    attack = os.path.join(rawdata_base, f'trace_attack_{e}.npy.gz')
    blacklist = os.path.join(rawdata_base, f'trace_blacklist_{e}.json')

    try:
        visualize(flows, combi, attack, blacklist, None, None, .0001, 599)
    except FileNotFoundError:
        pass
