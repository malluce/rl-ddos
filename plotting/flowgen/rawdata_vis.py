from plotting.eval_result_paths import *
from scenario_vis import visualize


# This script is used to plot the "Generated filter rules" plots from the eval chapter.
# (i.e., Figures 6.8, 6.9, 6.10, 6.13, 6.15, 6.23, 6.24, 6.30, 6.31)
# Select a section to reproduce these plots for by changing the line of code in main, then run the script.


def create_generated_rules_plot(raw_path):
    rawdata_base = raw_path.path
    ep = raw_path.episode_number
    flows = os.path.join(rawdata_base, f'trace_flows_{ep}.npy.gz')
    combined = os.path.join(rawdata_base, f'trace_combined_{ep}.npy.gz')
    attack = os.path.join(rawdata_base, f'trace_attack_{ep}.npy.gz')
    blacklist = os.path.join(rawdata_base, f'trace_blacklist_{ep}.json')
    visualize(flows, combined, attack, blacklist, None, None, 0.0001, 599, trace_id=scenario)


if __name__ == '__main__':
    # change to one of (6.4, 6.5, 6.6.1, 6.6.2.2, 6.6.2.4) to reproduce "Generated filter rules" plots
    section = '6.4'

    if section in ['6.4', '6.5', '6.6.1']:
        scenario = 'S1'
    elif section == '6.6.2.2':
        scenario = 'S2'
    elif section == '6.6.2.4':
        scenario = 'S3'
    else:
        raise ValueError(f'Section {section} unknown')

    if section == '6.4':
        create_generated_rules_plot(raw_ppo_s1)  # Fig 6.8
        create_generated_rules_plot(raw_dqn_s1)  # Fig 6.9
        create_generated_rules_plot(raw_ddpg_s1)  # Fig 6.10
    elif section == '6.5':
        create_generated_rules_plot(raw_ddpg_s1_no_bn)  # Fig 6.13
    elif section == '6.6.1':
        create_generated_rules_plot(raw_fix_params_no_woc)  # Fig 6.15
    elif section == '6.6.2.2':
        create_generated_rules_plot(raw_dqn_rej_s2)  # Fig 6.23
        create_generated_rules_plot(raw_dqn_l_s2)  # Fig 6.24
    elif section == '6.6.2.4':
        create_generated_rules_plot(raw_dqn_rej_s3_bn)  # Fig 6.30
        create_generated_rules_plot(raw_dqn_l_s3_bn)  # Fig 6.31
