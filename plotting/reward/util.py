from gyms.hhh.reward import RewardCalc
from gyms.hhh.state import State


def get_reward(reward_calculator: RewardCalc, fpr, bl_size, precision, recall):
    dummy_state = State(None)
    dummy_state.fpr = fpr
    dummy_state.blacklist_size = bl_size
    dummy_state.precision = precision
    dummy_state.recall = recall
    return reward_calculator.calc_reward(dummy_state)
