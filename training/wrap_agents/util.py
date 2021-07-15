from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import ExponentialDecay
from tensorflow.keras.optimizers import Adam


def get_optimizer(lr, lr_decay_rate, lr_decay_steps):
    if lr_decay_rate is not None:
        return Adam(ExponentialDecay(lr, lr_decay_steps, lr_decay_rate, staircase=True))
    else:
        return Adam(lr)
