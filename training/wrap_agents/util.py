import tensorflow
from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import ExponentialDecay, LearningRateSchedule, \
    PolynomialDecay
from tensorflow.keras.optimizers import Adam


def get_optimizer(lr, lr_decay_rate, lr_decay_steps, exp_min_lr=None, linear_decay_end_lr=None,
                  linear_decay_steps=None):
    if lr_decay_rate is not None and linear_decay_end_lr is not None:
        raise ValueError('Either use linear (linear_decay_end_lr, linear_decay_steps) or '
                         'exponential decay (lr_decay_rate, lr_decay_steps), not both.')

    if lr_decay_rate is not None:
        if exp_min_lr is not None:
            return Adam(MinExpSchedule(lr, lr_decay_steps, lr_decay_rate, exp_min_lr))
        else:
            return Adam(ExponentialDecay(lr, lr_decay_steps, lr_decay_rate, staircase=False))
    elif linear_decay_end_lr is not None:
        return Adam(PolynomialDecay(lr, linear_decay_steps, linear_decay_end_lr))
    else:
        return Adam(lr)


class MinExpSchedule(LearningRateSchedule):

    def __init__(self, lr, lr_decay_steps, lr_decay_rate, min_lr):
        self.min_lr = min_lr
        self.exp_schedule = ExponentialDecay(lr, lr_decay_steps, lr_decay_rate, staircase=False)

    def __call__(self, step):
        return tensorflow.maximum(self.min_lr, self.exp_schedule(step))

    def get_config(self):
        pass
