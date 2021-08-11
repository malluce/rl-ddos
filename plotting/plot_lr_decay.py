import matplotlib.pyplot as plt
import numpy as np


def linear_decay(x, initial_lr, end_lr, steps):
    step = min(x, steps)
    return ((initial_lr - end_lr) * (1 - step / steps)) + end_lr


def exp_decay(x, initial_lr, rate, steps):
    return max(initial_lr * rate ** (x / steps), 1e-5)


x = np.arange(0, 1000000, step=1000)

LIN_INIT_LR = 1e-3
LIN_END_LR = 1e-5
LIN_STEPS = 50000
lin = list(map(lambda x: linear_decay(x, LIN_INIT_LR, LIN_END_LR, LIN_STEPS), x))
EXP_INIT_LR = 5e-4
EXP_RATE = 0.9
EXP_STEPS = 5000
exp = list(map(lambda x: exp_decay(x, EXP_INIT_LR, EXP_RATE, EXP_STEPS), x))

plt.plot(x, lin, label='lin')
plt.plot(x, exp, label='exp')
plt.yticks([1e-5, 1e-4, 1e-3])
plt.legend()
plt.show()
