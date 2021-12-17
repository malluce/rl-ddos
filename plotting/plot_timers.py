import matplotlib.pyplot as plt
import numpy as np

from gyms.hhh.loop import TimedWorstOffenderCache
from absl import logging

logging.set_verbosity('debug')


def simulate_timers(rejections, num_steps):
    cache = TimedWorstOffenderCache(background_decrement=1)
    current_timers = []
    background_timers = []

    cache.add(0, 1, 1)
    current_timers.append(list(cache.active_timers.items())[0][1])
    background_timers.append(list(cache.background_timers.items())[0][1])

    rejections_handled = 0
    since_recovery = 0

    for i in range(num_steps):
        print('tick')

        prev = list(cache.active_timers.items())[0][1] if len(list(cache.active_timers.items())) > 0 else 0
        cache.rule_recovery()
        recovery_event = len(list(cache.active_timers.items())) == 0 and prev != 0
        if recovery_event:
            since_recovery = 0

        if len(list(cache.active_timers.items())) > 0:
            print(f'storing {list(cache.active_timers.items())[0][1]}')
            current_timers.append(list(cache.active_timers.items())[0][1])
        else:
            print(f'storing 0')
            current_timers.append(0)

        if len(list(cache.background_timers.items())) > 0:
            background_timers.append(list(cache.background_timers.items())[0][1])
        else:
            background_timers.append(0)

        if len(list(cache.active_timers.items())) == 0:
            if rejections_handled < len(rejections) and rejections[rejections_handled] == since_recovery:
                print('rejecting')
                cache.add(0, 1, 1)
                rejections_handled += 1

        since_recovery += 1

    return current_timers, background_timers


NUM_STEPS = 100

rejections = [0, 4, 30]
current2, bg2 = simulate_timers(rejections, num_steps=NUM_STEPS)
print(bg2)
x = range(NUM_STEPS + 1)

plt.plot(x, current2, label='timer')
plt.plot(x, bg2, label='background timer', linestyle='--')
plt.legend()
plt.xlabel('time index')
plt.ylabel('timer value')
plt.show()
