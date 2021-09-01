import random

from gyms.hhh.cpp.hhhmodule import SketchHHH as HHHAlgo
from ipaddress import IPv4Address
from matplotlib import pyplot as plt
import matplotlib as mpl
from gyms.hhh.flowgen.disttrace import DistributionTrace
from gyms.hhh.flowgen.traffic_traces import T3, T4
from gyms.hhh.images import generate_hhh_image

hhh = HHHAlgo(0.0001)
t = DistributionTrace(traffic_trace=T4(maxaddr=0xffff))
t.rewind()
fin_cnt = 0
for packet, step_finished in t:
    if step_finished:
        fin_cnt += 1
    if fin_cnt == 600:
        break
    if fin_cnt >= 300:
        hhh.update(packet.ip, 100)

# for _ in range(1000):
#    hhh.update(int(random.gauss(0x8000, 0.4)), 1)

# for _ in range(1000):
#    hhh.update(random.randint(0, 0x8000), 1)

image = generate_hhh_image(hhh)

mpl.rcParams['figure.dpi'] = 300
plt.imshow(image, cmap='gist_gray', interpolation='nearest')  # , #cmap='viridis')
plt.show()

# print('=== BLOCKLIST ===')
# for r in res:
#    padded_ip = str(IPv4Address(r.id)).rjust(15, ' ')
#    print(f'{padded_ip}/{r.len} {r.lo, r.hi}')
