import random

from gyms.hhh.cpp.hhhmodule import SketchHHH as HHHAlgo
from ipaddress import IPv4Address
from matplotlib import pyplot as plt
import matplotlib as mpl
from gyms.hhh.flowgen.disttrace import DistributionTrace
from gyms.hhh.flowgen.traffic_traces import T3, T4
from gyms.hhh.images import generate_filter_image, generate_hhh_image

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


def test_hhh_image(hhh):
    return generate_hhh_image(hhh)


def test_filter_image(hhh):
    hhhs = hhh.query(0.3, 18)
    return generate_filter_image(hhhs)


filter_img = test_filter_image(hhh)
hhh_img = test_hhh_image(hhh)
mpl.rcParams['figure.dpi'] = 300
plt.imshow(filter_img, cmap='gist_gray', interpolation='nearest')  # , #cmap='viridis')
plt.show()

plt.imshow(hhh_img, cmap='gist_gray', interpolation='nearest')  # , #cmap='viridis')
plt.show()

# print('=== BLOCKLIST ===')
# for r in res:
#    padded_ip = str(IPv4Address(r.id)).rjust(15, ' ')
#    print(f'{padded_ip}/{r.len} {r.lo, r.hi}')
