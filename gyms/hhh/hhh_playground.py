import time

from gyms.hhh.cpp.hhhmodule import SketchHHH as HHHAlgo
from ipaddress import IPv4Address

hhh = HHHAlgo(0.0001)
# hhh.update(0xFFFFFFFF, 5)
# hhh.update(0x00000000, 10)
# hhh.update(0x00000001, 1)
# hhh.update(0xFFFFFFFF, 20)
# hhh.update(0x00000001, 3)
# hhh.update(0x00000002, 2)
# hhh.update(0x00000002, 2)
# hhh.update(0xF0000000, 10)
# hhh.update(0xA0000000, 15)
hhh.update(0x7ff, 25)
hhh.update(0x800, 50)
hhh.update(0x900, 1)
for i in range(2 ** 32 - 500, 2 ** 32):
    hhh.update(i, 1)

for i in range(2 ** 31 - 500, 2 ** 31):
    hhh.update(i, 1)

res = hhh.query_all()

import numpy as np

IMAGE_SIZE = 128
max_addr = 2 ** 32

start = time.time()

# output as array
res = np.asarray(res)

# unique hierarchy levels (currently [0..32])
unique_indices = np.unique(res[:, 0], return_index=True)[1]

# 33 lists in ascending order of hierarchy levels (start with 0, end with 32)
# each list has shape num_items(level) X 2; columns are IP, count
split_by_level = np.split(res[:, 1:], np.sort(unique_indices)[1:])[::-1]

# the bounds of IMAGE_SIZE many bins to separate the address space
bounds = np.linspace(0, max_addr, num=IMAGE_SIZE, dtype=np.int)

# the output image
image = np.zeros((len(split_by_level), IMAGE_SIZE), dtype=np.int)

# iterate over all hierarchy levels and the corresponding item lists
for level, l in enumerate(split_by_level):
    # bin the item lists according to the IP address
    l_binned = np.digitize(l[:, 0], bins=bounds)

    # increment the image pixels for each level and bin according the item list's count
    for bin_index, x in enumerate(l[:, 1]):
        image[level, l_binned[bin_index]] += x

print(f'image build time={time.time() - start}')

from matplotlib import pyplot as plt

print(image[32])

# image = np.repeat(image, 4, axis=0)

plt.imshow(image, interpolation='nearest')  # , #cmap='gray', vmin=0, vmax=255)
plt.show()

# for r in res:
#    print(r)
# print('=== BLOCKLIST ===')
# for r in res:
#    padded_ip = str(IPv4Address(r.id)).rjust(15, ' ')
#    print(f'{padded_ip}/{r.len} {r.lo, r.hi}')
