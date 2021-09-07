import random
import sys

import numpy as np
from PIL import Image
from gyms.hhh.cpp.hhhmodule import SketchHHH as HHHAlgo
from ipaddress import IPv4Address
from matplotlib import colors, pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from matplotlib.axes import Axes
import tensorflow as tf
from matplotlib.figure import Figure

from gyms.hhh.flowgen.disttrace import DistributionTrace
from gyms.hhh.flowgen.traffic_traces import T3, T4, THauke
from gyms.hhh.images import ImageGenerator

hhh = HHHAlgo(0.0001)
# t = DistributionTrace(traffic_trace=THauke(benign_flows=500, attack_flows=1000, maxtime=600, maxaddr=0xffff))
t = DistributionTrace(traffic_trace=T3(num_benign=300, num_attack=150, maxtime=600, maxaddr=0xffff))
t.rewind()
fin_cnt = 0
for packet, step_finished in t:
    if step_finished:
        fin_cnt += 1
    if fin_cnt == 150:
        break
    if fin_cnt >= 0:
        hhh.update(packet.ip, 100)


# for i in range(0, 5):
#    hhh.update(random.randint(0, 0xffff), 2)
#    hhh.update(random.randint(0, 0xffff), 200)


def test_hhh_image(hhh, img_gen):
    return img_gen.generate_hhh_image(hhh)


def test_filter_image(hhh, img_gen):
    class HHHMock:
        def __init__(self):
            self.id = 0x0000
            self.len = 17

    hhhs = hhh.query(0.05, 21)
    for hhh in hhhs:
        print(hhh.id, hhh.len)

    # hhhs = [HHHMock()]
    return img_gen.generate_filter_image(hhhs)


def show_image(image, cmap, show_border=False, show_hist=False):
    if not show_border:
        fig = plt.figure()
        fig.set_size_inches(image.shape[1] / image.shape[0], 1, forward=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
    else:
        fig, ax = plt.subplots()
    ax.imshow(image, cmap=cmap, interpolation='nearest')
    plt.show()
    if show_hist:
        plt.hist(image.ravel(), bins=128, range=(0.0, 1.0), fc='k', ec='k')
        plt.show()


def gen_and_show_images(hhh, image_gen):
    filter_img = test_filter_image(hhh, image_gen)
    show_image(filter_img, cmap='binary', show_border=False)

    hhh_img = test_hhh_image(hhh, image_gen)
    show_image(hhh_img, cmap='inferno', show_border=False)

    # print(f'hhh shape={image_gen.get_hhh_img_spec()}')
    # print(f'filter shape={image_gen.get_filter_img_spec()}')


mpl.rcParams['figure.dpi'] = 300

img_gen = ImageGenerator(hhh_squash_threshold=1, img_width_px=64)
img_gen2 = ImageGenerator(hhh_squash_threshold=1, img_width_px=128)
img_gen3 = ImageGenerator(hhh_squash_threshold=1, img_width_px=256)
img_gen4 = ImageGenerator(hhh_squash_threshold=1, img_width_px=512)

gen_and_show_images(hhh, img_gen)
gen_and_show_images(hhh, img_gen2)
gen_and_show_images(hhh, img_gen3)
gen_and_show_images(hhh, img_gen4)

# cnn = tf.keras.models.Sequential([
#    # 17,512,1
#    tf.keras.layers.Conv2D(16, (2, 4), activation=tf.keras.activations.relu, strides=(1, 2)),
#    # 16,255,16
#    tf.keras.layers.MaxPool2D(pool_size=(1, 2), strides=(1, 2)),
#    # 16,127,16
#    tf.keras.layers.Conv2D(32, (2, 4), activation=tf.keras.activations.relu, strides=(1, 2)),
#    # 15,62,32
#    tf.keras.layers.MaxPool2D(pool_size=(1, 2), strides=(1, 2)),
#    # 15,31,32
#    tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.relu, strides=3),
#    # 5,10,64
#    tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
#    # 2,4,64
#    tf.keras.layers.Flatten(),
#    # 1,512
#    tf.keras.layers.Dense(64, activation=tf.keras.activations.relu)
# ])
# cnn.build(input_shape=np.expand_dims(hhh_img, 0).shape)
# cnn.summary()


# for c in range(0, 8):
#    cnn_processed = np.squeeze(cnn(np.expand_dims(hhh_img, 0)), 0)
#    print(f'processed={cnn_processed.shape}')
#    show_image(cnn_processed[:, :, c], cmap='inferno')

# print('=== BLOCKLIST ===')
# for r in res:
#    padded_ip = str(IPv4Address(r.id)).rjust(15, ' ')
#    print(f'{padded_ip}/{r.len} {r.lo, r.hi}')

sys.exit(0)
