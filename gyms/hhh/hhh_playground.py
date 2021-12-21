import random
import sys
from collections import defaultdict

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
from gyms.hhh.flowgen.traffic_traces import BotTrace, BotnetSourcePattern, NTPTrace, SSDPTrace, T2, T3, T4, THauke, \
    TRandomPatternSwitch, \
    UniformRandomSourcePattern
from gyms.hhh.images import ImageGenerator
from gyms.hhh.label import Label
from plotting.plot_traces import plot_botnet_pattern
from training.wrap_agents.util import build_cnn_from_spec
from tensorflow.python.keras.utils.vis_utils import plot_model

hhh = HHHAlgo(0.0001)
PHI = 0.01
L = 17
t = DistributionTrace(traffic_trace_construct=lambda is_eval: SSDPTrace(is_eval=is_eval,benign_flows=10000), is_eval=True)


# atk_addr, benign_addr = plot_botnet_pattern()
# for addr in np.concatenate((atk_addr, benign_addr)):
#    hhh.update(addr, 1)

# hhh.update(0x0, 10)
# hhh.update(0x1, 5)
# hhh.update(0x2, 5)


def run_filter_image(hhh, img_gen):
    class HHHMock:
        def __init__(self):
            self.id = 0x0000
            self.len = 17

    hhhs = hhh.query(PHI, L)

    # hhhs = [HHHMock()]
    return img_gen.generate_filter_image(hhhs)


def show_image(image, cmap, show_border=False, show_hist=False):
    print(image.shape)
    if image.shape[-1] != 1:
        vmin = np.min(image[:,:,[0,1,3,4]])
        vmax = np.max(image[:,:,[0,1,3,4]])
    for c in range(image.shape[-1]): # all channels
        print(c)
        if not show_border:
            fig = plt.figure()
            fig.set_size_inches(image.shape[1] / image.shape[0], 1, forward=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
        else:
            fig, ax = plt.subplots()
        if image.shape[-1] != 1:
            if c in [0,1,3,4]:
                ax.imshow(image[:, :, c], cmap=cmap, interpolation='nearest',vmin=vmin,vmax=vmax)
            else:
                ax.imshow(image[:,:,c], cmap=cmap, interpolation='nearest')
        else:
            ax.imshow(image[:, :, c], cmap=cmap, interpolation='nearest')
        plt.show()
        if show_hist:
            plt.hist(image.ravel(), bins=128, range=(0.0, 1.0))  # , fc='country', ec='country')
            plt.show()


def gen_and_show_images(hhh, image_gen):
    hhh_img = image_gen.generate_image(hhh,None)
    for c in range(hhh_img.shape[-1]):
        print(np.mean(hhh_img[:,:,c]), np.std(hhh_img[:,:,c]))
    show_image(hhh_img, cmap='gray', show_border=False)

    # print(f'hhh shape={image_gen.get_hhh_img_spec()}')
    # print(f'filter shape={image_gen.get_filter_img_spec()}')

    return hhh_img


mpl.rcParams['figure.dpi'] = 300

for _ in range(1):
    print(t.traffic_trace.get_source_pattern_id(9))
    fin_cnt = 0
    for packet, step_finished in t:
        if step_finished:
            fin_cnt += 1
        if fin_cnt == 100:
            break
        if fin_cnt >= 90:
            hhh.update(packet.ip, 100)
    img_gen3 = ImageGenerator(hhh_squash_threshold=-1, img_width_px=128, max_pixel_value=1.0,
                              crop_standalone_hhh_image=True,mode='multi')
    img = gen_and_show_images(hhh, img_gen3)
    t.rewind()
    hhh.clear()
# cnn = tf.keras.models.Sequential([
#    # 17,256,2
#    tf.keras.layers.Conv2D(8, (2, 4), activation=tf.keras.activations.relu, strides=(1, 2)),
#    # 16,255,16
#    tf.keras.layers.MaxPool2D(pool_size=(1, 2), strides=(1, 2)),
#    # 16,127,16
#    tf.keras.layers.Conv2D(16, (2, 4), activation=tf.keras.activations.relu, strides=(1, 1)),
#    # 15,62,32
#    tf.keras.layers.MaxPool2D(pool_size=(1, 2), strides=(1, 2)),
#    # 15,31,32
#    tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.relu, strides=3),
#    # 5,10,64
#    tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
#    # 2,4,64
#    tf.keras.layers.Flatten(),
#    # 1,512
#    tf.keras.layers.Dense(64, activation=tf.keras.activations.relu)
#    # 1,64
# ])
# cnn.build(input_shape=np.expand_dims(combined_img, 0).shape)
# cnn.summary()

cnn_256 = (
    (
        [8, 16, 32],  # conv filters
        [(2, 4), (2, 4), (3, 3)],  # conv kernel sizes
        [(1, 2), (1, 1), 3]  # conv strides
    ),
    (
        [(1, 2), (1, 2), 3],  # pool sizes
        [(1, 2), (1, 2), 2]  # pool strides
    ),
    [64],  # fc units after flatten
    1  # start pooling layers at idx 1
)

cnn_256_cropped = ((
                       [8, 16, 32],  # conv filters
                       [(2, 4), (2, 4), (1, 2)],  # conv kernel sizes
                       [(1, 2), (1, 2), (1, 2)]  # conv strides
                   ),
                   (
                       [(1, 2), (1, 2), (2, 2)],  # pool sizes
                       [(1, 2), (1, 2), 2]  # pool strides
                   ),
                   [64, 64],  # fc units after flatten
                   1
)

cnn_128_multi = ((
                       [8, 16, 32],  # conv filters
                       [(2, 3), (2, 3), (2, 2)],  # conv kernel sizes
                       [(1, 2), (1, 2), 1]  # conv strides
                   ),
                   (
                       [(1, 2), (2, 2), (2, 2)],  # pool sizes
                       [(1, 2), 2, 2]  # pool strides
                   ),
                   [64, 64],  # fc units after flatten
                   1
)

print(img.shape)
cnn = build_cnn_from_spec(cnn_128_multi, tf.keras.activations.relu)
print(np.expand_dims(img, 0).shape)
cnn.build(input_shape=np.expand_dims(img, 0).shape)
cnn.summary()
sys.exit(0)
