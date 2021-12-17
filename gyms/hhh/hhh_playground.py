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
t = DistributionTrace(traffic_trace_construct=lambda is_eval: NTPTrace(is_eval=is_eval), is_eval=True)


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
        plt.hist(image.ravel(), bins=128, range=(0.0, 1.0))  # , fc='country', ec='country')
        plt.show()


def gen_and_show_images(hhh, image_gen):
    filter_img = run_filter_image(hhh, image_gen)
    # show_image(filter_img, cmap='binary', show_border=False)

    hhh_img = image_gen.generate_hhh_image(hhh, crop=image_gen.crop_standalone_hhh_image)
    print(sorted(np.unique(hhh_img.flatten()), reverse=True))
    show_image(hhh_img, cmap='gray', show_border=False)

    # print(f'hhh shape={image_gen.get_hhh_img_spec()}')
    # print(f'filter shape={image_gen.get_filter_img_spec()}')

    return filter_img, hhh_img


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
    # img_gen = ImageGenerator(hhh_squash_threshold=1, img_width_px=64)
    # img_gen2 = ImageGenerator(hhh_squash_threshold=1, img_width_px=128)
    # img_gen3 = ImageGenerator(hhh_squash_threshold=1, img_width_px=256, max_pixel_value=1.0)
    # img_gen4 = ImageGenerator(hhh_squash_threshold=1, img_width_px=512)
    # gen_and_show_images(hhh, img_gen3)
    img_gen3 = ImageGenerator(hhh_squash_threshold=1, img_width_px=256, max_pixel_value=1.0,
                              crop_standalone_hhh_image=True)
    # img_gen4 = ImageGenerator(hhh_squash_threshold=1, img_width_px=512)
    _, img = gen_and_show_images(hhh, img_gen3)

    img_gen3 = ImageGenerator(hhh_squash_threshold=-1, img_width_px=256, max_pixel_value=1.0,
                              crop_standalone_hhh_image=True)
    # img_gen4 = ImageGenerator(hhh_squash_threshold=1, img_width_px=512)
    _, img = gen_and_show_images(hhh, img_gen3)
    t.rewind()
    hhh.clear()
# gen_and_show_images(hhh, img_gen)
# gen_and_show_images(hhh, img_gen2)
# gen_and_show_images(hhh, img_gen3)
# filter_img, hhh_img = gen_and_show_images(hhh, img_gen4)
# combined_img = img_gen4.generate_image(hhh, hhh.query(PHI, L))
# combined_img = img_gen3.generate_image(hhh, hhh.query(PHI, L))
# print(combined_img.shape)
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

import visualkeras

cnn = build_cnn_from_spec(cnn_256_cropped, tf.keras.activations.relu)
cnn.build(input_shape=np.expand_dims(img, 0).shape)
cnn.summary()
visualkeras.layered_view(cnn, to_file='model_vis.png', legend=True, spacing=10)
# plot_model(cnn, show_shapes=False, show_layer_names=False, rankdir='LR', dpi=200)
# for c in range(0, 8):
#    cnn_processed = np.squeeze(cnn(np.expand_dims(hhh_img, 0)), 0)
#    print(f'processed={cnn_processed.shape}')
#    show_image(cnn_processed[:, :, c], cmap='inferno')

# print('=== BLOCKLIST ===')
# for r in res:
#    padded_ip = str(IPv4Address(r.id)).rjust(15, ' ')
#    print(f'{padded_ip}/{r.len} {r.lo, r.hi}')
sys.exit(0)
