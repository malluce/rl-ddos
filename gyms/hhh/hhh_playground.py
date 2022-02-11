import sys

import numpy as np
from gyms.hhh.cpp.hhhmodule import SketchHHH as HHHAlgo
from matplotlib import pyplot as plt
import matplotlib as mpl
import tensorflow as tf

from gyms.hhh.flowgen.disttrace import DistributionTrace
from gyms.hhh.flowgen.traffic_traces import S1, S2, S3
from gyms.hhh.images import ImageGenerator
from training.wrap_agents.util import build_cnn_from_spec


# This script was used to visualize the computed images of undiscounted HHH frequencies
# and summarize pre-processing CNN. Not integral part of the simulation, just used for testing and during design phase.

def show_image(image, cmap, show_border=False, show_hist=False):
    if image.shape[-1] != 1:
        vmin = np.min(image[:, :, [0, 1, 3, 4]])
        vmax = np.max(image[:, :, [0, 1, 3, 4]])
    for c in range(image.shape[-1]):  # all channels
        if not show_border:
            fig = plt.figure()
            fig.set_size_inches(image.shape[1] / image.shape[0], 1, forward=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
        else:
            fig, ax = plt.subplots()
        if image.shape[-1] != 1:
            if c in [0, 1, 3, 4]:
                ax.imshow(image[:, :, c], cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax)
            else:
                ax.imshow(image[:, :, c], cmap=cmap, interpolation='nearest')
        else:
            ax.imshow(image[:, :, c], cmap=cmap, interpolation='nearest')
        plt.show()
        if show_hist:
            plt.hist(image.ravel(), bins=128, range=(0.0, 1.0))  # , fc='country', ec='country')
            plt.show()


def gen_and_show_images(hhh, image_gen):
    hhh_img = image_gen.generate_image(hhh, None)
    show_image(hhh_img, cmap='gray', show_border=False)

    return hhh_img


if __name__ == '__main__':
    hhh = HHHAlgo(0.0001)
    STEP_START = 50
    STEPS_STOP = 59
    TRACE = S1
    t = DistributionTrace(traffic_trace_construct=lambda is_eval: TRACE(), is_eval=True)
    mpl.rcParams['figure.dpi'] = 300

    # show all channels of image, after updating HHH algo with adaptation steps from STEP_START to STEPS_STOP] from t
    fin_cnt = 0
    for packet, step_finished in t:
        if step_finished:
            fin_cnt += 1
        if fin_cnt == STEPS_STOP:
            break
        if fin_cnt >= STEP_START:
            hhh.update(packet.ip, 1)
    img_gen = ImageGenerator(hhh_squash_threshold=1, img_width_px=128, crop_standalone_hhh_image=True, mode='multi')
    img = gen_and_show_images(hhh, img_gen)

    # summarize CNN architecture
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
    cnn = build_cnn_from_spec(cnn_128_multi, tf.keras.activations.relu)
    cnn.build(input_shape=np.expand_dims(img, 0).shape)
    cnn.summary()
    sys.exit(0)
