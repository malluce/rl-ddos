import sys
from collections import Callable, OrderedDict

import gin

import tensorflow as tf
from absl import logging
from tensorflow.keras.layers import Lambda

from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import ExponentialDecay, LearningRateSchedule, \
    PolynomialDecay
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.python.keras.utils.vis_utils import plot_model


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
        # return SGD(lr) # TODO add SGD as option?


@gin.configurable
class MinExpSchedule(LearningRateSchedule):

    def __init__(self, lr, lr_decay_steps, lr_decay_rate, min_lr):
        self.min_lr = min_lr
        self.exp_schedule = ExponentialDecay(lr, lr_decay_steps, lr_decay_rate, staircase=False)

    def __call__(self, step=None):
        if step is None:
            step = tf.compat.v1.train.get_or_create_global_step()
        return tf.maximum(self.min_lr, self.exp_schedule(step))

    def get_config(self):
        pass


def build_cnn_from_spec(cnn_spec, act_func, batch_norm=False):
    """
    Creates a CNN from a given specification.
    Conv2D and pooling layers alternate (first pool layer at start_pool_idx), with one fully connected layer at the end.

    :param batch_norm: if to include batch norm layers after each layer
    :param cnn_spec: the specification of CNN are 3-tuples of the following contents.
    (
      ([conv_filters], [conv_kernel_sizes], [conv_strides]),
      ([pool_sizes], [pool_strides]),
      [fc_units],
      start_pool_idx)
    :param act_func: the activation function to use for the CNN
    :return: the keras model of the CNN
    """
    conv_layers, pool_layers, fc_units, start_pool_idx = cnn_spec
    conv_filters, conv_kernel_sizes, conv_strides = conv_layers
    pool_sizes, pool_strides = pool_layers
    keras_layers = []

    assert len(conv_filters) == len(conv_kernel_sizes) == len(conv_strides)
    assert len(pool_sizes) == len(pool_strides)

    total_layers = 0
    pool_idx = 0
    conv_idx = 0
    for i in range(len(conv_filters)):
        keras_layers.append(
            tf.keras.layers.Conv2D(conv_filters[conv_idx], conv_kernel_sizes[conv_idx], activation=act_func,
                                   strides=conv_strides[conv_idx])
        )
        if batch_norm:
            keras_layers.append(tf.keras.layers.BatchNormalization())
        conv_idx += 1
        total_layers += 1
        if total_layers >= start_pool_idx:
            keras_layers.append(
                tf.keras.layers.MaxPool2D(pool_size=pool_sizes[pool_idx], strides=pool_strides[pool_idx]))
            if batch_norm:
                keras_layers.append(tf.keras.layers.BatchNormalization())
            pool_idx += 1
            total_layers += 1

    # flatten and dense at the end
    keras_layers.append(tf.keras.layers.Flatten())
    for fc in fc_units:
        keras_layers.append(tf.keras.layers.Dense(fc, activation=act_func))
        if batch_norm:
            keras_layers.append(tf.keras.layers.BatchNormalization())
    return tf.keras.models.Sequential(keras_layers)


def get_preprocessing_cnn(cnn_spec, time_step_spec, act_func, batch_norm):
    """
    Returns a preprocessing CNN (layers and combiner), if time_step_spec contains images as observations.
    :param batch_norm: whether to use batch norm layers in between
    :param cnn_spec: the spec of CNN to build, see doc of build_cnn_from_spec()
    :param time_step_spec: the time step spec
    :param act_func: the activation function to use in the preprocessing layers
    :return: (preprocessing layers, combiner)
    """
    obs_spec = time_step_spec().observation if callable(time_step_spec) else time_step_spec.observation
    if type(obs_spec) is OrderedDict and 'hhh_image' in obs_spec:  # 'image' in obs_spec TODO
        logging.info('setting up CNN!')

        cnn = build_cnn_from_spec(cnn_spec, act_func, batch_norm)
        hhh_cnn = build_cnn_from_spec(cnn_spec, act_func, batch_norm)

        preprocessing_layers = {
            'vector': Lambda(lambda x: x),  # pass-through layer
            # 'image': cnn,
            'hhh_image': hhh_cnn
        }
        preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)
    else:
        logging.info('not using CNN')
        preprocessing_layers = None
        preprocessing_combiner = None
    return preprocessing_combiner, preprocessing_layers
