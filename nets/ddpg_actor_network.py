# coding=utf-8
# Copyright 2018 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# NOTE: Copied from TF-Agents, but modified to optionally include BatchNorm layers.

"""Sample Actor network to use with DDPG agents.

Note: This network scales actions to fit the given spec by using `tanh`. Due to
the nature of the `tanh` function, actions near the spec bounds cannot be
returned.
"""

import gin
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.networks import network
from tf_agents.utils import common
from nets.dqn_encoding_network import EncodingNetwork


@gin.configurable
class ActorNetwork(network.Network):
    """Creates an actor network."""

    def __init__(self,
                 input_tensor_spec,
                 output_tensor_spec,
                 fc_layer_params=None,
                 dropout_layer_params=None,
                 conv_layer_params=None,
                 activation_fn=tf.keras.activations.relu,
                 preprocessing_layers=None,
                 preprocessing_combiner=None,
                 batch_norm=False,
                 name='ActorNetwork'):
        """Creates an instance of `ActorNetwork`.

        Args:
          input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing the
            inputs.
          output_tensor_spec: A nest of `tensor_spec.BoundedTensorSpec` representing
            the outputs.
          fc_layer_params: Optional list of fully_connected parameters, where each
            item is the number of units in the layer.
          dropout_layer_params: Optional list of dropout layer parameters, each item
            is the fraction of input units to drop or a dictionary of parameters
            according to the keras.Dropout documentation. The additional parameter
            `permanent', if set to True, allows to apply dropout at inference for
            approximated Bayesian inference. The dropout layers are interleaved with
            the fully connected layers; there is a dropout layer after each fully
            connected layer, except if the entry in the list is None. This list must
            have the same length of fc_layer_params, or be None.
          conv_layer_params: Optional list of convolution layers parameters, where
            each item is a length-three tuple indicating (filters, kernel_size,
            stride).
          activation_fn: Activation function, e.g. tf.nn.relu, slim.leaky_relu, ...
          name: A string representing name of the network.

        Raises:
          ValueError: If `input_tensor_spec` or `action_spec` contains more than one
            item, or if the action data type is not `float`.
        """

        super(ActorNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name=name)

        self._action_spec = tf.nest.flatten(output_tensor_spec)

        self._encoder = EncodingNetwork(
            input_tensor_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params,
            dropout_layer_params=dropout_layer_params,
            activation_fn=activation_fn,
            batch_normalization=batch_norm,
            kernel_initializer=tf.compat.v1.keras.initializers.glorot_uniform(),
            batch_squash=True,
            dtype=tf.float32)

        self._last_layer = tf.keras.layers.Dense(
            self._action_spec[0].shape[0],
            activation=tf.keras.activations.tanh,
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-0.003, maxval=0.003),
            name='custom-actions')

        self._output_tensor_spec = output_tensor_spec

    def call(self, observations, step_type=(), network_state=(), training=False):
        state, network_state = self._encoder(
            observations,
            step_type=step_type,
            network_state=network_state,
            training=training)

        state = self._last_layer(state, training=training)
        actions = common.scale_to_spec(state, self._action_spec[0])
        output_actions = tf.nest.pack_sequence_as(self._output_tensor_spec,
                                                  [actions])

        return output_actions, network_state
