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

"""Sample Keras networks for DQN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tensorflow.keras import layers
from tf_agents.keras_layers import dynamic_unroll_layer
from tf_agents.networks import network
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step
from tf_agents.utils import nest_utils

from nets import dqn_encoding_network as encoding_network


def validate_specs(action_spec, observation_spec):
    """Validates the spec contains a single action."""
    del observation_spec  # not currently validated

    flat_action_spec = tf.nest.flatten(action_spec)
    if len(flat_action_spec) > 1:
        raise ValueError('Network only supports action_specs with a single action.')

    if flat_action_spec[0].shape not in [(), (1,)]:
        raise ValueError(
            'Network only supports action_specs with shape in [(), (1,)])')


@gin.configurable
class QNetwork(network.Network):
    """Feed Forward network."""

    def __init__(self,
                 input_tensor_spec,
                 action_spec,
                 preprocessing_layers=None,
                 preprocessing_combiner=None,
                 conv_layer_params=None,
                 fc_layer_params=(75, 40),
                 dropout_layer_params=None,
                 activation_fn=tf.keras.activations.relu,
                 kernel_initializer=None,
                 batch_squash=True,
                 batch_normalization=False,  # added by hauke
                 dtype=tf.float32,
                 name='QNetwork'):
        """Creates an instance of `QNetwork`.

        Args:
          input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing the
            input observations.
          action_spec: A nest of `tensor_spec.BoundedTensorSpec` representing the
            actions.
          preprocessing_layers: (Optional.) A nest of `tf.keras.layers.Layer`
            representing preprocessing for the different observations.
            All of these layers must not be already built. For more details see
            the documentation of `networks.EncodingNetwork`.
          preprocessing_combiner: (Optional.) A keras layer that takes a flat list
            of tensors and combines them. Good options include
            `tf.keras.layers.Add` and `tf.keras.layers.Concatenate(axis=-1)`.
            This layer must not be already built. For more details see
            the documentation of `networks.EncodingNetwork`.
          conv_layer_params: Optional list of convolution layers parameters, where
            each item is a length-three tuple indicating (filters, kernel_size,
            stride).
          fc_layer_params: Optional list of fully_connected parameters, where each
            item is the number of units in the layer.
          dropout_layer_params: Optional list of dropout layer parameters, where
            each item is the fraction of input units to drop. The dropout layers are
            interleaved with the fully connected layers; there is a dropout layer
            after each fully connected layer, except if the entry in the list is
            None. This list must have the same length of fc_layer_params, or be
            None.
          activation_fn: Activation function, e.g. tf.keras.activations.relu.
          kernel_initializer: Initializer to use for the kernels of the conv and
            dense layers. If none is provided a default variance_scaling_initializer
          batch_squash: If True the outer_ranks of the observation are squashed into
            the batch dimension. This allow encoding networks to be used with
            observations with shape [BxTx...].
          dtype: The dtype to use by the convolution and fully connected layers.
          name: A string representing the name of the network.

        Raises:
          ValueError: If `input_tensor_spec` contains more than one observation. Or
            if `action_spec` contains more than one action.
        """
        validate_specs(action_spec, input_tensor_spec)
        action_spec = tf.nest.flatten(action_spec)[0]
        num_actions = action_spec.maximum - action_spec.minimum + 1
        encoder_input_tensor_spec = input_tensor_spec

        encoder = encoding_network.EncodingNetwork(
            encoder_input_tensor_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params,
            dropout_layer_params=dropout_layer_params,
            activation_fn=activation_fn,
            kernel_initializer=kernel_initializer,
            batch_squash=batch_squash,
            batch_normalization=batch_normalization,
            dtype=dtype)

        q_value_layer = tf.keras.layers.Dense(
            num_actions,
            activation=None,
            kernel_initializer=tf.compat.v1.initializers.random_uniform(
                minval=-0.03, maxval=0.03),
            bias_initializer=tf.compat.v1.initializers.constant(-0.2),
            dtype=dtype)

        super(QNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name=name)

        self._encoder = encoder
        self._q_value_layer = q_value_layer

    def call(self, observation, step_type=None, network_state=(), training=False):
        """Runs the given observation through the network.

        Args:
          observation: The observation to provide to the network.
          step_type: The step type for the given observation. See `StepType` in
            time_step.py.
          network_state: A state tuple to pass to the network, mainly used by RNNs.
          training: Whether the output is being used for training.

        Returns:
          A tuple `(logits, network_state)`.
        """
        state, network_state = self._encoder(
            observation, step_type=step_type, network_state=network_state,
            training=training)
        q_value = self._q_value_layer(state, training=training)
        return q_value, network_state


KERAS_LSTM_FUSED = 2


@gin.configurable
class LSTMEncodingNetwork(network.Network):
    """Recurrent network."""

    def __init__(
            self,
            input_tensor_spec,
            preprocessing_layers=None,
            preprocessing_combiner=None,
            conv_layer_params=None,
            input_fc_layer_params=(75, 40),
            lstm_size=None,
            output_fc_layer_params=(75, 40),
            activation_fn=tf.keras.activations.relu,
            rnn_construction_fn=None,
            rnn_construction_kwargs=None,
            dtype=tf.float32,
            name='LSTMEncodingNetwork',
            batch_norm=False
    ):
        """Creates an instance of `LSTMEncodingNetwork`.

        Input preprocessing is possible via `preprocessing_layers` and
        `preprocessing_combiner` Layers.  If the `preprocessing_layers` nest is
        shallower than `input_tensor_spec`, then the layers will get the subnests.
        For example, if:

        ```python
        input_tensor_spec = ([TensorSpec(3)] * 2, [TensorSpec(3)] * 5)
        preprocessing_layers = (Layer1(), Layer2())
        ```

        then preprocessing will call:

        ```python
        preprocessed = [preprocessing_layers[0](observations[0]),
                        preprocessing_layers[1](observations[1])]
        ```

        However if

        ```python
        preprocessing_layers = ([Layer1() for _ in range(2)],
                                [Layer2() for _ in range(5)])
        ```

        then preprocessing will call:
        ```python
        preprocessed = [
          layer(obs) for layer, obs in zip(flatten(preprocessing_layers),
                                           flatten(observations))
        ]
        ```

        Args:
          input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing the
            observations.
          preprocessing_layers: (Optional.) A nest of `tf.keras.layers.Layer`
            representing preprocessing for the different observations. All of these
            layers must not be already built.
          preprocessing_combiner: (Optional.) A keras layer that takes a flat list
            of tensors and combines them.  Good options include
            `tf.keras.layers.Add` and `tf.keras.layers.Concatenate(axis=-1)`. This
            layer must not be already built.
          conv_layer_params: Optional list of convolution layers parameters, where
            each item is a length-three tuple indicating (filters, kernel_size,
            stride).
          input_fc_layer_params: Optional list of fully connected parameters, where
            each item is the number of units in the layer. These feed into the
            recurrent layer.
          lstm_size: An iterable of ints specifying the LSTM cell sizes to use.
          output_fc_layer_params: Optional list of fully connected parameters, where
            each item is the number of units in the layer. These are applied on top
            of the recurrent layer.
          activation_fn: Activation function, e.g. tf.keras.activations.relu,.
          rnn_construction_fn: (Optional.) Alternate RNN construction function, e.g.
            tf.keras.layers.LSTM, tf.keras.layers.CuDNNLSTM. It is invalid to
            provide both rnn_construction_fn and lstm_size.
          rnn_construction_kwargs: (Optional.) Dictionary or arguments to pass to
            rnn_construction_fn.

            The RNN will be constructed via:

            ```
            rnn_layer = rnn_construction_fn(**rnn_construction_kwargs)
            ```
          dtype: The dtype to use by the convolution, LSTM, and fully connected
            layers.
          name: A string representing name of the network.

        Raises:
          ValueError: If any of `preprocessing_layers` is already built.
          ValueError: If `preprocessing_combiner` is already built.
          ValueError: If neither `lstm_size` nor `rnn_construction_fn` are provided.
          ValueError: If both `lstm_size` and `rnn_construction_fn` are provided.
        """
        if lstm_size is None and rnn_construction_fn is None:
            raise ValueError('Need to provide either custom rnn_construction_fn or '
                             'lstm_size.')
        if lstm_size and rnn_construction_fn:
            raise ValueError('Cannot provide both custom rnn_construction_fn and '
                             'lstm_size.')

        kernel_initializer = tf.compat.v1.variance_scaling_initializer(
            scale=2.0, mode='fan_in', distribution='truncated_normal')

        input_encoder = encoding_network.EncodingNetwork(
            input_tensor_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            conv_layer_params=conv_layer_params,
            fc_layer_params=input_fc_layer_params,
            activation_fn=activation_fn,
            kernel_initializer=kernel_initializer,
            dtype=dtype, batch_normalization=batch_norm)

        # Create RNN cell
        if rnn_construction_fn:
            rnn_construction_kwargs = rnn_construction_kwargs or {}
            lstm_network = rnn_construction_fn(**rnn_construction_kwargs)
        else:
            if len(lstm_size) == 1:
                cell = tf.keras.layers.LSTMCell(
                    lstm_size[0],
                    dtype=dtype,
                    implementation=KERAS_LSTM_FUSED)
            else:
                cell = tf.keras.layers.StackedRNNCells(
                    [tf.keras.layers.LSTMCell(size, dtype=dtype,
                                              implementation=KERAS_LSTM_FUSED)
                     for size in lstm_size])
            lstm_network = dynamic_unroll_layer.DynamicUnroll(cell)

        output_encoder = []
        if output_fc_layer_params:
            output_encoder.append(tf.keras.layers.BatchNormalization())
            for units in output_fc_layer_params:
                output_encoder.append(tf.keras.layers.Dense(
                    units,
                    activation=activation_fn,
                    kernel_initializer=kernel_initializer,
                    dtype=dtype))
                output_encoder.append(tf.keras.layers.BatchNormalization())

        counter = [-1]

        def create_spec(size):
            counter[0] += 1
            return tensor_spec.TensorSpec(
                size, dtype=dtype, name='network_state_%d' % counter[0])

        state_spec = tf.nest.map_structure(create_spec,
                                           lstm_network.cell.state_size)

        super(LSTMEncodingNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec, state_spec=state_spec, name=name)

        self._conv_layer_params = conv_layer_params
        self._input_encoder = input_encoder
        self._lstm_network = lstm_network
        self._output_encoder = output_encoder

    def call(self,
             observation,
             step_type,
             network_state=(),
             training=False):
        """Apply the network.

        Args:
          observation: A tuple of tensors matching `input_tensor_spec`.
          step_type: A tensor of `StepType.
          network_state: (optional.) The network state.
          training: Whether the output is being used for training.

        Returns:
          `(outputs, network_state)` - the network output and next network state.

        Raises:
          ValueError: If observation tensors lack outer `(batch,)` or
            `(batch, time)` axes.
        """
        num_outer_dims = nest_utils.get_outer_rank(observation,
                                                   self.input_tensor_spec)
        if num_outer_dims not in (1, 2):
            raise ValueError(
                'Input observation must have a batch or batch x time outer shape.')

        has_time_dim = num_outer_dims == 2
        if not has_time_dim:
            # Add a time dimension to the inputs.
            observation = tf.nest.map_structure(lambda t: tf.expand_dims(t, 1),
                                                observation)
            step_type = tf.nest.map_structure(lambda t: tf.expand_dims(t, 1),
                                              step_type)

        state, _ = self._input_encoder(
            observation, step_type=step_type, network_state=(), training=training)

        network_kwargs = {}
        if isinstance(self._lstm_network, dynamic_unroll_layer.DynamicUnroll):
            network_kwargs['reset_mask'] = tf.equal(step_type,
                                                    time_step.StepType.FIRST,
                                                    name='mask')

        # Unroll over the time sequence.
        output = self._lstm_network(
            inputs=state,
            initial_state=network_state,
            training=training,
            **network_kwargs)

        if isinstance(self._lstm_network, dynamic_unroll_layer.DynamicUnroll):
            state, network_state = output
        else:
            state = output[0]
            network_state = tf.nest.pack_sequence_as(
                self._lstm_network.cell.state_size, tf.nest.flatten(output[1:]))

        for layer in self._output_encoder:
            state = layer(state, training=training)

        if not has_time_dim:
            # Remove time dimension from the state.
            state = tf.squeeze(state, [1])

        return state, network_state


@gin.configurable
class QRnnNetwork(LSTMEncodingNetwork):
    """Recurrent network."""

    def __init__(
            self,
            input_tensor_spec,
            action_spec,
            preprocessing_layers=None,
            preprocessing_combiner=None,
            conv_layer_params=None,
            input_fc_layer_params=(75, 40),
            lstm_size=None,
            output_fc_layer_params=(75, 40),
            activation_fn=tf.keras.activations.relu,
            rnn_construction_fn=None,
            rnn_construction_kwargs=None,
            dtype=tf.float32,
            name='QRnnNetwork',
            batch_norm=False
    ):
        """Creates an instance of `QRnnNetwork`.

        Args:
          input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing the
            input observations.
          action_spec: A nest of `tensor_spec.BoundedTensorSpec` representing the
            actions.
          preprocessing_layers: (Optional.) A nest of `tf.keras.layers.Layer`
            representing preprocessing for the different observations.
            All of these layers must not be already built. For more details see
            the documentation of `networks.EncodingNetwork`.
          preprocessing_combiner: (Optional.) A keras layer that takes a flat list
            of tensors and combines them.  Good options include
            `tf.keras.layers.Add` and `tf.keras.layers.Concatenate(axis=-1)`.
            This layer must not be already built. For more details see
            the documentation of `networks.EncodingNetwork`.
          conv_layer_params: Optional list of convolution layers parameters, where
            each item is a length-three tuple indicating (filters, kernel_size,
            stride).
          input_fc_layer_params: Optional list of fully connected parameters, where
            each item is the number of units in the layer. These feed into the
            recurrent layer.
          lstm_size: An iterable of ints specifying the LSTM cell sizes to use.
          output_fc_layer_params: Optional list of fully connected parameters, where
            each item is the number of units in the layer. These are applied on top
            of the recurrent layer.
          activation_fn: Activation function, e.g. tf.keras.activations.relu,.
          rnn_construction_fn: (Optional.) Alternate RNN construction function, e.g.
            tf.keras.layers.LSTM, tf.keras.layers.CuDNNLSTM. It is invalid to
            provide both rnn_construction_fn and lstm_size.
          rnn_construction_kwargs: (Optional.) Dictionary or arguments to pass to
            rnn_construction_fn.

            The RNN will be constructed via:

            ```
            rnn_layer = rnn_construction_fn(**rnn_construction_kwargs)
            ```
          dtype: The dtype to use by the convolution, LSTM, and fully connected
            layers.
          name: A string representing name of the network.

        Raises:
          ValueError: If any of `preprocessing_layers` is already built.
          ValueError: If `preprocessing_combiner` is already built.
          ValueError: If `action_spec` contains more than one action.
          ValueError: If neither `lstm_size` nor `rnn_construction_fn` are provided.
          ValueError: If both `lstm_size` and `rnn_construction_fn` are provided.
        """
        validate_specs(action_spec, input_tensor_spec)
        action_spec = tf.nest.flatten(action_spec)[0]
        num_actions = action_spec.maximum - action_spec.minimum + 1

        q_projection = layers.Dense(
            num_actions,
            activation=None,
            kernel_initializer=tf.random_uniform_initializer(
                minval=-0.03, maxval=0.03),
            bias_initializer=tf.constant_initializer(-0.2),
            dtype=dtype,
            name='num_action_project/dense')

        super(QRnnNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            conv_layer_params=conv_layer_params,
            input_fc_layer_params=input_fc_layer_params,
            lstm_size=lstm_size,
            output_fc_layer_params=output_fc_layer_params,
            activation_fn=activation_fn,
            rnn_construction_fn=rnn_construction_fn,
            rnn_construction_kwargs=rnn_construction_kwargs,
            dtype=dtype,
            name=name, batch_norm=batch_norm)

        self._output_encoder.append(q_projection)
