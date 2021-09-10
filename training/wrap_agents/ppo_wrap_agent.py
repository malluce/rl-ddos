from collections import OrderedDict
from typing import Any, List, Optional, Tuple

import gin
from tensorflow.python.keras.layers import Lambda
from tf_agents.agents.ppo.ppo_clip_agent import PPOClipAgent
from tf_agents.agents.tf_agent import LossInfo
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork
from tf_agents.networks.value_network import ValueNetwork
from tf_agents.networks.value_rnn_network import ValueRnnNetwork
from tf_agents.typing import types
import tensorflow as tf

from training.wrap_agents.util import get_optimizer
from training.wrap_agents.wrap_agent import WrapAgent


@gin.configurable
class PPOWrapAgent(PPOClipAgent, WrapAgent):

    def __init__(self, time_step_spec, action_spec,
                 lr, lr_decay_steps=None, lr_decay_rate=None, exp_min_lr=None, linear_decay_end_lr=None,
                 linear_decay_steps=None,
                 # learning rate
                 gamma=0.99, num_epochs=5, importance_ratio_clipping=0.2,
                 actor_layers=(200, 100), value_layers=(200, 100),
                 actor_act_func=tf.keras.activations.tanh, value_act_func=tf.keras.activations.tanh,
                 use_actor_rnn=False, act_rnn_in_layers=(128, 64), act_rnn_lstm=(64,), act_rnn_out_layers=(128, 64),
                 use_value_rnn=False, val_rnn_in_layers=(128, 64), val_rnn_lstm=(64,), val_rnn_out_layers=(128, 64),
                 entropy_regularization=0.0, use_gae=False, gae_lambda=0.95, sum_grad_vars=False, gradient_clip=None,
                 cnn_act_func=tf.keras.activations.relu,
                 # CNN specs are 3-tuples of the following contents (all lists must have same length)
                 # ( ([conv_filters], [conv_kernel_sizes], [conv_strides]), ([pool_sizes], [pool_strides]), fc_units )
                 cnn_spec=None
                 ):
        self.gamma = gamma
        self.cnn_act_func = cnn_act_func
        self.optimizer = get_optimizer(lr, lr_decay_rate, lr_decay_steps, linear_decay_end_lr=linear_decay_end_lr,
                                       linear_decay_steps=linear_decay_steps, exp_min_lr=exp_min_lr)

        obs_spec = time_step_spec().observation
        if type(obs_spec) is OrderedDict and 'image' in obs_spec:
            print('setting up CNN!')

            cnn = self.build_cnn_from_spec(cnn_spec)
            hhh_cnn = self.build_cnn_from_spec(cnn_spec)

            preprocessing_layers = {
                'vector': Lambda(lambda x: x),  # pass-through layer
                'image': cnn,
                'hhh_image': hhh_cnn
            }
            preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)
        else:
            print('not using CNN')
            preprocessing_layers = None
            preprocessing_combiner = None

        # set actor net
        if use_actor_rnn:
            actor_net = ActorDistributionRnnNetwork(time_step_spec().observation, action_spec(),
                                                    input_fc_layer_params=act_rnn_in_layers, lstm_size=act_rnn_lstm,
                                                    output_fc_layer_params=act_rnn_out_layers,
                                                    activation_fn=actor_act_func,
                                                    preprocessing_layers=preprocessing_layers,
                                                    preprocessing_combiner=preprocessing_combiner)
        else:
            actor_net = ActorDistributionNetwork(time_step_spec().observation, action_spec(),
                                                 fc_layer_params=actor_layers, activation_fn=actor_act_func,
                                                 preprocessing_layers=preprocessing_layers,
                                                 preprocessing_combiner=preprocessing_combiner)

        # set value net
        if use_value_rnn:
            value_net = ValueRnnNetwork(time_step_spec().observation, input_fc_layer_params=val_rnn_in_layers,
                                        lstm_size=val_rnn_lstm, output_fc_layer_params=val_rnn_out_layers,
                                        activation_fn=value_act_func,
                                        preprocessing_layers=preprocessing_layers,
                                        preprocessing_combiner=preprocessing_combiner)
        else:
            value_net = ValueNetwork(time_step_spec().observation, fc_layer_params=value_layers,
                                     activation_fn=value_act_func,
                                     preprocessing_layers=preprocessing_layers,
                                     preprocessing_combiner=preprocessing_combiner)

        super().__init__(time_step_spec(), action_spec(), optimizer=self.optimizer,
                         actor_net=actor_net, value_net=value_net,
                         importance_ratio_clipping=importance_ratio_clipping, discount_factor=gamma,
                         num_epochs=num_epochs, name='ppo', entropy_regularization=entropy_regularization,
                         use_gae=use_gae, lambda_value=gae_lambda, summarize_grads_and_vars=sum_grad_vars,
                         debug_summaries=sum_grad_vars, gradient_clipping=gradient_clip
                         )

    def build_cnn_from_spec(self, cnn_spec):
        conv_layers, pool_layers, fc_units = cnn_spec
        conv_filters, conv_kernel_sizes, conv_strides = conv_layers
        pool_sizes, pool_strides = pool_layers
        assert len(conv_filters) == len(conv_kernel_sizes) == len(conv_strides) == len(pool_sizes) == \
               len(pool_strides)
        keras_layers = []
        # alternating conv and max pool layers
        for filters, kernel_size, stride, pool_size, pool_stride in zip(conv_filters, conv_kernel_sizes,
                                                                        conv_strides, pool_sizes, pool_strides):
            keras_layers.append(
                tf.keras.layers.Conv2D(filters, kernel_size, activation=self.cnn_act_func, strides=stride)
            )
            keras_layers.append(tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=pool_stride))
        # flatten and dense at the end
        keras_layers.append(tf.keras.layers.Flatten())
        keras_layers.append(tf.keras.layers.Dense(fc_units, activation=self.cnn_act_func))
        return tf.keras.models.Sequential(keras_layers)

    def _loss(self, experience: types.NestedTensor, weights: types.Tensor) -> Optional[LossInfo]:
        pass

    def get_scalars_to_log(self) -> List[Tuple[Any, str]]:  # TODO as method in superclass once more agents are added
        return [(self.optimizer._decayed_lr(tf.float32), 'lr')]

    def get_gamma(self):
        return self.gamma
