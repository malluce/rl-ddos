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
                 entropy_regularization=0.0, use_gae=False, gae_lambda=0.95, sum_grad_vars=False, gradient_clip=None
                 ):
        self.gamma = gamma
        self.optimizer = get_optimizer(lr, lr_decay_rate, lr_decay_steps, linear_decay_end_lr=linear_decay_end_lr,
                                       linear_decay_steps=linear_decay_steps, exp_min_lr=exp_min_lr)

        obs_spec = time_step_spec().observation
        if type(obs_spec) is OrderedDict and 'hhh_image' in obs_spec and 'filter_image' in obs_spec:
            print('setting up CNNs!')
            hhh_cnn = tf.keras.models.Sequential([
                # B,17,512,1
                tf.keras.layers.Conv2D(16, (2, 4), activation=tf.keras.activations.relu, strides=(1, 2)),
                # B,16,255,16
                tf.keras.layers.MaxPool2D(pool_size=(1, 2), strides=(1, 2)),
                # B,16,127,16
                tf.keras.layers.Conv2D(32, (2, 4), activation=tf.keras.activations.relu, strides=(1, 2)),
                # B,15,62,32
                tf.keras.layers.MaxPool2D(pool_size=(1, 2), strides=(1, 2)),
                # B,15,31,32
                tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.relu, strides=3),
                # B,5,10,64
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
                # B,2,4,64
                tf.keras.layers.Flatten(),
                # B,512
                tf.keras.layers.Dense(64, activation=tf.keras.activations.relu)
            ])

            filter_cnn = tf.keras.models.Sequential([
                # B,17,512,1
                tf.keras.layers.Conv2D(16, (2, 4), activation=tf.keras.activations.relu, strides=(1, 2)),
                # B,16,255,16
                tf.keras.layers.MaxPool2D(pool_size=(1, 2), strides=(1, 2)),
                # B,16,127,16
                tf.keras.layers.Conv2D(32, (2, 4), activation=tf.keras.activations.relu, strides=(1, 2)),
                # B,15,62,32
                tf.keras.layers.MaxPool2D(pool_size=(1, 2), strides=(1, 2)),
                # B,15,31,32
                tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.relu, strides=3),
                # B,5,10,64
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
                # B,2,4,64
                tf.keras.layers.Flatten(),
                # B,512
                tf.keras.layers.Dense(64, activation=tf.keras.activations.relu)
            ])

            preprocessing_layers = {
                'vector': Lambda(lambda x: x),  # pass-through layer
                'hhh_image': hhh_cnn,
                'filter_image': filter_cnn
            }
            preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)
        else:
            print('not using CNNs')
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

    def _loss(self, experience: types.NestedTensor, weights: types.Tensor) -> Optional[LossInfo]:
        pass

    def get_scalars_to_log(self) -> List[Tuple[Any, str]]:  # TODO as method in superclass once more agents are added
        return [(self.optimizer._decayed_lr(tf.float32), 'lr')]

    def get_gamma(self):
        return self.gamma
