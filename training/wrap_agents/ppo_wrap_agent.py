from typing import Any, List, Optional, Tuple

import gin
import numpy as np
from tf_agents.agents.ppo.ppo_clip_agent import PPOClipAgent
from tf_agents.agents.tf_agent import LossInfo
from tf_agents.networks import normal_projection_network
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork
from tf_agents.networks.value_network import ValueNetwork
from tf_agents.networks.value_rnn_network import ValueRnnNetwork
from tf_agents.typing import types
import tensorflow as tf

from training.wrap_agents.util import get_optimizer, get_preprocessing_cnn
from training.wrap_agents.wrap_agent import WrapAgent


def _normal_projection_net(action_spec,
                           init_action_stddev=0.1,  # 0.1 instead of 0.35 -> less exploration?
                           init_means_output_factor=0.1):
    std_bias_initializer_value = np.log(np.exp(init_action_stddev) - 1)

    return normal_projection_network.NormalProjectionNetwork(
        action_spec,
        init_means_output_factor=init_means_output_factor,
        std_bias_initializer_value=std_bias_initializer_value,
        scale_distribution=False)


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
                 cnn_spec=None
                 ):
        self.gamma = gamma
        self.optimizer = get_optimizer(lr, lr_decay_rate, lr_decay_steps, linear_decay_end_lr=linear_decay_end_lr,
                                       linear_decay_steps=linear_decay_steps, exp_min_lr=exp_min_lr)

        preprocessing_combiner, preprocessing_layers = get_preprocessing_cnn(cnn_spec, time_step_spec, cnn_act_func)

        # set actor net
        if use_actor_rnn:
            actor_net = ActorDistributionRnnNetwork(time_step_spec().observation, action_spec(),
                                                    input_fc_layer_params=act_rnn_in_layers, lstm_size=act_rnn_lstm,
                                                    output_fc_layer_params=act_rnn_out_layers,
                                                    activation_fn=actor_act_func,
                                                    preprocessing_layers=preprocessing_layers,
                                                    preprocessing_combiner=preprocessing_combiner,
                                                    continuous_projection_net=_normal_projection_net)
        else:
            actor_net = ActorDistributionNetwork(time_step_spec().observation, action_spec(),
                                                 fc_layer_params=actor_layers, activation_fn=actor_act_func,
                                                 preprocessing_layers=preprocessing_layers,
                                                 preprocessing_combiner=preprocessing_combiner,
                                                 continuous_projection_net=_normal_projection_net)

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
