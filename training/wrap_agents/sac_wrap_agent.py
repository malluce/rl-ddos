from typing import Any, List, Optional, Tuple

import gin
import numpy as np
import tensorflow as tf
from tf_agents.agents import SacAgent
from tf_agents.agents.ddpg.critic_rnn_network import CriticRnnNetwork
from tf_agents.agents.tf_agent import LossInfo
from tf_agents.networks import network, normal_projection_network
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork

from agents.nets.ddpg_critic_network import CriticNetwork
from training.wrap_agents.util import get_optimizer, get_preprocessing_cnn
from training.wrap_agents.wrap_agent import WrapAgent
from tf_agents.typing import types


def _normal_projection_net(action_spec,
                           init_action_stddev=0.1,  # 0.1 instead of 0.35 -> less exploration?
                           init_means_output_factor=0.1):
    std_bias_initializer_value = np.log(np.exp(init_action_stddev) - 1)

    return normal_projection_network.NormalProjectionNetwork(
        action_spec,
        init_means_output_factor=init_means_output_factor,
        std_bias_initializer_value=std_bias_initializer_value,
        scale_distribution=False)


@gin.configurable('SACWrapAgent')
class SACWrapAgent(SacAgent, WrapAgent):

    def __init__(self, time_step_spec, action_spec,
                 actor_layers=(256, 256), critic_obs_layers=None, critic_act_layers=None,
                 critic_joint_layers=(256, 256),
                 target_update_tau=5e-3, reward_scale=1.0,
                 gamma=0.99, actor_lr=3e-4, actor_lr_decay_steps=None,
                 actor_lr_decay_rate=None, critic_lr=3e-4, critic_lr_decay_steps=None, critic_lr_decay_rate=None,
                 alpha_lr=3e-4, alpha_lr_decay_rate=None, alpha_lr_decay_steps=None,
                 use_actor_rnn=False, rnn_act_in_fc_layers=(200, 100), rnn_act_out_fc_layers=(200,),
                 rnn_act_lstm_size=(50,), use_crt_rnn=False, rnn_crt_act_fc_layers=None, rnn_crt_obs_fc_layers=(200,),
                 rnn_crt_joint_fc_layers=(300,), rnn_crt_lstm_size=(50,), rnn_crt_out_fc_layers=(200,), cnn_spec=None,
                 cnn_act_func=tf.keras.activations.relu):

        self.gamma = gamma

        preprocessing_combiner, preprocessing_layers = get_preprocessing_cnn(cnn_spec, time_step_spec, cnn_act_func)

        # set actor net
        if use_actor_rnn:
            actor_net = ActorDistributionRnnNetwork(time_step_spec.observation, action_spec,
                                                    input_fc_layer_params=rnn_act_in_fc_layers,
                                                    lstm_size=rnn_act_lstm_size,
                                                    output_fc_layer_params=rnn_act_out_fc_layers,
                                                    activation_fn=tf.keras.activations.relu,
                                                    preprocessing_layers=preprocessing_layers,
                                                    preprocessing_combiner=preprocessing_combiner,
                                                    continuous_projection_net=_normal_projection_net)
        else:
            actor_net = ActorDistributionNetwork(time_step_spec.observation, action_spec,
                                                 fc_layer_params=actor_layers, activation_fn=tf.keras.activations.relu,
                                                 preprocessing_layers=preprocessing_layers,
                                                 preprocessing_combiner=preprocessing_combiner,
                                                 continuous_projection_net=_normal_projection_net)

        # set critic net
        if use_crt_rnn:
            # TODO preprocessing for RNN
            critic_net = CriticRnnNetwork((time_step_spec.observation, action_spec),
                                          observation_fc_layer_params=rnn_crt_obs_fc_layers,
                                          action_fc_layer_params=rnn_crt_act_fc_layers,
                                          joint_fc_layer_params=rnn_crt_joint_fc_layers,
                                          lstm_size=rnn_crt_lstm_size,
                                          output_fc_layer_params=rnn_crt_out_fc_layers,
                                          preprocessing_layers=preprocessing_layers,
                                          preprocessing_combiner=preprocessing_combiner)
        else:
            critic_net = CriticNetwork((time_step_spec.observation, action_spec),
                                       observation_fc_layer_params=critic_obs_layers,
                                       action_fc_layer_params=critic_act_layers,
                                       joint_fc_layer_params=critic_joint_layers,
                                       preprocessing_layers=preprocessing_layers,
                                       preprocessing_combiner=preprocessing_combiner
                                       )

        # set lr (decay)
        self.actor_optimizer = get_optimizer(actor_lr, actor_lr_decay_rate, actor_lr_decay_steps)
        self.critic_optimizer = get_optimizer(critic_lr, critic_lr_decay_rate, critic_lr_decay_steps)
        self.alpha_optimizer = get_optimizer(alpha_lr, alpha_lr_decay_rate, alpha_lr_decay_steps)

        # create agent
        super().__init__(
            time_step_spec, action_spec,
            critic_net, actor_net,
            self.actor_optimizer, self.critic_optimizer, self.alpha_optimizer,
            target_update_tau=target_update_tau,
            name='sac',
            reward_scale_factor=reward_scale
        )

    def _loss(self, experience: types.NestedTensor, weights: types.Tensor) -> Optional[LossInfo]:
        pass

    def get_scalars_to_log(self) -> List[
        Tuple[Any, str]]:  # TODO as method in superclass once more agents are added
        actor_lr = self.actor_optimizer._decayed_lr(tf.float32)
        critic_lr = self.critic_optimizer._decayed_lr(tf.float32)
        alpha_lr = self.alpha_optimizer._decayed_lr(tf.float32)
        return [(actor_lr, 'actor_lr'), (critic_lr, 'critic_lr'), (alpha_lr, 'alpha_lr')]

    def get_gamma(self):
        return self.gamma
