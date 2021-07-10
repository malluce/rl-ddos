import logging
from typing import Any, List, Optional, Tuple

import gin
import tf_agents.utils.common
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tf_agents.agents import Td3Agent
from tf_agents.agents.ddpg.actor_network import ActorNetwork
from tf_agents.agents.ddpg.actor_rnn_network import ActorRnnNetwork
from tf_agents.agents.ddpg.critic_network import CriticNetwork
from tf_agents.agents.ddpg.critic_rnn_network import CriticRnnNetwork
from tf_agents.agents.tf_agent import LossInfo
from tf_agents.typing import types
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


@gin.configurable
class TD3WrapAgent(Td3Agent):

    def __init__(self, time_step_spec, action_spec, actor_layers=(400, 300), critic_obs_layers=(400,),
                 critic_act_layers=None, critic_joint_layers=(300,), actor_lr=1e-3, critic_lr=1e-3,
                 exploration_noise_std=0.1, target_update_tau=5e-3, target_update_period=2, actor_update_period=2,
                 gamma=0.99,
                 target_policy_noise=0.2, target_policy_clip=0.5,
                 ## RNN params
                 # actor
                 use_act_rnn=False, rnn_act_in_fc_layers=(200, 100), rnn_act_out_fc_layers=(200,),
                 rnn_act_lstm_size=(50,),
                 # critic
                 use_crt_rnn=False, rnn_crt_act_fc_layers=None, rnn_crt_obs_fc_layers=(200,),
                 rnn_crt_joint_fc_layers=(300,),
                 rnn_crt_lstm_size=(50,), rnn_crt_out_fc_layers=(200,)
                 ):
        actor_net = None
        if use_act_rnn:
            actor_net = ActorRnnNetwork(time_step_spec.observation, action_spec,
                                        input_fc_layer_params=rnn_act_in_fc_layers,
                                        lstm_size=rnn_act_lstm_size, output_fc_layer_params=rnn_act_out_fc_layers)
        else:
            actor_net = ActorNetwork(time_step_spec.observation, action_spec, actor_layers)

        critic_net = None
        if use_crt_rnn:
            critic_net = CriticRnnNetwork((time_step_spec.observation, action_spec),
                                          observation_fc_layer_params=rnn_crt_obs_fc_layers,
                                          action_fc_layer_params=rnn_crt_act_fc_layers,
                                          joint_fc_layer_params=rnn_crt_joint_fc_layers,
                                          lstm_size=rnn_crt_lstm_size,
                                          output_fc_layer_params=rnn_crt_out_fc_layers)
        else:
            critic_net = CriticNetwork((time_step_spec.observation, action_spec),
                                       observation_fc_layer_params=critic_obs_layers,
                                       action_fc_layer_params=critic_act_layers,
                                       joint_fc_layer_params=critic_joint_layers
                                       )

        # TODO lr decay?
        self.actor_optimizer = Adam(ExponentialDecay(1e-5, 20000, 0.96, staircase=True))
        self.critic_optimizer = Adam(ExponentialDecay(1e-5, 20000, 0.96, staircase=True))

        super(TD3WrapAgent, self).__init__(
            time_step_spec, action_spec,
            actor_net, critic_net,
            self.actor_optimizer, self.critic_optimizer,
            exploration_noise_std=exploration_noise_std,
            target_update_tau=target_update_tau, target_update_period=target_update_period,
            actor_update_period=actor_update_period,
            gamma=gamma, target_policy_noise=target_policy_noise, target_policy_noise_clip=target_policy_clip,
            td_errors_loss_fn=tf_agents.utils.common.element_wise_huber_loss,
            name='td3'
        )

    def _loss(self, experience: types.NestedTensor, weights: types.Tensor) -> Optional[LossInfo]:
        pass

    def get_scalars_to_log(self) -> List[Tuple[Any, str]]:  # TODO as method in superclass once more agents are added
        actor_lr = self.actor_optimizer._decayed_lr(tf.float32)
        critic_lr = self.critic_optimizer._decayed_lr(tf.float32)
        return [(actor_lr, 'actor_lr'), (critic_lr, 'actor_lr')]
