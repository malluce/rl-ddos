import logging
from typing import Any, List, Tuple

import gin
import numpy as np
from tf_agents.agents import DdpgAgent
from tf_agents.networks import normal_projection_network
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
import tensorflow as tf

from agents.nets.ddpg_actor_network import ActorNetwork
from agents.nets.ddpg_critic_network import CriticNetwork
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
class DDPGWrapAgent(DdpgAgent, WrapAgent):

    def __init__(self, time_step_spec, action_spec, actor_layers=(400, 300), critic_obs_fc_layers=(400,),
                 critic_action_fc_layers=None, critic_joint_fc_layers=(300,),
                 target_update_tau=0.001, target_update_period=1, gamma=0, critic_lr=1e-3, actor_lr=1e-4,
                 ou_std=0.2, ou_mean=0.15,
                 cnn_spec=None, cnn_act_func=tf.keras.activations.relu

                 ):
        self.gamma = gamma

        preprocessing_combiner, preprocessing_layers = get_preprocessing_cnn(cnn_spec, time_step_spec, cnn_act_func)

        actor_net = ActorNetwork(time_step_spec.observation, action_spec, actor_layers,
                                 preprocessing_layers=preprocessing_layers,
                                 preprocessing_combiner=preprocessing_combiner)

        critic_net = CriticNetwork((time_step_spec.observation, action_spec),
                                   observation_fc_layer_params=critic_obs_fc_layers,
                                   action_fc_layer_params=critic_action_fc_layers,
                                   joint_fc_layer_params=critic_joint_fc_layers,
                                   preprocessing_layers=preprocessing_layers,
                                   preprocessing_combiner=preprocessing_combiner
                                   )

        # set lr (decay)
        self.crit_opt = get_optimizer(critic_lr, None, None)
        self.act_opt = get_optimizer(actor_lr, None, None)

        super().__init__(time_step_spec, action_spec, actor_net, critic_net, self.act_opt, self.crit_opt, ou_std,
                         ou_mean,
                         gamma=gamma, target_update_period=target_update_period, target_update_tau=target_update_tau,
                         name='ddpg')

    def get_scalars_to_log(self) -> List[Tuple[Any, str]]:  # TODO as method in superclass once more agents are added
        return [(self.crit_opt._decayed_lr(tf.float32), 'critic_lr'),
                (self.act_opt._decayed_lr(tf.float32), 'actor_lr')]

    def get_gamma(self):
        return self.gamma
