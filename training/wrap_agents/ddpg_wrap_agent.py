from typing import Any, List, Optional, Tuple

import gin
import numpy as np
from tf_agents.agents import DdpgAgent
from tf_agents.agents.tf_agent import LossInfo
from tf_agents.networks import normal_projection_network
import tensorflow as tf
from tf_agents.typing import types

from nets.ddpg_actor_network import ActorNetwork
from nets.ddpg_critic_network import CriticNetwork
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
                 target_update_tau=0.001, target_update_period=1, gamma=0,
                 critic_lr=1e-3, critic_lr_decay_rate=None, critic_lr_decay_steps=None, critic_exp_min_lr=None,
                 actor_lr=1e-4, actor_lr_decay_rate=None, actor_lr_decay_steps=None, actor_exp_min_lr=None,
                 ou_std=0.2, ou_mean=0.15,
                 cnn_spec=None, cnn_act_func=tf.keras.activations.relu,
                 gradient_clip=None, dqda_clip=None,
                 batch_norm=False,
                 debug=False
                 ):
        self.gamma = gamma

        preprocessing_combiner, preprocessing_layers = get_preprocessing_cnn(cnn_spec, time_step_spec, cnn_act_func,
                                                                             batch_norm=batch_norm)

        actor_net = ActorNetwork(time_step_spec.observation, action_spec, actor_layers,
                                 preprocessing_layers=preprocessing_layers,
                                 preprocessing_combiner=preprocessing_combiner,
                                 batch_norm=batch_norm)

        critic_net = CriticNetwork((time_step_spec.observation, action_spec),
                                   observation_fc_layer_params=critic_obs_fc_layers,
                                   action_fc_layer_params=critic_action_fc_layers,
                                   joint_fc_layer_params=critic_joint_fc_layers,
                                   preprocessing_layers=preprocessing_layers,
                                   preprocessing_combiner=preprocessing_combiner,
                                   batch_norm=batch_norm
                                   )

        # set lr (decay)
        self.crit_opt = get_optimizer(critic_lr, critic_lr_decay_rate, critic_lr_decay_steps,
                                      exp_min_lr=critic_exp_min_lr)
        self.act_opt = get_optimizer(actor_lr, actor_lr_decay_rate, actor_lr_decay_steps, exp_min_lr=actor_exp_min_lr)

        super().__init__(time_step_spec, action_spec, actor_net, critic_net, self.act_opt, self.crit_opt, ou_std,
                         ou_mean,
                         gamma=gamma, target_update_period=target_update_period, target_update_tau=target_update_tau,
                         gradient_clipping=gradient_clip, dqda_clipping=dqda_clip, debug_summaries=debug,
                         name='ddpg')

    def get_scalars_to_log(self) -> List[Tuple[Any, str]]:  # TODO as method in superclass once more agents are added
        return [(self.crit_opt._decayed_lr(tf.float32), 'critic_lr'),
                (self.act_opt._decayed_lr(tf.float32), 'actor_lr')]

    def get_gamma(self):
        return self.gamma

    def _loss(self, experience: types.NestedTensor, weights: types.Tensor) -> Optional[LossInfo]:
        pass
