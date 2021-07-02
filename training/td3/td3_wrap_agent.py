from typing import Optional

import gin
from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import ExponentialDecay, LearningRateSchedule
from tf_agents.agents import Td3Agent
from tf_agents.agents.ddpg.actor_network import ActorNetwork
from tf_agents.agents.ddpg.critic_network import CriticNetwork
from tf_agents.agents.tf_agent import LossInfo
from tf_agents.typing import types
from tensorflow.keras.optimizers import Adam


@gin.configurable
class TD3WrapAgent(Td3Agent):

    def __init__(self, time_step_spec, action_spec, actor_layers=(400, 300), critic_obs_layers=(400,),
                 critic_act_layers=None, critic_joint_layers=(300,), actor_lr=1e-3, critic_lr=1e-3,
                 exploration_noise_std=0.1, target_update_tau=5e-3, target_update_period=2, actor_update_period=2,
                 gamma=0.99,
                 target_policy_noise=0.2, target_policy_clip=0.5
                 ):
        actor_net = ActorNetwork(time_step_spec.observation, action_spec, actor_layers)
        critic_net = CriticNetwork((time_step_spec.observation, action_spec),
                                   observation_fc_layer_params=critic_obs_layers,
                                   action_fc_layer_params=critic_act_layers,
                                   joint_fc_layer_params=critic_joint_layers
                                   )
        # TODO lr decay?
        actor_optimizer = Adam(actor_lr)
        critic_optimizer = Adam(critic_lr)

        super(TD3WrapAgent, self).__init__(
            time_step_spec, action_spec,
            actor_net, critic_net,
            actor_optimizer, critic_optimizer,
            exploration_noise_std=exploration_noise_std,
            target_update_tau=target_update_tau, target_update_period=target_update_period,
            actor_update_period=actor_update_period,
            gamma=gamma, target_policy_noise=target_policy_noise, target_policy_noise_clip=target_policy_clip,
            name='td3'
        )

    def _loss(self, experience: types.NestedTensor, weights: types.Tensor) -> Optional[LossInfo]:
        pass
