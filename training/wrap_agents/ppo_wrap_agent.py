from typing import Any, List, Optional, Tuple

import gin
from tf_agents.agents.ppo.ppo_clip_agent import PPOClipAgent
from tf_agents.agents.tf_agent import LossInfo
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork
from tf_agents.networks.value_network import ValueNetwork
from tf_agents.networks.value_rnn_network import ValueRnnNetwork
from tf_agents.typing import types
import tensorflow as tf

from training.wrap_agents.util import get_optimizer


@gin.configurable
class PPOWrapAgent(PPOClipAgent):

    def __init__(self, time_step_spec, action_spec,
                 lr, lr_decay_steps, lr_decay_rate, exp_min_lr, linear_decay_end_lr, linear_decay_steps,
                 # learning rate
                 gamma, num_epochs, importance_ratio_clipping=0.2,
                 actor_layers=(200, 100), value_layers=(200, 100),
                 use_actor_rnn=False, act_rnn_in_layers=(128, 64), act_rnn_lstm=(64,), act_rnn_out_layers=(128, 64),
                 use_value_rnn=False, val_rnn_in_layers=(128, 64), val_rnn_lstm=(64,), val_rnn_out_layers=(128, 64),
                 entropy_regularization=0.0
                 ):
        self.optimizer = get_optimizer(lr, lr_decay_rate, lr_decay_steps, linear_decay_end_lr=linear_decay_end_lr,
                                       linear_decay_steps=linear_decay_steps, exp_min_lr=exp_min_lr)
        # set actor net
        if use_actor_rnn:
            actor_net = ActorDistributionRnnNetwork(time_step_spec().observation, action_spec(),
                                                    input_fc_layer_params=act_rnn_in_layers, lstm_size=act_rnn_lstm,
                                                    output_fc_layer_params=act_rnn_out_layers)
        else:
            actor_net = ActorDistributionNetwork(time_step_spec().observation, action_spec(),
                                                 fc_layer_params=actor_layers)

        # set value net
        if use_value_rnn:
            value_net = ValueRnnNetwork(time_step_spec().observation, input_fc_layer_params=val_rnn_in_layers,
                                        lstm_size=val_rnn_lstm, output_fc_layer_params=val_rnn_out_layers)
        else:
            value_net = ValueNetwork(time_step_spec().observation, fc_layer_params=value_layers)

        super().__init__(time_step_spec(), action_spec(), optimizer=self.optimizer,
                         actor_net=actor_net, value_net=value_net,
                         importance_ratio_clipping=importance_ratio_clipping, discount_factor=gamma,
                         num_epochs=num_epochs, name='ppo', entropy_regularization=entropy_regularization
                         )

    def _loss(self, experience: types.NestedTensor, weights: types.Tensor) -> Optional[LossInfo]:
        pass

    def get_scalars_to_log(self) -> List[Tuple[Any, str]]:  # TODO as method in superclass once more agents are added
        return [(self.optimizer._decayed_lr(tf.float32), 'lr')]
