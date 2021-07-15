import logging
from typing import Any, List, Tuple

import gin
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tf_agents.agents import DqnAgent
from tf_agents.networks.q_rnn_network import QRnnNetwork
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from agents.nets.dqn_q_network import QNetwork
from training.wrap_agents.util import get_optimizer


@gin.configurable
class DQNWrapAgent(DqnAgent):

    def __init__(self, time_step_spec, action_spec, q_layers=(75, 40),
                 use_rnn=False, rnn_input_layers=(75, 40), rnn_lstm_size=(128, 128, 128), rnn_output_layers=(75, 40),
                 target_update_tau=1, target_update_period=5, gamma=0.99, lr=1e-3, lr_decay_steps=None,
                 lr_decay_rate=None, eps_greedy=0.1):
        # set q net
        if not use_rnn:
            self.q_net = QNetwork(time_step_spec.observation, action_spec, batch_normalization=True,
                                  fc_layer_params=q_layers)
        else:
            self.q_net = QRnnNetwork(time_step_spec.observation, action_spec, input_fc_layer_params=rnn_input_layers,
                                     lstm_size=rnn_lstm_size, output_fc_layer_params=rnn_output_layers)

        # set lr (decay)
        self.optimizer = get_optimizer(lr, lr_decay_rate, lr_decay_steps)

        super().__init__(time_step_spec, action_spec, self.q_net, self.optimizer, epsilon_greedy=eps_greedy,
                         gamma=gamma, target_update_period=target_update_period, target_update_tau=target_update_tau,
                         name='dqn')

    def get_scalars_to_log(self) -> List[Tuple[Any, str]]:  # TODO as method in superclass once more agents are added
        return [(self.optimizer._decayed_lr(tf.float32), 'lr')]
