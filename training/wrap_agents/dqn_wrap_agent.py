import logging
from typing import Any, List, Tuple

import gin
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tf_agents.agents import DqnAgent
from tf_agents.networks.q_rnn_network import QRnnNetwork
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from agents.nets.dqn_q_network import QNetwork
from training.wrap_agents.util import MinExpSchedule, get_optimizer, get_preprocessing_cnn
from training.wrap_agents.wrap_agent import WrapAgent


@gin.configurable
class DQNWrapAgent(DqnAgent, WrapAgent):

    def __init__(self, time_step_spec, action_spec, q_layers=(75, 40),
                 use_rnn=False, rnn_input_layers=(75, 40), rnn_lstm_size=(128, 128, 128), rnn_output_layers=(75, 40),
                 target_update_tau=1, target_update_period=5, gamma=0.99, lr=1e-3, lr_decay_steps=None,
                 lr_decay_rate=None, eps_greedy=0.05, eps_greedy_end=None, eps_greedy_steps=None,
                 eps_greedy_decay_exp=False,
                 cnn_spec=None, cnn_act_func=tf.keras.activations.relu, batch_norm=False):
        self.gamma = gamma
        self.eps_greedy = eps_greedy

        preprocessing_combiner, preprocessing_layers = get_preprocessing_cnn(cnn_spec, time_step_spec, cnn_act_func)

        # set q net
        if not use_rnn:
            self.q_net = QNetwork(time_step_spec.observation, action_spec, batch_normalization=batch_norm,
                                  fc_layer_params=q_layers, preprocessing_layers=preprocessing_layers,
                                  preprocessing_combiner=preprocessing_combiner)
        else:
            self.q_net = QRnnNetwork(time_step_spec.observation, action_spec, input_fc_layer_params=rnn_input_layers,
                                     lstm_size=rnn_lstm_size, output_fc_layer_params=rnn_output_layers,
                                     preprocessing_layers=preprocessing_layers,
                                     preprocessing_combiner=preprocessing_combiner)

        # set lr (decay)
        self.optimizer = get_optimizer(lr, lr_decay_rate, lr_decay_steps)

        if eps_greedy_decay_exp:
            self.eps_greedy = MinExpSchedule()  # params set via gin (hafner)
        elif eps_greedy_end is not None and eps_greedy_steps is not None:
            self.eps_greedy = tf.compat.v1.train.polynomial_decay(
                learning_rate=eps_greedy, global_step=tf.compat.v1.train.get_or_create_global_step(),
                decay_steps=eps_greedy_steps, end_learning_rate=eps_greedy_end)

        super().__init__(time_step_spec, action_spec, self.q_net, self.optimizer, epsilon_greedy=self.eps_greedy,
                         gamma=gamma, target_update_period=target_update_period, target_update_tau=target_update_tau,
                         name='dqn')

    def get_scalars_to_log(self) -> List[Tuple[Any, str]]:  # TODO as method in superclass once more agents are added
        return [(self.optimizer._decayed_lr(tf.float32), 'lr'),
                (self.eps_greedy if type(self.eps_greedy) is int else self.eps_greedy(), 'epsilon')]

    def get_gamma(self):
        return self.gamma
