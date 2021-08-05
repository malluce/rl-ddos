#!/usr/bin/env python

import gin

from .distgen import TraceSampler, FlowGroupSampler
from .distgen import UniformSampler, WeibullSampler, NormalSampler
from .packet import Packet


@gin.configurable
class DistributionTrace(object):
    _MAXADDR = 0xffff
    _NUM_BENIGN_FLOWS = 25  # 500
    _NUM_ATTACK_FLOWS = 50  # 1000

    @staticmethod
    def __init_sampler(maxtime, maxaddr, benign_flows, attack_flows):
        """
        Initializes several FlowGroupSamplers and instantiates a TraceSampler from them, which is invoked to create
        flows.

        :param maxtime: the maximum time step of the flows that are to be generated (passed to samplers)
        :param maxaddr: the maximum ip address of the flows that are to be generated (passed to samplers)
        :param benign_flows: the number of benign flows
        :param attack_flows: the number of attack flows
        :return: the TraceSampler object
        """
        flowsamplers = [
            # 1st set of benign flows
            FlowGroupSampler(benign_flows,
                             UniformSampler(0, .95 * maxtime),
                             # 99% of all flows shall end before maxtime
                             WeibullSampler(3 / 2,
                                            (1 / WeibullSampler.quantile(99, 3 / 2)) * 1 / 8 * maxtime),
                             NormalSampler(1 / 2 * maxaddr, .17 * maxaddr, 1, maxaddr),
                             attack=False),
            # 1st set of attack flows
            FlowGroupSampler(attack_flows // 3,
                             UniformSampler(0, 3 / 6 * maxtime),
                             WeibullSampler(2,
                                            (1 / WeibullSampler.quantile(99, 2)) * 1 / 6 * maxtime),
                             NormalSampler(1 / 4 * maxaddr, .09 * maxaddr, 1, maxaddr),
                             attack=True),
            # 2nd set of attack flows
            FlowGroupSampler(attack_flows // 3,
                             UniformSampler(2 / 6 * maxtime, 5 / 6 * maxtime),
                             WeibullSampler(2,
                                            (1 / WeibullSampler.quantile(99, 2)) * 1 / 6 * maxtime),
                             NormalSampler(3 / 4 * maxaddr, .09 * maxaddr, 1, maxaddr),
                             attack=True),
            # 3rd set of attack flows
            FlowGroupSampler(attack_flows // 6,
                             UniformSampler(0, 5 / 6 * maxtime),
                             WeibullSampler(2,
                                            (1 / WeibullSampler.quantile(99, 2)) * 1 / 6 * maxtime),
                             NormalSampler(1 / 8 * maxaddr, .05 * maxaddr, 1, maxaddr),
                             attack=True),
            # 4th set of attack flows
            FlowGroupSampler(attack_flows // 6,
                             UniformSampler(2 / 6, 5 / 6 * maxtime),
                             WeibullSampler(2,
                                            (1 / WeibullSampler.quantile(99, 2)) * 1 / 6 * maxtime),
                             NormalSampler(3 / 8 * maxaddr, .05 * maxaddr, 1, maxaddr),
                             attack=True),
        ]

        # TODO replace by above (only a simple example)
        flowsamplers = [
            # 1st set of benign flows
            FlowGroupSampler(benign_flows,
                             UniformSampler(0, 1),
                             UniformSampler(maxtime, maxtime + 1),
                             UniformSampler(0x000, 0x7ff),  # subnet 0.0.0.0/21
                             attack=False),
            # 1st set of attack flows
            FlowGroupSampler(attack_flows,
                             UniformSampler(0, 1),
                             UniformSampler(maxtime / 4, maxtime / 2),  #
                             UniformSampler(0x800, 0xfff),  # subnet 0.0.8.0/21
                             attack=True)
        ]

        trace_sampler = TraceSampler(flowsamplers, maxtime)
        trace_sampler.init_flows()

        return trace_sampler

    def __init__(self, maxtime, maxaddr=_MAXADDR, num_benign_flows=_NUM_BENIGN_FLOWS,
                 num_attack_flows=_NUM_ATTACK_FLOWS):
        """
        Creates a new DistributionTrace, which will create packets for a given number of time steps whose ip address is
        not larger than maxaddr. The packets are drawn from benign_flows many benign and num_attack_flows many attack flows.

        :param maxtime: the number of time steps to generate packets for
        :param maxaddr: the maximum ip address of generated packets
        :param num_benign_flows: the number of benign flows to generate packets from
        :param num_attack_flows: the number of attack flows to generate packets from
        """
        self.maxtime = maxtime
        self.maxaddr = maxaddr
        self.benign_flows = num_benign_flows
        self.attack_flows = num_attack_flows
        self.rewind()

    def __next__(self):
        return self.next()

    def next(self) -> (Packet, bool):
        """
        Returns the next packet from the trace and a bool which indicates whether this packet is the last
        packet of the current time step.

        :raises StopIteration: when the end of the episode is reached (the trace finished)
        :returns: packet, step_finished
        """
        return next(self.samples)

    def __iter__(self):
        return self

    def rewind(self):
        """
        Re-draws samples from the TraceSampler, which is also re-initialized.
        Has to be used before calling next() after next() returned step_finished=True.
        """
        self.trace_sampler = DistributionTrace.__init_sampler(
            self.maxtime, self.maxaddr, self.benign_flows, self.attack_flows)
        self.samples = self.trace_sampler.samples()
        self.N = self.trace_sampler.num_samples
        print(f'number of packets next episode: {self.N}')

    def __len__(self):
        """
        Returns the number of packets of this trace. (equal to episode length in packets)

        :return: number of trace packets
        """
        return self.N
