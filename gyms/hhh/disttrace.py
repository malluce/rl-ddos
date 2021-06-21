#!/usr/bin/env python

import gin

from .distgen import TraceSampler, FlowGroupSampler
from .distgen import UniformSampler, WeibullSampler, NormalSampler
from .packet import Packet


@gin.configurable
class DistributionTrace(object):
    _MAXADDR = 0xffff
    _NUM_BENIGN_FLOWS = 1000
    _NUM_ATTACK_FLOWS = 2000

    @staticmethod
    def __init_sampler(maxtime, maxaddr, benign_flows, attack_flows):
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

        trace_sampler = TraceSampler(flowsamplers, maxtime)
        trace_sampler.init_flows()

        return trace_sampler

    def __init__(self, maxtime, maxaddr=_MAXADDR, benign_flows=_NUM_BENIGN_FLOWS,
                 attack_flows=_NUM_ATTACK_FLOWS):
        self.maxtime = maxtime
        self.maxaddr = maxaddr
        self.benign_flows = benign_flows
        self.attack_flows = attack_flows
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
        if self.i == self.N:
            raise StopIteration()

        addr, rate, attack, step_finished = next(self.samples)
        self.i += 1

        return Packet(addr, attack == 1), step_finished

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
        self.i = 0
        print(f'number of packets next episode: {self.N}')

    def __len__(self):
        return self.N
