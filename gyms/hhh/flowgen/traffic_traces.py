import math
from abc import ABC, abstractmethod

import gin

from gyms.hhh.flowgen.distgen import FlowGroupSampler, NormalSampler, UniformSampler, WeibullSampler


@gin.configurable
class TrafficTrace(ABC):
    @abstractmethod
    def get_flow_group_samplers(self):
        pass


@gin.configurable
class T1(TrafficTrace):
    def __init__(self, num_benign, num_attack, maxtime):
        self.num_benign = num_benign
        self.num_attack = num_attack
        self.maxtime = maxtime

    def get_flow_group_samplers(self):
        return [
            # 1st set of benign flows
            FlowGroupSampler(self.num_benign,
                             UniformSampler(0, 1),
                             UniformSampler(self.maxtime, self.maxtime + 1),
                             UniformSampler(0x000, 0x7ff),  # subnet 0.0.0.0/21
                             attack=False),
            # 1st set of attack flows
            FlowGroupSampler(self.num_attack,
                             UniformSampler(0, 1),
                             UniformSampler(self.maxtime, self.maxtime + 1),  #
                             UniformSampler(0x800, 0xfff),  # subnet 0.0.8.0/21
                             attack=True)
        ]


@gin.configurable
class T2(TrafficTrace):

    def __init__(self, num_benign, num_attack, maxtime):
        self.num_benign = num_benign
        self.num_attack = num_attack
        self.maxtime = maxtime

    def get_flow_group_samplers(self):
        return [
            # 1st set of benign flows
            FlowGroupSampler(self.num_benign,
                             UniformSampler(0, 1),
                             UniformSampler(self.maxtime, self.maxtime + 1),
                             UniformSampler(0x000, 0x7ff),  # subnet 0.0.0.0/21
                             attack=False),
            # 1st set of attack flows
            FlowGroupSampler(self.num_attack,
                             UniformSampler(0, 1),
                             UniformSampler(self.maxtime / 4, self.maxtime / 2),  #
                             UniformSampler(0x800, 0xfff),  # subnet 0.0.8.0/21
                             attack=True)
        ]


@gin.configurable
class T3(TrafficTrace):

    def __init__(self, num_benign=300, num_attack=150, maxtime=1000, maxaddr=0xff):
        self.num_benign = num_benign
        self.num_attack = num_attack
        self.maxtime = maxtime
        self.maxaddr = maxaddr

    def get_flow_group_samplers(self):
        return [
            # 1st set of benign flows
            FlowGroupSampler(self.num_benign,
                             UniformSampler(0, 0.95 * self.maxtime),
                             WeibullSampler(3 / 2,
                                            (1 / WeibullSampler.quantile(99, 3 / 2)) * 1 / 8 * self.maxtime),
                             NormalSampler(1 / 2 * self.maxaddr, .17 * self.maxaddr, 1, self.maxaddr),
                             attack=False),
            # 1st set of attack flows
            FlowGroupSampler(self.num_attack // 3,
                             UniformSampler(0, 1),
                             UniformSampler(0.2 * self.maxtime, 0.25 * self.maxtime),
                             UniformSampler(0, math.floor(self.maxaddr / 2)),
                             attack=True),
            # 2nd set of attack flows
            FlowGroupSampler(2 * self.num_attack // 3,
                             UniformSampler(0.6 * self.maxtime, 0.65 * self.maxtime + 1),
                             UniformSampler(0.9 * self.maxtime, self.maxtime),
                             UniformSampler(math.ceil(self.maxaddr / 2), self.maxaddr),
                             attack=True)
        ]


@gin.configurable
class THauke(TrafficTrace):

    def __init__(self, benign_flows, attack_flows, maxtime, maxaddr):
        self.attack_flows = attack_flows
        self.benign_flows = benign_flows
        self.maxtime = maxtime
        self.maxaddr = maxaddr

    def get_flow_group_samplers(self):
        return [
            # 1st set of benign flows
            FlowGroupSampler(self.benign_flows,
                             UniformSampler(0, .95 * self.maxtime),
                             # 99% of all flows shall end before self.maxtime
                             WeibullSampler(3 / 2,
                                            (1 / WeibullSampler.quantile(99, 3 / 2)) * 1 / 8 * self.maxtime),
                             NormalSampler(1 / 2 * self.maxaddr, .17 * self.maxaddr, 1, self.maxaddr),
                             attack=False),
            # 1st set of attack flows
            FlowGroupSampler(self.attack_flows // 3,
                             UniformSampler(0, 3 / 6 * self.maxtime),
                             WeibullSampler(2,
                                            (1 / WeibullSampler.quantile(99, 2)) * 1 / 6 * self.maxtime),
                             NormalSampler(1 / 4 * self.maxaddr, .09 * self.maxaddr, 1, self.maxaddr),
                             attack=True),
            # 2nd set of attack flows
            FlowGroupSampler(self.attack_flows // 3,
                             UniformSampler(2 / 6 * self.maxtime, 5 / 6 * self.maxtime),
                             WeibullSampler(2,
                                            (1 / WeibullSampler.quantile(99, 2)) * 1 / 6 * self.maxtime),
                             NormalSampler(3 / 4 * self.maxaddr, .09 * self.maxaddr, 1, self.maxaddr),
                             attack=True),
            # 3rd set of attack flows
            FlowGroupSampler(self.attack_flows // 6,
                             UniformSampler(0, 5 / 6 * self.maxtime),
                             WeibullSampler(2,
                                            (1 / WeibullSampler.quantile(99, 2)) * 1 / 6 * self.maxtime),
                             NormalSampler(1 / 8 * self.maxaddr, .05 * self.maxaddr, 1, self.maxaddr),
                             attack=True),
            # 4th set of attack flows
            FlowGroupSampler(self.attack_flows // 6,
                             UniformSampler(2 / 6, 5 / 6 * self.maxtime),
                             WeibullSampler(2,
                                            (1 / WeibullSampler.quantile(99, 2)) * 1 / 6 * self.maxtime),
                             NormalSampler(3 / 8 * self.maxaddr, .05 * self.maxaddr, 1, self.maxaddr),
                             attack=True),
        ]
