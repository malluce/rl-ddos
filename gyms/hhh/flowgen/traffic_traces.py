import math
from abc import ABC, abstractmethod

import gin

from gyms.hhh.flowgen.distgen import FlowGroupSampler, NormalSampler, UniformSampler, WeibullSampler


@gin.configurable
class SamplerTrafficTrace(ABC):
    def __init__(self, maxtime):
        self.maxtime = maxtime

    @abstractmethod
    def get_flow_group_samplers(self):
        pass

    def get_max_time(self):
        return self.maxtime


@gin.configurable
class T1(SamplerTrafficTrace):
    def __init__(self, num_benign=25, num_attack=50, maxtime=600, maxaddr=0xffff):
        super().__init__(maxtime)
        self.num_benign = num_benign
        self.num_attack = num_attack
        self.maxtime = maxtime
        assert maxaddr == 0xffff

    def get_flow_group_samplers(self):
        return [
            # 1st set of benign flows
            FlowGroupSampler(self.num_benign,
                             UniformSampler(0, 1),
                             UniformSampler(self.maxtime, self.maxtime + 1),
                             UniformSampler(0x000, 0x7fff),
                             attack=False),
            # 1st set of attack flows
            FlowGroupSampler(self.num_attack,
                             UniformSampler(0, 1),
                             UniformSampler(self.maxtime, self.maxtime + 1),  #
                             UniformSampler(0x8000, 0xffff),
                             attack=True)
        ]


@gin.configurable
class T2(SamplerTrafficTrace):

    def __init__(self, num_benign=50, num_attack=100, maxtime=600, maxaddr=0xffff):
        super().__init__(maxtime)
        self.num_benign = num_benign
        self.num_attack = num_attack
        self.maxtime = maxtime
        assert maxaddr == 0xffff

    def get_flow_group_samplers(self):
        return [
            # 1st set of benign flows
            FlowGroupSampler(self.num_benign,
                             UniformSampler(0, 1),
                             UniformSampler(self.maxtime, self.maxtime + 1),
                             UniformSampler(0x000, 0x7fff),
                             attack=False),
            # 1st set of attack flows
            FlowGroupSampler(self.num_attack,
                             UniformSampler(0, 1),
                             UniformSampler(self.maxtime / 4, self.maxtime / 2),  #
                             UniformSampler(0x8000, 0xffff),
                             attack=True)
        ]


@gin.configurable
class T3(SamplerTrafficTrace):

    def __init__(self, num_benign=300, num_attack=150, maxtime=600, maxaddr=0xffff):
        super().__init__(maxtime)
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
class T4(SamplerTrafficTrace):

    def __init__(self, num_benign=300, num_attack=150, maxtime=600, maxaddr=0xffff):
        super().__init__(maxtime)
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
            # direct attack 1
            FlowGroupSampler(self.num_attack // 5,
                             UniformSampler(0, 1),
                             UniformSampler(0.3 * self.maxtime, 0.6 * self.maxtime),
                             UniformSampler(0.05, 0.35 * self.maxaddr),
                             attack=True),
            # direct attack 2
            FlowGroupSampler(self.num_attack // 5,
                             UniformSampler(0, 1),
                             UniformSampler(0.3 * self.maxtime, 0.6 * self.maxtime),
                             UniformSampler(0.55 * self.maxaddr, 0.85 * self.maxaddr),
                             attack=True),
            # 1st refl/ampl
            FlowGroupSampler(self.num_attack // 5,
                             UniformSampler(0.6 * self.maxtime, 0.65 * self.maxtime + 1),
                             UniformSampler(0.9 * self.maxtime, self.maxtime),
                             UniformSampler(0.3 * self.maxaddr, 0.32 * self.maxaddr),
                             attack=True),
            # 2nd refl/ampl
            FlowGroupSampler(self.num_attack // 5,
                             UniformSampler(0.6 * self.maxtime, 0.65 * self.maxtime + 1),
                             UniformSampler(0.9 * self.maxtime, self.maxtime),
                             UniformSampler(0.8 * self.maxaddr, 0.82 * self.maxaddr),
                             attack=True),
            # 3rd refl/ampl
            FlowGroupSampler(self.num_attack // 5,
                             UniformSampler(0.6 * self.maxtime, 0.65 * self.maxtime + 1),
                             UniformSampler(0.9 * self.maxtime, self.maxtime),
                             UniformSampler(0.1 * self.maxaddr, 0.12 * self.maxaddr),
                             attack=True)
        ]


@gin.configurable
class THauke(SamplerTrafficTrace):

    def __init__(self, benign_flows, attack_flows, maxtime, maxaddr):
        super().__init__(maxtime)
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
