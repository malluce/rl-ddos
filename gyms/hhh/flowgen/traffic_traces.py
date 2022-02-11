from abc import ABC, abstractmethod
import gin
from gyms.hhh.flowgen.distgen import FlowGroupSampler, NormalSampler, UniformSampler, WeibullSampler


# initially planned to use traces with different source, rate, and change patterns that would be identified
# in the end only used the class name to identify each single trace
class IdentifiableTrace(ABC):
    @abstractmethod
    def get_source_pattern_id(self, time_step):
        pass

    @abstractmethod
    def get_rate_pattern_id(self, time_step):
        pass

    @abstractmethod
    def get_change_pattern_id(self):
        pass


@gin.configurable
class SamplerTrafficTrace(IdentifiableTrace):
    def __init__(self, maxtime):
        self.maxtime = maxtime

    @abstractmethod
    def get_flow_group_samplers(self):
        pass

    def get_max_time(self):
        return self.maxtime

    # simply return class name for fix traces
    def get_source_pattern_id(self, time_step):
        return self.__class__.__name__

    def get_rate_pattern_id(self, time_step):
        return self.__class__.__name__

    def get_change_pattern_id(self):
        return self.__class__.__name__


@gin.configurable
class S1(SamplerTrafficTrace):
    """Scenario S1 in thesis (simple transition from botnet to reflector attack)"""

    def __init__(self, num_benign=300, num_attack=150, maxtime=599, maxaddr=0xffff, **kwargs):
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
                             UniformSampler(0.4 * self.maxtime, 0.5 * self.maxtime),
                             UniformSampler(0.05, 0.35 * self.maxaddr),
                             attack=True),
            # direct attack 2
            FlowGroupSampler(self.num_attack // 5,
                             UniformSampler(0, 1),
                             UniformSampler(0.4 * self.maxtime, 0.5 * self.maxtime),
                             UniformSampler(0.55 * self.maxaddr, 0.85 * self.maxaddr),
                             attack=True),
            # 1st refl/ampl
            FlowGroupSampler(self.num_attack // 5,
                             UniformSampler(0.6 * self.maxtime, 0.65 * self.maxtime + 1),
                             UniformSampler(0.9 * self.maxtime, self.maxtime),
                             UniformSampler(0.3 * self.maxaddr, 0.32 * self.maxaddr),
                             rate_sampler=UniformSampler(1, 3),
                             attack=True),
            # 2nd refl/ampl
            FlowGroupSampler(self.num_attack // 5,
                             UniformSampler(0.6 * self.maxtime, 0.65 * self.maxtime + 1),
                             UniformSampler(0.9 * self.maxtime, self.maxtime),
                             UniformSampler(0.8 * self.maxaddr, 0.82 * self.maxaddr),
                             rate_sampler=UniformSampler(1, 3),
                             attack=True),
            # 3rd refl/ampl
            FlowGroupSampler(self.num_attack // 5,
                             UniformSampler(0.6 * self.maxtime, 0.65 * self.maxtime + 1),
                             UniformSampler(0.9 * self.maxtime, self.maxtime),
                             UniformSampler(0.1 * self.maxaddr, 0.12 * self.maxaddr),
                             rate_sampler=UniformSampler(1, 3),
                             attack=True)
        ]


@gin.configurable
class S2(SamplerTrafficTrace):
    """Scenario S2 in thesis (botnet attacks with varying overlap)"""

    def __init__(self, num_benign=500, **kwargs):
        self.maxtime = 599
        self.maxaddr = 0xffff
        super().__init__(self.maxtime)
        self.num_benign = num_benign
        self.num_bot = 30

        self.fgs = [
            # 1st set of benign flows
            FlowGroupSampler(self.num_benign,
                             UniformSampler(0, 0.95 * self.maxtime),
                             WeibullSampler(3 / 2,
                                            (1 / WeibullSampler.quantile(99, 3 / 2)) * 1 / 8 * self.maxtime),
                             NormalSampler(1 / 2 * self.maxaddr, .15 * self.maxaddr, 1, self.maxaddr),
                             attack=False),
            # 1st Botnet (high overlap)
            FlowGroupSampler(self.num_bot * 12,
                             UniformSampler(0, 99),
                             WeibullSampler(3,
                                            (1 / WeibullSampler.quantile(99.99, 3)) * 1 / 8 * self.maxtime),
                             NormalSampler(0.45 * self.maxaddr, .05 * self.maxaddr, 0, self.maxaddr),
                             attack=True),
            # 2nd Botnet (medium overlap)
            FlowGroupSampler(self.num_bot * 12,
                             UniformSampler(110, 209),
                             WeibullSampler(3,
                                            (1 / WeibullSampler.quantile(99.99, 3)) * 1 / 8 * self.maxtime),
                             NormalSampler(0.6 * self.maxaddr, .05 * self.maxaddr, min=0, max=self.maxaddr),
                             attack=True),
            # 3rd Botnet (low overlap)
            FlowGroupSampler(self.num_bot * 12,
                             UniformSampler(220, 319),
                             WeibullSampler(3,
                                            (1 / WeibullSampler.quantile(99.99, 3)) * 1 / 8 * self.maxtime),
                             NormalSampler(0.75 * self.maxaddr, 0.05 * self.maxaddr, min=0, max=self.maxaddr),
                             attack=True),
            # 4th Botnet (high intensity, lowest overlap)
            FlowGroupSampler(self.num_bot * 40,  # previously * 12
                             UniformSampler(330, 519),
                             WeibullSampler(3,
                                            (1 / WeibullSampler.quantile(99.99, 3)) * 1 / 8 * self.maxtime),
                             NormalSampler(0.9 * self.maxaddr, 0.03 * self.maxaddr, min=0, max=self.maxaddr),
                             attack=True),
            ## 5th Botnet (high intensity, lowest overlap)
            # FlowGroupSampler(self.num_bot * 20,  # previously * 12
            #                 UniformSampler(330, 519),
            #                 WeibullSampler(3,
            #                                (1 / WeibullSampler.quantile(99.99, 3)) * 1 / 8 * self.maxtime),
            #                 NormalSampler(0.1 * self.maxaddr, 0.03 * self.maxaddr, min=0, max=self.maxaddr),
            #                 attack=True),
            # 6th Botnet (medium overlap 2)
            FlowGroupSampler(self.num_bot * 12,
                             UniformSampler(500, 589),
                             WeibullSampler(3,
                                            (1 / WeibullSampler.quantile(99.99, 3)) * 1 / 8 * self.maxtime),
                             NormalSampler(0.5 * self.maxaddr, 0.05 * self.maxaddr, min=0, max=self.maxaddr),
                             attack=True),
        ]

    def get_flow_group_samplers(self):
        return self.fgs


@gin.configurable
class S3(SamplerTrafficTrace):
    """Scenario S3 in thesis (multi-vector phases, reflector and bot at same time)"""

    def __init__(self, num_benign=500, num_dns=10, **kwargs):
        self.maxtime = 599
        self.maxaddr = 0xffff
        super().__init__(self.maxtime)
        self.num_benign = num_benign
        self.num_dns = num_dns
        self.num_bot = self.num_dns * 3
        self._total_rate = self.num_dns * 30

        self.fgs = [
            # 1st set of benign flows (done)
            FlowGroupSampler(self.num_benign,
                             UniformSampler(0, 0.95 * self.maxtime),
                             WeibullSampler(3 / 2,
                                            (1 / WeibullSampler.quantile(99, 3 / 2)) * 1 / 8 * self.maxtime),
                             NormalSampler(1 / 2 * self.maxaddr, .17 * self.maxaddr, 1, self.maxaddr),
                             attack=False),
            # 1st Botnet (done)
            FlowGroupSampler(self.num_bot * 10,
                             UniformSampler(80, 149),
                             WeibullSampler(3,
                                            (1 / WeibullSampler.quantile(99.99, 3)) * 1 / 8 * self.maxtime),
                             NormalSampler(0.45 * self.maxaddr, .05 * self.maxaddr, 0, self.maxaddr),
                             attack=True),
            # 1st DNS Reflection (done)
            FlowGroupSampler(self.num_dns,
                             UniformSampler(0, 9),
                             WeibullSampler(20,
                                            (1 / WeibullSampler.quantile(99.99, 20)) * 1 / 3 * self.maxtime),
                             UniformSampler(0, self.maxaddr),
                             rate_sampler=UniformSampler(self._total_rate // self.num_dns,
                                                         self._total_rate // self.num_dns),
                             attack=True),
            # 2nd DNS Reflection (done)
            FlowGroupSampler(self.num_dns // 2,  # self.num_dns * 2,
                             UniformSampler(280, 289),
                             UniformSampler(600, 600),
                             UniformSampler(0.55 * self.maxaddr, 0.95 * self.maxaddr),
                             rate_sampler=UniformSampler(self._total_rate // self.num_dns,  # / 3,  # / 6,
                                                         self._total_rate // self.num_dns),  # /3  # / 6),
                             attack=True),
            # 2nd Botnet (done)
            FlowGroupSampler(self.num_bot * 20,
                             UniformSampler(160, 329),
                             WeibullSampler(3,
                                            (1 / WeibullSampler.quantile(99.99, 3)) * 1 / 8 * self.maxtime),
                             NormalSampler(0.2 * self.maxaddr, 0.09 * self.maxaddr, min=0, max=self.maxaddr),
                             attack=True),
            # 3rd Botnet (done)
            FlowGroupSampler(self.num_bot * 15,
                             UniformSampler(350, 429),
                             WeibullSampler(3,
                                            (1 / WeibullSampler.quantile(99.99, 3)) * 1 / 8 * self.maxtime),
                             NormalSampler(0.15 * self.maxaddr, 0.02 * self.maxaddr, min=0, max=self.maxaddr),
                             attack=True),
            # 4th Botnet (3 groups)
            FlowGroupSampler(self.num_bot * 8,
                             UniformSampler(460, 569),
                             WeibullSampler(3,
                                            (1 / WeibullSampler.quantile(99.99, 3)) * 1 / 8 * self.maxtime),
                             NormalSampler(0.25 * self.maxaddr, 0.02 * self.maxaddr, min=0, max=self.maxaddr),
                             attack=True),
            FlowGroupSampler(self.num_bot * 8,
                             UniformSampler(460, 569),
                             WeibullSampler(3,
                                            (1 / WeibullSampler.quantile(99.99, 3)) * 1 / 8 * self.maxtime),
                             NormalSampler(0.5 * self.maxaddr, 0.02 * self.maxaddr, min=0, max=self.maxaddr),
                             attack=True),
            FlowGroupSampler(self.num_bot * 8,
                             UniformSampler(460, 569),
                             WeibullSampler(3,
                                            (1 / WeibullSampler.quantile(99.99, 3)) * 1 / 8 * self.maxtime),
                             NormalSampler(0.75 * self.maxaddr, 0.02 * self.maxaddr, min=0, max=self.maxaddr),
                             attack=True),
        ]

    def get_flow_group_samplers(self):
        return self.fgs
