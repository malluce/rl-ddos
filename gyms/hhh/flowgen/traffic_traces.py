import math
import time
from abc import ABC, abstractmethod
from typing import List, Tuple

import gin
import numpy as np
from ipaddress import IPv4Address
from numpy.random import default_rng

from gyms.hhh.flowgen.distgen import ChoiceSampler, FlowGroupSampler, NormalSampler, UniformSampler, WeibullSampler
from gyms.hhh.label import Label
from gyms.hhh.loop import Loop


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
class T1(SamplerTrafficTrace):

    def __init__(self, num_benign=25, num_attack=50, maxtime=599, maxaddr=0xffff, **kwargs):
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

    def __init__(self, num_benign=50, num_attack=100, maxtime=599, maxaddr=0xffff, **kwargs):
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
class T3WithoutPause(SamplerTrafficTrace):

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
            # 1st set of attack flows
            FlowGroupSampler(2 * self.num_attack // 3,
                             UniformSampler(0, 1),
                             UniformSampler(0.4 * self.maxtime, 0.45 * self.maxtime),
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
class T4(SamplerTrafficTrace):  # scenario "S1" in thesis (simple adaptivity)

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
class T5(SamplerTrafficTrace):  # scenario "S2" in thesis ("wandering" bot attack, overlap test)

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
            # 1st Botnet
            FlowGroupSampler(self.num_bot * 12,
                             UniformSampler(0, 99),
                             WeibullSampler(3,
                                            (1 / WeibullSampler.quantile(99.99, 3)) * 1 / 8 * self.maxtime),
                             NormalSampler(0.45 * self.maxaddr, .05 * self.maxaddr, 0, self.maxaddr),
                             attack=True),
            # 2nd Botnet
            FlowGroupSampler(self.num_bot * 12,
                             UniformSampler(110, 209),
                             WeibullSampler(3,
                                            (1 / WeibullSampler.quantile(99.99, 3)) * 1 / 8 * self.maxtime),
                             NormalSampler(0.6 * self.maxaddr, .05 * self.maxaddr, min=0, max=self.maxaddr),
                             attack=True),
            # 3rd Botnet
            FlowGroupSampler(self.num_bot * 12,
                             UniformSampler(220, 319),
                             WeibullSampler(3,
                                            (1 / WeibullSampler.quantile(99.99, 3)) * 1 / 8 * self.maxtime),
                             NormalSampler(0.75 * self.maxaddr, 0.05 * self.maxaddr, min=0, max=self.maxaddr),
                             attack=True),
            # 4th Botnet
            FlowGroupSampler(self.num_bot * 20,  # previously * 12
                             UniformSampler(330, 519),
                             WeibullSampler(3,
                                            (1 / WeibullSampler.quantile(99.99, 3)) * 1 / 8 * self.maxtime),
                             NormalSampler(0.9 * self.maxaddr, 0.03 * self.maxaddr, min=0, max=self.maxaddr),
                             attack=True),
            # 5th Botnet
            FlowGroupSampler(self.num_bot * 20,  # previously * 12
                             UniformSampler(330, 519),
                             WeibullSampler(3,
                                            (1 / WeibullSampler.quantile(99.99, 3)) * 1 / 8 * self.maxtime),
                             NormalSampler(0.1 * self.maxaddr, 0.03 * self.maxaddr, min=0, max=self.maxaddr),
                             attack=True),
            # 6th Botnet
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
class T6(SamplerTrafficTrace):  # scenario "S3" in thesis (reflector and bot at same time)

    def __init__(self, num_benign=500, num_dns=10, **kwargs):
        self.maxtime = 599
        self.maxaddr = 0xffff
        super().__init__(self.maxtime)
        self.num_benign = num_benign
        self.num_dns = num_dns
        self.num_bot = self.num_dns * 3
        self._total_rate = self.num_dns * 30

        self.fgs = [
            # 1st set of benign flows
            FlowGroupSampler(self.num_benign,
                             UniformSampler(0, 0.95 * self.maxtime),
                             WeibullSampler(3 / 2,
                                            (1 / WeibullSampler.quantile(99, 3 / 2)) * 1 / 8 * self.maxtime),
                             NormalSampler(1 / 2 * self.maxaddr, .17 * self.maxaddr, 1, self.maxaddr),
                             attack=False),
            # 1st Botnet
            FlowGroupSampler(self.num_bot * 10,
                             UniformSampler(0, 159),
                             WeibullSampler(3,
                                            (1 / WeibullSampler.quantile(99.99, 3)) * 1 / 8 * self.maxtime),
                             NormalSampler(0.45 * self.maxaddr, .05 * self.maxaddr, 0, self.maxaddr),
                             attack=True),
            # 1st DNS Reflection
            FlowGroupSampler(self.num_dns,
                             UniformSampler(100, 109),
                             WeibullSampler(20,
                                            (1 / WeibullSampler.quantile(99.99, 20)) * 1 / 3 * self.maxtime),
                             UniformSampler(0, self.maxaddr),  # TODO DNS addresses?
                             rate_sampler=UniformSampler(self._total_rate // self.num_dns,
                                                         self._total_rate // self.num_dns),
                             attack=True),
            # 2nd DNS Reflection
            FlowGroupSampler(self.num_dns // 2,
                             UniformSampler(360, 369),
                             UniformSampler(600, 600),
                             UniformSampler(0.55 * self.maxaddr, 0.95 * self.maxaddr),  # TODO DNS addresses?
                             rate_sampler=UniformSampler(self._total_rate // self.num_dns,
                                                         self._total_rate // self.num_dns),
                             attack=True),
            # 2nd Botnet
            FlowGroupSampler(self.num_bot * 20,
                             UniformSampler(270, 439),
                             WeibullSampler(3,
                                            (1 / WeibullSampler.quantile(99.99, 3)) * 1 / 10 * self.maxtime),
                             NormalSampler(0.2 * self.maxaddr, 0.09 * self.maxaddr, min=0, max=self.maxaddr),
                             attack=True),
            # 3rd Botnet
            FlowGroupSampler(self.num_bot * 15,
                             UniformSampler(460, 569),
                             WeibullSampler(3,
                                            (1 / WeibullSampler.quantile(99.99, 3)) * 1 / 10 * self.maxtime),
                             NormalSampler(0.2 * self.maxaddr, 0.02 * self.maxaddr, min=0, max=self.maxaddr),
                             attack=True),
        ]

    def get_flow_group_samplers(self):
        return self.fgs


@gin.configurable
class HafnerT1(SamplerTrafficTrace):

    def __init__(self, benign_flows=50, maxtime=399, **kwargs):
        super().__init__(maxtime)
        self.benign_flows = benign_flows
        self.attack_flows = 4 * self.benign_flows  # ratio of attack to benign traffic 4:1
        self.maxtime = maxtime

    def get_flow_group_samplers(self):
        return [
            FlowGroupSampler(self.attack_flows,
                             UniformSampler(0, 0),
                             UniformSampler(self.maxtime, self.maxtime),
                             UniformSampler(0, 0x7fff),
                             attack=True),
            FlowGroupSampler(self.benign_flows,
                             UniformSampler(0, 0),
                             UniformSampler(self.maxtime, self.maxtime),
                             UniformSampler(0x8000, 0xffff),
                             attack=False)
        ]


@gin.configurable
class HafnerT2(SamplerTrafficTrace):

    def __init__(self, benign_flows=50, maxtime=399, **kwargs):
        super().__init__(maxtime)
        self.benign_flows = benign_flows
        self.attack_flows = 4 * self.benign_flows  # ratio of attack to benign traffic 4:1
        self.maxtime = maxtime

    def get_flow_group_samplers(self):
        return [
            FlowGroupSampler(self.benign_flows,
                             UniformSampler(0, 0),
                             UniformSampler(self.maxtime, self.maxtime),
                             UniformSampler(0, 0xffff),
                             attack=False),
            FlowGroupSampler(self.attack_flows,
                             UniformSampler(0, 0),
                             UniformSampler(self.maxtime, self.maxtime),
                             ChoiceSampler(
                                 list(range(0x0000, 0x00ff + 1)) +  # 0.0.0.0/24
                                 list(range(0x1000, 0x10ff + 1)) +  # 0.0.16.0/24
                                 list(range(0x2000, 0x20ff + 1)) +  # 0.0.32.0/24
                                 list(range(0x3000, 0x30ff + 1)) +  # 0.0.48.0/24
                                 list(range(0x4000, 0x40ff + 1)) +  # 0.0.64.0/24
                                 list(range(0x8000, 0x80ff + 1)) +  # 0.0.128.0/24
                                 list(range(0xA000, 0xA0ff + 1)) +  # 0.0.160.0/24
                                 list(range(0xFF00, 0xFFff + 1)),  # 0.0.255.0/24
                                 replace=True
                             ),
                             attack=True),
        ]


@gin.configurable
class HafnerRandomSwitch(SamplerTrafficTrace):

    def __init__(self, benign_flows=50, maxtime=399, is_eval=False):
        super().__init__(maxtime)
        self.benign_flows = benign_flows
        self.t1_fgs = HafnerT1(benign_flows=benign_flows, maxtime=maxtime).get_flow_group_samplers()
        self.t2_fgs = HafnerT2(benign_flows=benign_flows, maxtime=maxtime).get_flow_group_samplers()
        self.current_fgs = None
        self.is_eval = is_eval
        if self.is_eval:
            self.current_fgs = self.t1_fgs

    def get_flow_group_samplers(self):
        if self.is_eval:
            # toggle the two fgs
            if self.current_fgs == self.t1_fgs:
                self.current_fgs = self.t2_fgs
                print('T1 eval')
                return self.t1_fgs
            elif self.current_fgs == self.t2_fgs:
                self.current_fgs = self.t1_fgs
                print('T2 eval')
                return self.t2_fgs
            else:
                raise ValueError('No fgs set in eval.')
        # with probability 0.5 select one of the fgs
        if default_rng().random() < 0.5:
            print('T1')
            self.current_fgs = self.t1_fgs
        else:
            print('T2')
            self.current_fgs = self.t2_fgs
        return self.current_fgs

    def get_source_pattern_id(self, time_step):
        if self.is_eval:  # due to toggling return opposite id
            if self.current_fgs == self.t1_fgs:
                return 'HafnT2'
            elif self.current_fgs == self.t2_fgs:
                return 'HafnT1'
        else:
            if self.current_fgs == self.t1_fgs:
                return 'HafnT1'
            elif self.current_fgs == self.t2_fgs:
                return 'HafnT2'
            else:
                raise ValueError('No flow group sampler set, call get_flow_group_samplers() first!')


@gin.register
class BotTrace(SamplerTrafficTrace):

    def __init__(self, benign_flows=200, attack_flows=50, maxtime=599, maxaddr=0xffff, is_eval=False):
        super().__init__(maxtime)
        self.attack_flows = attack_flows
        self.benign_flows = benign_flows
        self.maxtime = maxtime
        self.maxaddr = maxaddr
        self.interval = 10

        self.benign_fgs = [FlowGroupSampler(self.benign_flows,
                                            UniformSampler(0 - self.interval, self.maxtime - self.interval),
                                            WeibullSampler(3 / 2,
                                                           (1 / WeibullSampler.quantile(95,
                                                                                        3 / 2)) * 1 / 8 * self.maxtime),
                                            UniformSampler(0, self.maxaddr),
                                            attack=False
                                            )]

        bot = BotnetSourcePattern(subnet=22).generate_addresses(int(self.attack_flows))
        self.bot_fgs = [FlowGroupSampler(num, UniformSampler(0, 0),
                                         UniformSampler(self.maxtime, self.maxtime),
                                         ChoiceSampler(addr, replace=False),
                                         rate_sampler=None,
                                         attack=True)
                        for
                        num, addr in
                        bot.values()]

    def get_flow_group_samplers(self):
        return self.benign_fgs + self.bot_fgs


@gin.register
class BotSSDPTrace(SamplerTrafficTrace):

    def __init__(self, benign_flows=200, attack_flows=50, maxtime=599, maxaddr=0xffff, is_eval=False):
        super().__init__(maxtime)
        self.attack_flows = attack_flows
        self.benign_flows = benign_flows
        self.maxtime = maxtime
        self.maxaddr = maxaddr
        self.interval = 10

        self.benign_fgs = [FlowGroupSampler(self.benign_flows,
                                            UniformSampler(0 - self.interval, self.maxtime - self.interval),
                                            WeibullSampler(3 / 2,
                                                           (1 / WeibullSampler.quantile(95,
                                                                                        3 / 2)) * 1 / 8 * self.maxtime),
                                            UniformSampler(0, self.maxaddr),
                                            attack=False
                                            )]

        bot = BotnetSourcePattern(subnet=22).generate_addresses(int(self.attack_flows))
        self.bot_fgs = [FlowGroupSampler(num, UniformSampler(0, 0),
                                         UniformSampler(self.maxtime, self.maxtime),
                                         ChoiceSampler(addr, replace=False),
                                         rate_sampler=None,
                                         attack=True)
                        for
                        num, addr in
                        bot.values()]

        address_space = round(math.log2(self.maxaddr))
        ssdp = ReflectorSourcePattern('ssdp', address_space).generate_addresses(100)
        self.ssdp_fgs = [FlowGroupSampler(num, UniformSampler(0, 0),
                                          UniformSampler(self.maxtime, self.maxtime),
                                          ChoiceSampler(addr, replace=False),
                                          rate_sampler=UniformSampler(
                                              self.attack_flows * 4 // list(ssdp.values())[0][0],
                                              self.attack_flows * 4 // list(ssdp.values())[0][0]),
                                          attack=True)
                         for
                         num, addr in
                         ssdp.values()]

        self.use_bot = False

    def get_flow_group_samplers(self):
        if self.use_bot:
            fgs = self.benign_fgs + self.bot_fgs
        else:
            fgs = self.benign_fgs + self.ssdp_fgs
        self.use_bot = not self.use_bot
        return fgs


@gin.register
class NTPTrace(SamplerTrafficTrace):

    def __init__(self, benign_flows=200, attack_flows=50, maxtime=599, maxaddr=0xffff, is_eval=False):
        super().__init__(maxtime)
        self.attack_flows = attack_flows
        self.benign_flows = benign_flows
        self.maxtime = maxtime
        self.maxaddr = maxaddr
        self.interval = 10

        self.benign_fgs = [FlowGroupSampler(self.benign_flows,
                                            UniformSampler(0 - self.interval, self.maxtime - self.interval),
                                            WeibullSampler(3 / 2,
                                                           (1 / WeibullSampler.quantile(95,
                                                                                        3 / 2)) * 1 / 8 * self.maxtime),
                                            UniformSampler(0, self.maxaddr),
                                            attack=False
                                            )]
        address_space = round(math.log2(self.maxaddr))
        ntp = ReflectorSourcePattern('ntp', address_space).generate_addresses(1)
        self.ntp_fgs = [FlowGroupSampler(num, UniformSampler(0, 0),
                                         UniformSampler(self.maxtime, self.maxtime),
                                         ChoiceSampler(addr, replace=False),
                                         rate_sampler=UniformSampler(self.attack_flows * 2 // list(ntp.values())[0][0],
                                                                     self.attack_flows * 2 // list(ntp.values())[0][
                                                                         0]),
                                         attack=True)
                        for
                        num, addr in
                        ntp.values()]

    def get_flow_group_samplers(self):
        return self.benign_fgs + self.ntp_fgs


@gin.register
class SSDPTrace(SamplerTrafficTrace):

    def __init__(self, benign_flows=200, attack_flows=50, maxtime=599, maxaddr=0xffff, is_eval=False):
        super().__init__(maxtime)
        self.attack_flows = attack_flows
        self.benign_flows = benign_flows
        self.maxtime = maxtime
        self.maxaddr = maxaddr
        self.interval = 10

        self.benign_fgs = [FlowGroupSampler(self.benign_flows,
                                            UniformSampler(0 - self.interval, self.maxtime - self.interval),
                                            WeibullSampler(3 / 2,
                                                           (1 / WeibullSampler.quantile(95,
                                                                                        3 / 2)) * 1 / 8 * self.maxtime),
                                            UniformSampler(0, self.maxaddr),
                                            attack=False
                                            )]
        address_space = round(math.log2(self.maxaddr))
        ssdp = ReflectorSourcePattern('ssdp', address_space).generate_addresses(100)
        self.ssdp_fgs = [FlowGroupSampler(num, UniformSampler(0, 0),
                                          UniformSampler(self.maxtime, self.maxtime),
                                          ChoiceSampler(addr, replace=False),
                                          rate_sampler=UniformSampler(
                                              self.attack_flows * 4 // list(ssdp.values())[0][0],
                                              self.attack_flows * 4 // list(ssdp.values())[0][0]),
                                          attack=True)
                         for
                         num, addr in
                         ssdp.values()]

    def get_flow_group_samplers(self):
        return self.benign_fgs + self.ssdp_fgs


@gin.register
class MixedSSDPBot(SamplerTrafficTrace):
    def __init__(self, benign_flows=200, attack_flows=50, maxtime=599, maxaddr=0xffff, is_eval=False):
        super().__init__(maxtime)
        self.attack_flows = attack_flows
        self.benign_flows = benign_flows
        self.maxtime = maxtime
        self.maxaddr = maxaddr
        self.interval = 10

        self.benign_fgs = [FlowGroupSampler(self.benign_flows,
                                            UniformSampler(0 - self.interval, self.maxtime - self.interval),
                                            WeibullSampler(3 / 2,
                                                           (1 / WeibullSampler.quantile(95,
                                                                                        3 / 2)) * 1 / 8 * self.maxtime),
                                            UniformSampler(0, self.maxaddr),
                                            attack=False
                                            )]

        bot = BotnetSourcePattern(address_space=15, subnet=22).generate_addresses(int(self.attack_flows / 2))

        ssdp = ReflectorSourcePattern('ssdp', address_space=16, start_address=2 ** 15).generate_addresses(
            int(self.attack_flows / 2))
        self.bot_fgs = [FlowGroupSampler(num, UniformSampler(0, 0),
                                         UniformSampler(self.maxtime, self.maxtime),
                                         ChoiceSampler(addr, replace=False),
                                         rate_sampler=None,
                                         attack=True)
                        for
                        num, addr in
                        bot.values()]

        self.ssdp_fgs = [FlowGroupSampler(num, UniformSampler(0, 0),
                                          UniformSampler(self.maxtime, self.maxtime),
                                          ChoiceSampler(addr, replace=False),
                                          rate_sampler=UniformSampler(
                                              self.attack_flows / 2 * 4 // list(ssdp.values())[0][0],
                                              self.attack_flows / 2 * 4 // list(ssdp.values())[0][0]),
                                          attack=True)
                         for
                         num, addr in
                         ssdp.values()]

    def get_flow_group_samplers(self):
        return self.benign_fgs + self.ssdp_fgs + self.bot_fgs

    def get_change_pattern_id(self):
        return 'None'

    def get_rate_pattern_id(self, time_step):
        return 'constant-rate'

    def get_source_pattern_id(self, time_step):
        return 'ssdp;bot'


@gin.register
class MixedNTPBot(SamplerTrafficTrace):
    def __init__(self, benign_flows=200, attack_flows=50, maxtime=599, maxaddr=0xffff, is_eval=False):
        super().__init__(maxtime)
        self.attack_flows = attack_flows
        self.benign_flows = benign_flows
        self.maxtime = maxtime
        self.maxaddr = maxaddr
        self.interval = 10

        self.benign_fgs = [FlowGroupSampler(self.benign_flows,
                                            UniformSampler(0 - self.interval, self.maxtime - self.interval),
                                            WeibullSampler(3 / 2,
                                                           (1 / WeibullSampler.quantile(95,
                                                                                        3 / 2)) * 1 / 8 * self.maxtime),
                                            UniformSampler(0, self.maxaddr),
                                            attack=False
                                            )]

        bot = BotnetSourcePattern(address_space=15, subnet=22).generate_addresses(int(self.attack_flows / 2))

        ntp = ReflectorSourcePattern('ntp', address_space=16, start_address=2 ** 15).generate_addresses(1)
        self.bot_fgs = [FlowGroupSampler(num, UniformSampler(0, 0),
                                         UniformSampler(self.maxtime, self.maxtime),
                                         ChoiceSampler(addr, replace=False),
                                         rate_sampler=None,
                                         attack=True)
                        for
                        num, addr in
                        bot.values()]
        self.ntp_fgs = [FlowGroupSampler(num, UniformSampler(0, 0),
                                         UniformSampler(self.maxtime, self.maxtime),
                                         ChoiceSampler(addr, replace=False),
                                         rate_sampler=UniformSampler(
                                             self.attack_flows // list(ntp.values())[0][0],
                                             self.attack_flows // list(ntp.values())[0][0]),
                                         attack=True)
                        for
                        num, addr in
                        ntp.values()]

    def get_flow_group_samplers(self):
        return self.benign_fgs + self.ntp_fgs + self.bot_fgs

    def get_change_pattern_id(self):
        return 'None'

    def get_rate_pattern_id(self, time_step):
        return 'constant-rate'

    def get_source_pattern_id(self, time_step):
        return 'ntp;bot'


@gin.configurable
class TRandomPatternSwitch(SamplerTrafficTrace):
    SMOOTH_TRANSITION_OFFSET_INTERVAL = 10  # number of intervals before and after toggle for smooth transition

    def __init__(self, benign_flows=200, attack_flows=50, maxtime=599, maxaddr=0xffff, is_eval=False,
                 random_toggle_time=False, smooth_transition=False, benign_normal=False):
        super().__init__(maxtime)
        self.attack_flows = attack_flows
        self.benign_flows = benign_flows
        self.maxtime = maxtime
        self.maxaddr = maxaddr
        self.deterministic_cycle = is_eval  # cycle all possible pattern switches in eval or sample them in train
        self.interval = Loop.ACTION_INTERVAL

        if self.deterministic_cycle:
            self.current_pattern_combination = 0  # idx for possible pattern combinations

        benign_addr_sampler = NormalSampler(0.5 * self.maxaddr, .15 * self.maxaddr, min=0,
                                            max=self.maxaddr) if benign_normal else UniformSampler(0, self.maxaddr)

        self.benign_fgs = FlowGroupSampler(self.benign_flows,
                                           UniformSampler(0 - self.interval, self.maxtime - self.interval),
                                           WeibullSampler(3 / 2,
                                                          (1 / WeibullSampler.quantile(95,
                                                                                       3 / 2)) * 1 / 8 * self.maxtime),
                                           benign_addr_sampler,
                                           attack=False
                                           )

        self.random_toggle_time = random_toggle_time  # whether to toggle patterns at random point (otherwise at half)
        self.smooth_transition = smooth_transition  # whether to transition smoothly (ramp-up/ramp-down) or hard toggle
        self.smooth_transition_offset = self.SMOOTH_TRANSITION_OFFSET_INTERVAL * self.interval

    def get_flow_group_samplers(self):
        address_space = round(math.log2(self.maxaddr))

        def botnet():
            return BotnetSourcePattern(address_space, subnet=22).generate_addresses(self.attack_flows)

        def ntp_reflection():
            return ReflectorSourcePattern('ntp', address_space).generate_addresses(1)

        def ssdp_reflection():
            return ReflectorSourcePattern('ssdp', address_space).generate_addresses(100)

        if not self.deterministic_cycle:
            # sample patterns
            used_patterns = default_rng().choice([botnet, ntp_reflection, ssdp_reflection], 2, replace=True)
        else:
            # return combinations of patterns
            all_combinations = [(x, y) for x in [botnet, ntp_reflection, ssdp_reflection]
                                for y in [botnet, ntp_reflection, ssdp_reflection]]
            used_patterns = all_combinations[self.current_pattern_combination]
            self.current_pattern_combination = (self.current_pattern_combination + 1) % len(all_combinations)

        first_pattern = used_patterns[0]()
        second_pattern = used_patterns[1]()

        self.first_source_pattern_id = self._get_source_pattern_id_for_attack_pattern(first_pattern)
        self.second_source_pattern_id = self._get_source_pattern_id_for_attack_pattern(second_pattern)

        fgs = [self.benign_fgs]

        trace_duration = self.maxtime + 1

        if self.random_toggle_time:
            self.toggle_time = default_rng().uniform(15 * self.interval, trace_duration - 15 * self.interval)
        else:
            self.toggle_time = trace_duration / 2

        if self.smooth_transition:
            first_duration_sampler = NormalSampler(self.toggle_time - 1, .05 * self.maxtime,
                                                   min=self.toggle_time - 1 - self.smooth_transition_offset,
                                                   max=self.toggle_time - 1 + self.smooth_transition_offset)
        else:
            first_duration_sampler = UniformSampler(self.toggle_time - 1, self.toggle_time - 1)

        first_attack_fgs = [
            FlowGroupSampler(num, UniformSampler(0, 0),
                             first_duration_sampler,
                             ChoiceSampler(addr, replace=False),
                             rate_sampler=self._get_rate_sampler_for_attack_pattern(first_pattern),
                             attack=True)
            for
            num, addr in
            first_pattern.values()
        ]

        if self.smooth_transition:
            second_start_sampler = NormalSampler(self.toggle_time, .05 * self.maxtime,
                                                 min=self.toggle_time - self.smooth_transition_offset,
                                                 max=self.toggle_time + self.smooth_transition_offset)
        else:
            second_start_sampler = UniformSampler(self.toggle_time, self.toggle_time)

        second_attack_fgs = [
            FlowGroupSampler(num,
                             second_start_sampler,
                             UniformSampler(self.maxtime, self.maxtime),  # until end (clipped in TraceSampler anyway)
                             ChoiceSampler(addr, replace=False), attack=True,
                             rate_sampler=self._get_rate_sampler_for_attack_pattern(second_pattern))
            for num, addr in second_pattern.values()
        ]

        fgs.extend(first_attack_fgs)
        fgs.extend(second_attack_fgs)

        return fgs

    def _get_source_pattern_id_for_attack_pattern(self, pattern):  # TODO refactor
        id_from_dict = list(pattern.keys())[0]
        if id_from_dict in ['ntp', 'ssdp']:
            return id_from_dict
        else:
            return 'bot'

    def _get_rate_sampler_for_attack_pattern(self, pattern):  # TODO refactor
        if list(pattern.keys())[0] == 'ntp':
            return UniformSampler(self.attack_flows * 2 // list(pattern.values())[0][0],
                                  self.attack_flows * 2 // list(pattern.values())[0][0])
        elif list(pattern.keys())[0] == 'ssdp':
            return UniformSampler(self.attack_flows * 4 // list(pattern.values())[0][0],
                                  self.attack_flows * 4 // list(pattern.values())[0][0])
        else:
            return None

    def get_change_pattern_id(self):
        if self.random_toggle_time and self.smooth_transition:
            return f'smooth={self.toggle_time}+/-{self.smooth_transition_offset}'
        elif self.random_toggle_time:
            return f'var={self.toggle_time}'
        else:
            return f'fix={self.toggle_time}'

    def get_rate_pattern_id(self, time_step):
        return 'constant-rate'

    def _is_before_pattern_switch(self, time_step):
        return time_step < self.toggle_time / self.interval

    def _is_before_trace_end(self, time_step):
        return time_step <= self.maxtime / self.interval

    def _is_in_between_patterns(self, time_step):
        if not self.smooth_transition:
            return False
        else:
            # time index bounds of smooth transition
            smooth_region_start = self.toggle_time - 1 - self.smooth_transition_offset
            smooth_region_end = self.toggle_time + self.smooth_transition_offset

            # time index bounds of time step
            time_step_start_idx = time_step * self.interval
            time_step_end_idx = time_step * self.interval + self.interval - 1

            # if at least time index is in between smooth region bounds return True
            for time_index in range(time_step_start_idx, time_step_end_idx + 1):
                if smooth_region_start <= time_index <= smooth_region_end:
                    return True
            return False

    def get_source_pattern_id(self, time_step):
        if self._is_in_between_patterns(time_step):
            return f'{self.first_source_pattern_id}+{self.second_source_pattern_id}'
        elif self._is_before_pattern_switch(time_step):
            return self.first_source_pattern_id
        elif self._is_before_trace_end(time_step):
            return self.second_source_pattern_id
        else:
            raise ValueError(f'No source pattern for time step {time_step}; maxtime={self.maxtime}!')


class SourcePattern(ABC):
    @abstractmethod
    def generate_addresses(self, num_addr):
        pass


class UniformRandomSourcePattern(SourcePattern):

    def __init__(self, address_space):
        self.address_space = address_space

    def generate_addresses(self, num_benign) -> List[int]:
        all_addresses = range(0, 2 ** self.address_space)
        rng = default_rng()
        return rng.choice(all_addresses, size=num_benign, replace=False)


class ReflectorSourcePattern(SourcePattern):
    def __init__(self, refl_id, address_space=16, start_address=0):
        self.address_space = address_space
        self.refl_id = refl_id

        # if set, indicates that addresses should be generated from [start_addr,2**addr_space)
        # otherwise [0,2**addr_space)
        self.start_address = start_address

    def generate_addresses(self, num_addr):
        # reflectors can be all over the address space (simplification)
        return {self.refl_id: (num_addr, np.arange(self.start_address, 2 ** self.address_space))}


class BotnetSourcePattern(SourcePattern):
    ATK_PERCENTAGES = {
        'China': 0.336,
        'India': 0.322,
        'USA': 0.187,
        'Thailand': 0.155
    }

    def __init__(self, address_space=16, subnet=22):
        self.subnet = subnet
        self.number_of_subnets = int(2 ** address_space / Label.subnet_size(subnet))

        # how many /SUBNET networks to assign to each botnet country
        # depends on share of global IP addresses; but at least one per country
        self.number_of_country_subnets = {
            'China': max(1, round(0.077 * self.number_of_subnets)),
            'India': max(1, round(0.008 * self.number_of_subnets)),
            'USA': max(1, round(0.359 * self.number_of_subnets)),
            'Thailand': max(1, round(0.002 * self.number_of_subnets))
        }

    def generate_addresses(self, num_bots):
        result = {}
        rng = default_rng()
        # subnet start addresses when dividing address space based on /SUBNET subnets
        # list of randomly chosen start addresses that belong to the botnet countries
        self.subnet_starts = [x * Label.subnet_size(self.subnet) for x in range(0, self.number_of_subnets)]
        botnet_countries_starts = rng.choice(self.subnet_starts,
                                             size=sum((self.number_of_country_subnets.values())),
                                             replace=False)
        # assign each country its share of subnets
        start_idx = 0
        for country in self.number_of_country_subnets.keys():
            end_idx = start_idx + self.number_of_country_subnets[country]
            country_starts = botnet_countries_starts[start_idx:end_idx]
            country_ends = country_starts + Label.subnet_size(self.subnet)
            country_addresses = np.concatenate(
                list(map(lambda t: range(t[0], t[1]), zip(country_starts, country_ends))))
            number_of_bots = round(self.ATK_PERCENTAGES[country] * num_bots)
            result[country] = (number_of_bots, country_addresses)
            start_idx = end_idx
        return result
