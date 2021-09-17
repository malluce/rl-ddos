import math
import time
from abc import ABC, abstractmethod
from typing import List, Tuple

import gin
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from numpy.random import default_rng

from gyms.hhh.flowgen.distgen import ChoiceSampler, FlowGroupSampler, NormalSampler, UniformSampler, WeibullSampler
from gyms.hhh.label import Label


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
    def __init__(self, num_benign=25, num_attack=50, maxtime=600, maxaddr=0xffff, **kwargs):
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

    def __init__(self, num_benign=50, num_attack=100, maxtime=600, maxaddr=0xffff, **kwargs):
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

    def __init__(self, num_benign=300, num_attack=150, maxtime=600, maxaddr=0xffff, **kwargs):
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

    def __init__(self, num_benign=300, num_attack=150, maxtime=600, maxaddr=0xffff, **kwargs):
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


@gin.configurable
class TRandomPatternSwitch(SamplerTrafficTrace):
    def __init__(self, benign_flows=200, attack_flows=40, maxtime=600, maxaddr=0xffff, is_eval=False):
        super().__init__(maxtime)
        self.attack_flows = attack_flows
        self.benign_flows = benign_flows
        self.maxtime = maxtime
        self.maxaddr = maxaddr
        self.deterministic_cycle = is_eval  # cycle all possible pattern switches in eval or sample them in train

        if self.deterministic_cycle:
            self.current_pattern_combination = 0  # idx for possible pattern combinations

        self.benign_fgs = FlowGroupSampler(self.benign_flows,
                                           UniformSampler(0, 0.95 * self.maxtime),
                                           WeibullSampler(3 / 2,
                                                          (1 / WeibullSampler.quantile(99,
                                                                                       3 / 2)) * 1 / 8 * self.maxtime),
                                           UniformSampler(0, self.maxaddr),
                                           attack=False
                                           )

    def get_flow_group_samplers(self):
        address_space = round(math.log2(self.maxaddr))
        botnet = BotnetSourcePattern(address_space, subnet=22).generate_addresses(self.attack_flows)
        ntp_reflection = ReflectorSourcePattern('ntp', address_space).generate_addresses(1)
        ssdp_reflection = ReflectorSourcePattern('ssdp', address_space).generate_addresses(100)

        if not self.deterministic_cycle:
            # sample patterns
            used_patterns = default_rng().choice([botnet, ntp_reflection, ssdp_reflection], 2, replace=True)
        else:
            # return combinations of patterns
            all_combinations = [(x, y) for x in [botnet, ntp_reflection, ssdp_reflection]
                                for y in [botnet, ntp_reflection, ssdp_reflection]]
            used_patterns = all_combinations[self.current_pattern_combination]
            self.current_pattern_combination = (self.current_pattern_combination + 1) % len(all_combinations)
        print(used_patterns)
        fgs = [self.benign_fgs]

        first_attack_fgs = [
            FlowGroupSampler(num, UniformSampler(0, 0), UniformSampler(self.maxtime / 2 - 1, self.maxtime / 2 - 1),
                             ChoiceSampler(addr, replace=False),
                             rate_sampler=self._get_rate_sampler_for_attack_pattern(used_patterns[0]), attack=True) for
            num, addr in
            used_patterns[0].values()
        ]

        second_attack_fgs = [
            FlowGroupSampler(num, UniformSampler(self.maxtime / 2, self.maxtime / 2),
                             UniformSampler(self.maxtime / 2, self.maxtime / 2),
                             ChoiceSampler(addr, replace=False), attack=True,
                             rate_sampler=self._get_rate_sampler_for_attack_pattern(used_patterns[1]))
            for num, addr in used_patterns[1].values()
        ]

        fgs.extend(first_attack_fgs)
        fgs.extend(second_attack_fgs)

        return fgs

    def _get_rate_sampler_for_attack_pattern(self, pattern):  # TODO refactor
        if list(pattern.keys())[0] == 'ntp':
            return UniformSampler(self.attack_flows * 2 // list(pattern.values())[0][0],
                                  self.attack_flows * 2 // list(pattern.values())[0][0])
        elif list(pattern.keys())[0] == 'ssdp':
            return UniformSampler(self.attack_flows * 4 // list(pattern.values())[0][0],
                                  self.attack_flows * 4 // list(pattern.values())[0][0])
        else:
            return None


class RatePattern(ABC):
    @abstractmethod
    def generate_rates(self):
        pass


class ConstantRatePattern(RatePattern):

    def __init__(self, maxtime):
        self.maxtime = maxtime

    def generate_rates(self, addresses):
        pass


class SourcePattern(ABC):
    @abstractmethod
    def generate_addresses(self, num_addr):
        pass


class Pattern:
    def __init__(self, source_pattern: SourcePattern, rate_pattern: RatePattern):
        pass  # TODO


class UniformRandomSourcePattern(SourcePattern):

    def __init__(self, address_space):
        self.address_space = address_space

    def generate_addresses(self, num_benign) -> List[int]:
        all_addresses = range(0, 2 ** self.address_space)
        rng = default_rng()
        return rng.choice(all_addresses, size=num_benign, replace=False)


class ReflectorSourcePattern(SourcePattern):
    def __init__(self, refl_id, address_space=16):
        self.address_space = address_space
        self.refl_id = refl_id

    def generate_addresses(self, num_addr):
        # reflectors can be all over the address space (simplification)
        return {self.refl_id: (num_addr, np.arange(0, 2 ** self.address_space))}


class BotnetSourcePattern(SourcePattern):
    ATK_PERCENTAGES = {
        'China': 0.336,
        'India': 0.322,
        'USA': 0.187,
        'Thailand': 0.155
    }

    def __init__(self, address_space=16, subnet=22):
        self.address_space = address_space
        self.subnet = subnet
        self.number_of_subnets = 2 ** (subnet - address_space)

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
        botnet_countries_starts = rng.choice(self.subnet_starts, size=sum((self.number_of_country_subnets.values())),
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
