#!/usr/bin/env python

import gin

from .distgen import TraceSampler
from gyms.hhh.packet import Packet
from absl import logging


@gin.configurable
class DistributionTrace(object):

    def __init__(self, traffic_trace_construct, is_eval: bool):
        """
        Creates a new DistributionTrace.
        """
        self.samples = None
        self.N = None
        self.traffic_trace = traffic_trace_construct(is_eval=is_eval)
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
        flow_samplers = self.traffic_trace.get_flow_group_samplers()
        self.trace_sampler = TraceSampler(flow_samplers, maxtime=self.traffic_trace.get_max_time())
        self.trace_sampler.init_flows()

        self.samples = self.trace_sampler.samples()
        self.N = self.trace_sampler.num_samples
        logging.info(f'number of packets next episode: {self.N}')

    def __len__(self):
        """
        Returns the number of packets of this trace. (equal to episode length in packets)

        :return: number of trace packets
        """
        return self.N
