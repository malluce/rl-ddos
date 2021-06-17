#!/usr/bin/env python

class Packet(object):

	def __init__(self, ip, malicious):
		self.ip = ip
		self.malicious = malicious
