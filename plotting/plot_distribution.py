from gyms.hhh.flowgen.distgen import WeibullSampler
import matplotlib.pyplot as plt
import numpy as np

# This script is used to visualize Weibull distributions in the scenario descriptions in the evaluation chapter
# (Figures 6.1, 6.18, and 6.25)


# copy here the weibull sampler to visualize (from a scenario class in gyms.hhh.flowgen.traffic_traces)
sampler = WeibullSampler(20,
                         (1 / WeibullSampler.quantile(99.99, 20)) * 1 / 3 * 599)

labelsize = 11
plt.figure(dpi=200)
y = sampler.sample(1000000)
plt.hist(y, bins=45)
plt.xlabel('Attack flow duration in time indices (reflector flows)', fontsize=labelsize)
plt.ylabel('Number of samples (N=1000000)', fontsize=labelsize)
plt.tight_layout()
plt.show()
