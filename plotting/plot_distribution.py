from gyms.hhh.flowgen.distgen import WeibullSampler
import matplotlib.pyplot as plt
import numpy as np

labelsize = 11
sampler = WeibullSampler(3,
                         (1 / WeibullSampler.quantile(99.99, 3)) * 1 / 8 * 599)
for i in range(10):
    plt.figure(dpi=200)
    y = sampler.sample(1000000)
    plt.hist(y, bins=45)
    print(np.quantile(y, 0.9))
    plt.xlabel('Attack flow duration in time indices', fontsize=labelsize)
    plt.ylabel('Number of samples (N=1000000)', fontsize=labelsize)
    plt.tight_layout()
    plt.show()
