# read data
# calculate intermediate parameters
# fit TLM

import os
import numpy as np
from lir.classifiers import TLM_calc_MSwithin, TLM_calc_means, TLM_calc_kappa

dirname = os.path.dirname(__file__)
dataZ = np.loadtxt(os.path.join(dirname, 'data/TLM/input/ZDATA.csv'), delimiter=",", dtype="float", skiprows=1, usecols=range(1,12))
MSwithin = TLM_calc_MSwithin(dataZ[:,1:], dataZ[:,0])
means = TLM_calc_means(dataZ[:,1:], dataZ[:,0])
kappa = TLM_calc_kappa(dataZ[:,1:], dataZ[:,0])
print("t")