# pandas and numpy for data manipulation
import pandas as pd
import numpy as np

# scipy for algorithms
import scipy
from scipy import stats

# pymc3 for Bayesian Inference, pymc built on t
import pymc3 as pm

# matplotlib for plotting
import matplotlib.pyplot as plt
%matplotlib inline

from IPython.core.pylabtools import figsize
import matplotlib

import json
s = json.load(open('../style/bmh_matplotlibrc.json'))
matplotlib.rcParams.update(s)
matplotlib.rcParams['figure.figsize'] = (10, 3)
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['ytick.major.size'] = 20

# Number of samples for Markov Chain Monte Carlo
N_SAMPLES = 5000

na = 100
nb = 3
pos_a = 95
pos_b = 3

with pm.Model() as model:
    # priors
    p0a = pm.Beta('p0a', 1, 1)

    # likelihood
    obs_a = pm.Binomial("obs_a", n=na, p=p0a, observed=pos_a)

    # sample
    trace1_a = pm.sample(1000)


with pm.Model() as model:
    # priors
    p0b = pm.Beta('p0b', 1, 1)

    # likelihood
    obs_b = pm.Binomial("obs_b", n=nb, p=p0b, observed=pos_b)

    # sample
    trace1_b = pm.sample(1000)

######### Reading data ########3
seeds_data = np.array([[10, 23, 23, 26, 17, 5, 53, 55, 32, 46, 10, 8, 10, 8, 23, 0, 3, 22, 15, 32, 3], 
                       [39, 62, 81, 51, 39, 6, 74, 72, 51, 79, 13, 16, 30, 28, 45, 4, 12, 41, 30, 51, 7],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                       [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])

seeds_data=np.transpose(seeds_data)



