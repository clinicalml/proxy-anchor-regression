### Loading libraries
import pandas as pd
import numpy as np
from tqdm import tqdm

# Load file tools from parent folder
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from tools import Id, ar, cross, cb, simulate

"""
This file simulates data for the experiment in Section 5.3
See the appendix Section E.3 for details on this experiment
"""

### Set seed
np.random.seed(1)

### Dict with dimensions of variables
d = {"A": 6,
     "W": 6,
     'Z': 6,
     "Y": 1,
     "X": 6,
     "H": 1}
# We store the joint dimension of the outcomes X, Y and H as d['O']
d['O'] = d['X'] + d['Y'] + d['H']

### Specify locations of parameters.
# E.g. cX1 specifies indices of (Y, X, H) containing X1
# cA2 specifies indices of A containing A2
cY  = [0]
cX1 = [1, 2, 3]
cX2 = [4, 5, 6]
cA1 = [0, 1, 2]
cA2 = [3, 4, 5]

# Create parameter matrix M
M = np.zeros((d['O'], d['A']))
M[np.ix_(cX1,cA1)] = np.ones((len(cX1), len(cA1)))
M[np.ix_(cX2,cA2)] = np.ones((len(cX1), len(cA1)))

# Create parameter matrix B
B = np.zeros((d['O'], d['O']))
B[np.ix_(cY, cX1)] = [1/4, 1/4, 1/4]
B[np.ix_(cX2, cY)] = np.array([[4, 4, 4]]).T

# Pack parameters in dict
pars = {'M':        M,
        'B':        B,
        'beta':     Id(d['A']),
        'beta_z':   Id(d['A'])
        }

# Variable 'noise' specifies the error variance of the proxies.
# The experiment regards considers a larger variance in proxy of A1 than in A2.
noise = np.diag([1 for i in cA1] + [3 for i in cA2])


# 1) Simulate
n = 10000
out = None

# Loop repeats experiment 1000 times
for _ in tqdm(range(1000)):
    # Simulate data
    data = simulate(n, d, pars, noise_W=noise)
    X, Y, A, W, Z = data['X'], data['Y'], data['A'], data['W'], data['Z']

    # Fit estimators
    par5 = ar(X, Y, W, lamb=5)
    cross5 = cross(X, Y, W, Z, lamb=5)
    ar5 = ar(X, Y, A, lamb=5)

    # Cast to dataframe
    df = pd.DataFrame(cb(par5, cross5, ar5), columns=["par5", "cross5", "ar5"])

    # 'Causal' encodes for whether predictor is causal.
    df['Causal'] = 3*[1] + 3*[0]
    # 'X.coord' encodes variable number (e.g. X_1, X_2, X_3, ...)
    df['X.coord'] = np.arange(1, 7)
    # Melt dataframe
    df = df.melt(id_vars=["Causal", "X.coord"], var_name="Method", value_name = "Weight")
    # Add results from this simulation to overall results
    out = df if out is None else pd.concat((out, df))

# Plotting data is done with ggplot in R. See file 'experiment3-plot.R'
out.to_csv("experiment3-data.csv")
