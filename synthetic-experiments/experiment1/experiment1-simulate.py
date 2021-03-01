### Loading libraries
import pandas as pd
import numpy as np
from tqdm import tqdm

# Load file tools.py and population_estimators from parent folder
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from tools import ols, ar, get_mse, cross, simulate
from population_estimators import pack_params, gamma_par, gamma_ar, gamma_ols, gamma_cross, get_mse_v

"""
This file simulates data for the experiment in Section 5.1
See the appendix Section E.1 for details on this experiment
"""

### Set seed
np.random.seed(1)

### Dict with dimensions of variables
d = {"A": 3, "W": 3, 'Z': 3, "Y": 1, "X": 3, "H": 1}
# We store the joint dimension of the outcomes X, Y and H as d['O']
d['O'] = d['X'] + d['Y'] + d['H']

### Specify locations of parameters.
# E.g. c['X'] specifies indices of (Y, X, H) containing X
c = {"Y": [0], "X": [1, 2, 3]}

### Create parameter matrix M
M = np.array([[1, 0, -2],
              [0, 2, 1],
              [-1, 3, 0],
              [2, 2, -3],
              [0, -2, 2]])

### Create parameter matrix B
B = np.zeros((d['O'], d['O']))
B[0] = np.array([0, -2, 2, 0, 1])
B[3] = np.array([3, 0, 0, 0, 1])

# Pack parameters in dict
pars = {'M':        M,
        'B':        B,
        'beta':     np.diag([1.0, 1, 1]),
        'beta_z':   np.diag([1.0, 1, 1])}

# Fix lambda
lamb = 5


# 1) Simulate
results = []

# We label the population means with a number to avoid type-errors (but they are computed by closed form equations, not a sample)
theo = 1e9

# Select fixed intervention direction
v = np.array([[-4, 0.5, 1.0]]).T

# Normalize to unit length, scale to trust region boundary, and upscale by 20%
v = 1.2*v*np.sqrt(1+lamb)/np.sqrt(v.T@v)

# Select points over x-axis
x_ax = np.arange(1, 21)/20
for x in tqdm(x_ax):
    # Compute s^2
    s2 = (1-x)/x

    # Pack parameter inputs to population mean functions
    params = pack_params(pars, c, d, np.sqrt(s2))

    # We only the population version once at every value of s^2
    results.append([theo, x] + [get_mse_v(gamma_ar(params, lamb), v, params, c),
                                get_mse_v(gamma_par(params, lamb), v, params, c),
                                get_mse_v(gamma_cross(params, lamb), v, params, c),
                                get_mse_v(gamma_ols(params), v, params, c)])

    # Loop over simulations
    for n in [250, 2500]:
        for _ in range(5000):
            # Simulate training data
            data = simulate(n, d, pars, noise_W=np.sqrt(s2), noise_Z=np.sqrt(s2))
            X, Y, A, W, Z = data['X'], data['Y'], data['A'], data['W'], data['Z']

            # Compute estimators from training data
            gammas = {
                'ar': ar(X, Y, A, lamb=lamb),
                "par": ar(X, Y, W, lamb=lamb),
                "cross": cross(X, Y, W, Z, lamb=lamb),
                "ols": ols(X, Y)}

            # Simulate test data from intervention do(A:=v)
            test_data = simulate(n, d, pars, noise_W=np.sqrt(s2), noise_Z=np.sqrt(s2), v=v)

            # Append results
            results.append([n, x] + [get_mse(test_data, gamma)
                                     for gamma in gammas.values()])

# Store data in dataframe
df = pd.DataFrame(np.array(results), columns=["n", "x"] + list(gammas.keys()))
df = df.melt(["x", "n"], var_name="method")
# Encode population values as "theo" instead of 10e9 (Panda handles, but numpy didnt)
df['n'] = df['n'].replace(theo, "theo")

# Plotting data is done with ggplot in R. See file 'experiment1-plot.R'
df.to_csv("experiment1-data.csv")
