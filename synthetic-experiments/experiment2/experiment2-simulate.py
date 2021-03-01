### Loading libraries
import pandas as pd
import numpy as np
from tqdm import tqdm

# Load file tools.py and population_estimators from parent folder
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from tools import Id, ols, ar, inv, simulate


"""
This file simulates data for the experiment in Section 5.2
See the appendix Section E.2 for details on this experiment
"""

### Set seed
np.random.seed(1)

### Dict with dimensions of variables
d = {"A": 3, "W": 3, 'Z': 3, "Y": 1, "X": 3, "H": 1}

# We store the joint dimension of the outcomes X, Y and H as d['O']
d['O'] = d['X'] + d['Y'] + d['H']

### Specify locations of parameters.
# E.g. cX specifies indices of (Y, X, H) containing X
cY = [0]; cX = [1, 2, 3]

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

# Pack parameters
pars = {'M':        M,
        'B':        B,
        'beta':     np.diag([1.0, 1, 1]),
        'beta_z':   np.diag([1.0, 1, 1])
        }

# Fix lambda
lamb = 5

# 1) Simulate
results = []

# For computing the actual worst case losses, we compute the inverse of the matrix Id - B
IB = inv(Id(d['O']) - B)

# We specify the assumed signal-to-variance ratio to 40%
svr = 0.4

# Loop over x axis
x_ax = np.arange(1, 21)/20
for x in tqdm(x_ax):

    # The noise variance s^2 is set as (1-x)/x, where x is the actual signal-to-variance
    s2 = (1-x)/x
    for n in [1000]:
        for _ in range(1000):
            # Simulate data set
            data = simulate(n, d, pars, noise_W=np.sqrt(s2), noise_Z=np.sqrt(s2))
            X, Y, A, W, Z = data['X'], data['Y'], data['A'], data['W'], data['Z']

            # Compute believed worst case loss (PAR)
            gamma_par = ar(X, Y, W, lamb=lamb)
            R = (Y - gamma_par.T@X)
            WCL_belief = np.mean(R**2) + lamb*R@W.T@inv(W@W.T)@W@R.T/n


            # Find actual worst case intervention v in Omega_W(0.5) set (PAR)
            w_gamma = (IB[cY,] - gamma_par.T@IB[cX,]).T
            b_gamma = (w_gamma.T@M).T
            v = b_gamma*np.sqrt((1+lamb*svr)/(b_gamma.T@b_gamma))

            # Compute actual worst case loss (PAR)
            WCL_actual = (b_gamma.T@v)**2 + w_gamma.T@w_gamma

            # Find worst case intervention and loss (OLS)
            gamma_ols = ols(X, Y)
            w_gamma = (IB[cY,] - gamma_ols.T@IB[cX,]).T
            b_gamma = (w_gamma.T@M).T
            v = b_gamma*np.sqrt((1+lamb*svr)/(b_gamma.T@b_gamma))
            WCL_ols = (b_gamma.T@v)**2 + w_gamma.T@w_gamma

            # Append results
            results.append([n, x, WCL_belief[0,0], WCL_actual[0,0], WCL_ols[0,0]])

# Convert to data frame
df = pd.DataFrame(np.array(results), columns=("n", "x", "belief", "actual", "ols"))
df = df.melt(["x", "n"])

# Plotting data is done with ggplot in R. See file 'experiment2-plot.R'
df.to_csv("experiment2-data.csv")
