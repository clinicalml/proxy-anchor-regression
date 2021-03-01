### Loading libraries
import numpy as np
import pandas as pd
from tqdm import tqdm

# Load file tools.py and population_estimators from parent folder
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from tools import Id, simulate, ols, ar, get_mse, tar

"""
This file simulates data for the experiment in Section 5.4
See the appendix Section E.4 for details on this experiment
"""


### Set seed
np.random.seed(1)


### Dict with dimensions of variables
d = {"A": 2, "W": 2, 'Z': 2, "Y": 1, "X": 2, "H": 1}
# We store the joint dimension of the outcomes X, Y and H as d['O']
d['O'] = d['X'] + d['Y'] + d['H']

### Create parameter matrix M
M = np.array([[2, 1],
              [0, 1],
              [2, 2],
              [0, 3]])

### Create parameter matrix B
B = np.array([[ 0, -0.06,  0.07,  0.04],
              [ 0.05, 0,  0.19,  0.03],
              [ 0.11, -0.11, 0,  0.1 ],
              [-0.02,  0.02,  0.09, 0]])

### Pack parameters
pars = {'M': M,
        'B': B,
        'beta': Id(d['A'], d['W'])}


### Specify rotation and mean-shift of the anticipated distribution shift
rotat = np.diag([np.sqrt(2), 1])
shift = np.array([0, 2])
# Store setups
sim_setups = {
    "incorrect_shift": {"eta_tar": shift, "eta_sim": np.zeros(2),
                      "cov_A_tar": rotat@rotat.T, "cov_A_sim": Id(d['A'])},
    "correct_shift": {"eta_tar": shift, "eta_sim": shift,
                      "cov_A_tar": rotat@rotat.T, "cov_A_sim": rotat@rotat.T}
}
# We select lambda such that B B.T + eta eta.T <= (1+lambda) Id (EAA.T = Id)
eta = shift.reshape(-1, 1)
lamb = np.linalg.eigvals(rotat@rotat.T + eta@eta.T).max() - 1

### Simulate
results = []
n = 10000 # training size
m = 10000 # test size
for i in tqdm(range(10000)):
    # Simulate training data
    data = simulate(n, d, pars)
    A, X, Y, W, Z = data['A'], data['X'], data['Y'], data['W'], data['Z']

    # Fit estimators
    lamb = 4
    gammas = {"ols": ols(X, Y),
              "ar": ar(X, Y, A, lamb=lamb)}

    for setup, s in sim_setups.items():
        # Get simulation settings
        eta_tar, eta_sim, cov_A_tar, cov_A_sim = list(s.values())

        # Target etstimator
        gamma_tar, alpha_tar = tar(X, Y, A, Sigma = cov_A_tar, nu=eta_tar)

        # Simulate test data
        _data = simulate(m, d, pars, v = eta_sim, cov_A = cov_A_sim) #v=eta_sim

        # Append results
        results.append([
            get_mse(_data, gammas['ols']),
            get_mse(_data, gammas['ar']),
            get_mse(_data, gamma_tar, alpha=alpha_tar),
            setup
        ])

# Cast results to data frame
df = pd.DataFrame(results, columns=["ols", "ar", "tar", "setup"]).melt(id_vars = "setup", value_name="mse", var_name="method")

# Plotting data is done with ggplot in R. See file 'experiment4-plot.R'
df.to_csv("experiment4-data.csv")
