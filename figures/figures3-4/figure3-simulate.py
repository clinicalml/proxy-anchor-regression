### Loading libraries
import numpy as np
import pandas as pd
from tqdm import tqdm

# Load file tools and population_estimators from parent folder
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from population_estimators import pack_params, gamma_ar, gamma_par, gamma_ols, gamma_cross
from tools import Id, simulate, N, get_mse, inv

### Set seed
np.random.seed(1)

### Dimensions
d = {"A": 2, "W": 2, 'Z': 2, "Y": 1, "X": 2, "H": 1}
d['O'] = d['X'] + d['Y'] + d['H']

### Parameters
beta = Id(d['A'], d['W'])
pars = {'M': np.random.poisson(lam=2, size=(d['O'], d['A'])),
        'B': np.random.normal(size=(d['O'], d['O']))/3,
        'beta': beta}
np.fill_diagonal(pars['B'], 0)
noise_W = np.array([[0.5, 0], [-0.8, 1.5]])
lamb = 5


### Save matrix eigendecomposition for plotting in R
M_par = Id(2) + lamb*beta@inv(beta.T@beta + noise_W@noise_W.T)@beta.T
d_par, U_par = np.linalg.eig(M_par)

# Modifications to Figure 3
lamb2 = lamb/np.linalg.eigvals(beta@inv(beta.T@beta + noise_W@noise_W.T)@beta.T).min()
M_par2 = Id(2) + lamb2*beta@inv(beta.T@beta + noise_W@noise_W.T)@beta.T
d_par2, U_par2 = np.linalg.eig(M_par2)

# Save eigengalues for plotting ellipses
np.savetxt("figures/figures3-4/extra-files/figure-3-matrix-eigvals.csv", np.concatenate((np.array([d_par, d_par2]).T, U_par, U_par2), axis=1), delimiter=";")

# 1) Compute population estimators from parameters
c = {"Y": [0], "X": [1, 2]}
params = pack_params(pars, c, d, noise_W)

gammas = {"ols": gamma_ols(params),
          "par5": gamma_par(params, lamb),
          "par10": gamma_par(params, lamb2),
          "cross": gamma_cross(params, lamb),
          "ar": gamma_ar(params, lamb)
          }

# 2) Simulate interventions for scatter plot
results = {k: [] for k in gammas.keys()}
for intervention_strength in tqdm(np.arange(50)/8):
    # Interventions
    vs = N(int(8*(intervention_strength + 0.1)**1.1), d['A'])
    for v in vs:
        # Normalize
        v *= intervention_strength  # /norm(v)

        # Evaluate estimators in intervened dataset
        for method, gamma in gammas.items():
            results[method].append([intervention_strength,
                                    get_mse(simulate(n=50000, d=d, pars=pars, v=v, noise_W=noise_W), gamma),
                                    method,
                                    v])

# Convert to dataframe
df = pd.concat(pd.DataFrame(results[method], columns=(
    "Strength", "MSE", "Method", "A")) for method in gammas.keys()).reset_index(drop=True)
# Add columns with intervened value to plot in A-space
df = df.join(pd.DataFrame(df.A.tolist(), index=df.index,
                          columns=[f"A{i}" for i in range(d['A'])]))
df.to_csv("figures/figures3-4/figure-3-data.csv")
