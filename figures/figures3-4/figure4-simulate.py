### Loading libraries
import numpy as np
import pandas as pd
from tqdm import tqdm

# Load file tools from parent folder
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from tools import Id, simulate, ols, ar, N, get_mse, tar


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



# 1) Simulate
n = 100000
data = simulate(n, d, pars, noise_W=noise_W)
A, X, Y, W, Z = data['A'], data['X'], data['Y'], data['W'], data['Z']

### Simulation setups
# rotat = np.diag([np.sqrt(2), 1])
rotat = np.array([[1.6, 0.8],
                  [-0.8, .5]])
shift = np.array([3, -3])


# Set lambda
eta = shift.reshape(-1, 1)
lamb = np.linalg.eigvals(eta@eta.T).max() - 1

### Save matrix eigendecomposition for plotting in R
radius, U = np.linalg.eig(rotat@rotat.T)
np.savetxt("figures/figures3-4/extra-files/figure-4-matrix-eigvals.csv", np.concatenate((np.array([shift, radius, np.array([lamb,0])]).T, U), axis=1), delimiter=";")



# 1) Compute population estimators
gamma_tar, alpha_tar = tar(X, Y, A, Sigma = rotat@rotat.T, nu=shift)
gammas = {"ols": ols(X, Y),
          "tar": gamma_tar,
          "ar": ar(X, Y, A, lamb=lamb),
          }



# 2) Simulate interventions for scatter plot
results = {k: [] for k in gammas.keys()}
for intervention_strength in tqdm(np.arange(80)/8):
    # Interventions
    vs = N(int(np.round(8*(intervention_strength + 0.1)**1.1)), d['A'])
    for v in vs:
        # Normalize
        v *= intervention_strength  # /norm(v)

        # Evaluate estimators in intervened dataset
        for method, gamma in gammas.items():
            results[method].append([intervention_strength,
                                    get_mse(simulate(n=1000, d=d, pars=pars, v = v, cov_A = rotat@rotat.T), gamma, (alpha_tar if method=="tar" else 0)),
                                    method,
                                    v])

# Convert to dataframe
df = pd.concat(pd.DataFrame(results[method], columns=(
    "Strength", "MSE", "Method", "A")) for method in gammas.keys()).reset_index(drop=True)
# Add columns with intervened value to plot in A-space
df = df.join(pd.DataFrame(df.A.tolist(), index=df.index,
                          columns=[f"A{i}" for i in range(d['A'])]))
df.to_csv("figures/figures3-4/figure-4-data.csv")
