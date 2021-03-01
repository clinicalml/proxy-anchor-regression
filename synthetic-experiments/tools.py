import numpy as np

"""
This file contains implementations of estimators and a function to simulate data, used for experiments in the paper
"""

### Define convinience functions
inv = np.linalg.inv; norm = np.linalg.norm; Id = lambda d, d2=None: np.eye(d, d2)
# Column and row bind
def cb(*args): return np.concatenate(args, axis=1)
def rb(*args): return np.concatenate(args, axis=0)

# Multivariate gaussian
N = lambda d, n=1: np.random.normal(size=(d, n))

### Simulate function
def simulate(n, d, pars, v=None, shift=0, cov_A=None, noise_W=None, noise_Z=None):
    """ Simulate data from parameters. Dimensions d_A x n, etc. """
    # Unpack
    d_A = d['A']; d_W = d['W']; d_X = d['X']; d_Y = d['Y']; d_O = d['O']; d_Z = d['Z']
    M = pars['M']; B = pars['B']; beta = pars['beta']

    # If no noise is provided, use spherical unit variance as proxy noise
    if noise_W is None: noise_W = Id(d_W)
    elif len(np.shape(noise_W)) == 0:  noise_W = noise_W*Id(d_W)
    # If no noise for secondary proxy is supplied, use the same as W
    if noise_Z is None: noise_Z = noise_W
    elif len(np.shape(noise_Z)) == 0:  noise_Z = noise_Z*Id(d_Z)

    # If no parameter beta_z is provided, use same as beta_W
    if "beta_z" in pars.keys(): beta_z = pars['beta_z']
    else: beta_z = beta

    # If covariance matrix for A is given, use this, else use spherical noise
    # Since changed covariance matrices are only used for targeted, assumes also a v is given
    if cov_A is not None:
        A = np.random.multivariate_normal(v, cov=cov_A, size=n).T
    else:
        # Use either the intervention v tiled several times (fixed A), or a mean-zero gaussian
        A = (N(d_A, n) if v is None else np.tile(np.reshape(v, (d_A, 1)), n)) + shift
    # Compute the outcome O = (Y, X, H)
    O = inv(Id(d['O'])-B)@(M@A + N(d_O, n))
    Y, X, H = np.split(O, [d_Y, d_Y+d_X])
    #Simulate proxies
    W = beta.T@A + noise_W@N(d_W, n)
    Z = beta_z.T@A + noise_Z@N(d_Z, n)
    return {'A': A, 'W': W, 'Y': Y, 'X': X, 'H': H, 'Z': Z}

# Mean function
def E(X):
    return X.mean(axis=1).reshape(-1, 1)

### Estimators
# Ordinary least squares
def ols(X, Y, intercept=False):
    if intercept:
        X = np.concatenate((np.ones((1, X.shape[1])), X))
    return inv(X@X.T)@X@Y.T

# Anchor regression estimator
def ar(X, Y, A, lamb=1, intercept=False):
    if intercept:
        X = np.concatenate((np.ones((1, X.shape[1])), X))
    return inv(X@X.T + lamb*X@A.T@inv(A@A.T)@A@X.T)@(X@Y.T + lamb*X@A.T@inv(A@A.T)@A@Y.T)

# Cross estimator
def cross(X, Y, W, Z, lamb=1):
    ZW = inv(Z@W.T)
    denom = 2*X@X.T + lamb*(X@W.T@ZW@Z@X.T + X@Z.T@ZW.T@W@X.T)
    num = 2*X@Y.T + lamb*(X@W.T@ZW@Z@Y.T + X@Z.T@ZW.T@W@Y.T)
    return inv(denom)@num

# Targeted anchor regression, targeted to covariance Sigma and mean shift nu
def tar(X, Y, A, Sigma, nu=0):
    # Get dimensions
    d_A, n = A.shape
    if len(np.shape(nu)) == 0:
        nu = np.tile(nu, d_A).reshape(d_A, 1)

    # Compute alpha and gamma
    gamma = inv(X@X.T/n + X@A.T@inv(A@A.T)@(Sigma - A@A.T/n)@inv(A@A.T)@A@X.T)@(X@Y.T/n + X@A.T@inv(A@A.T)@(Sigma - A@A.T/n)@inv(A@A.T)@A@Y.T)
    alpha = (Y - gamma.T@X)@A.T@inv(A@A.T)@nu
    return gamma, alpha

# IV estimator
def iv(X, Y, A):
    return inv(X@A.T@inv(A@A.T)@A@X.T)@X@A.T@inv(A@A.T)@A@Y.T

# Function to evaluate the prediction MSE of a dataset and some gamma
def get_mse(data, gamma, alpha = 0):
    return ((data['Y'] - gamma.T@data['X'] - alpha)**2).mean()
