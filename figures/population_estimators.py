import numpy as np
inv = np.linalg.inv; Id = lambda d, d2=None: np.eye(d, d2)
"""
For some experiments we need the population versions of estimators instead of sample estimators.
This file has functions to such estimators, based on moments that are derived from parameter matrices.
"""

def pack_params(pars, c, d, noise_W):
    """
    The population estimators are computed based on moments.
    We compute the moments based on the parameters MB and the noise distributions.

    In the naming below, we the variable XW corresponds to E[XW^T] etc.
    We let O denote the stacked outcome (Y, X, W)
    """
    # Unpack inputs. c is a vector containing indices (e.g. O_1 is Y, O_2 is X_1,...)
    cY = c['Y']; cX = c['X'] # input
    beta, M, B = pars['beta'], pars['M'], pars['B']

    # Store the inverse of the matrix (Id - B)
    IB = inv(Id(d['O']) - B)

    # Compute moments relating to the outcome O
    OA = IB@M
    OW = IB@M@beta
    OO = IB@(M@M.T + Id(d['O']))@IB.T

    # Compute moment E[WW^T]
    if len(np.shape(noise_W)) == 0:
        WW = beta.T@beta + noise_W**2*Id(d['W'])
    else:
        WW = beta.T@beta + noise_W@noise_W.T

    # Compute covariance of A and cross proxies
    AA = Id(d['A'])
    ZW = beta.T@beta
    #Covariances relating to X
    XX = OO[cX][:,cX]; XY = OO[cX][:,cY]; XW = OW[cX]; XA = OA[cX]; XZ = XW
    # Covariances relating to Y
    YW = OW[cY]; YA = OA[cY]; YZ = YW

    # Return dict with all moments
    return {"IB": IB, "OA": OA, "OW":OW, "OO":OO, "WW":WW, "AA":AA, "ZW":ZW,"XX":XX,
            "XY":XY, "XW":XW, "XA":XA, "XZ":XZ,"YW":YW, "YA":YA, "YZ":YZ, "M":M}

# OLS
def gamma_ols(params):
    # Unpack moments
    XX, XY = params['XX'], params['XY']
    # Return estimator based on moments
    return inv(XX)@XY

# Proxy anchor regression
def gamma_par(params, lamb):
    # Unpack moments
    XX, XW, WW, XY, YW = params['XX'], params['XW'], params['WW'], params['XY'], params['YW']
    # Return estimator based on moments
    return inv(XX + lamb*XW@inv(WW)@XW.T)@(XY + lamb*XW@inv(WW)@YW.T)

# Anchor regression
def gamma_ar(params, lamb):
    # Unpack moments
    XX, AA, XY, XA, YA = params['XX'], params['AA'], params['XY'], params['XA'], params['YA']
    # Return estimator based on moments
    return inv(XX + lamb*XA@inv(AA)@XA.T)@(XY + lamb*XA@inv(AA)@YA.T)

def gamma_cross(params, lamb):
    # Unpack moments
    XX, XW, ZW, XZ, XY, YW, YZ = params['XX'], params['XW'], params['ZW'], params['XZ'], params['XY'], params['YW'], params['YZ']
    # Compute "denominator" (left-side inverse)
    denom = 2*XX + lamb*(XW@inv(ZW)@XZ.T + XZ@inv(ZW).T@XW.T)
    # Compute "numerator"
    num = 2*XY + lamb*(XW@inv(ZW)@YZ.T + XZ@inv(ZW).T@YW.T)
    return inv(denom)@num

def get_mse_v(gamma, v, params, c):
    """Compute the population mse of using an estimator gamma"""
    # Unpack
    M, IB = params['M'], params['IB']
    cY, cX = c['Y'], c['X']
    # Compute w_gamma
    w_gamma = (IB[cY,] - gamma.T@IB[cX,]).T
    # Output population MSE
    return (w_gamma.T@M@v@v.T@M.T@w_gamma + w_gamma.T@w_gamma)[0,0]
