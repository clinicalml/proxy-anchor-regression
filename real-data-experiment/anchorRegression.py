import numpy as np

from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.linear_model._base import LinearModel
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

# Convenience functions
inv = np.linalg.inv

class AnchorRegression(LinearModel):
    def __init__(self, lamb=1, fit_intercept=False, normalize=False, copy_X=False):
        self.lamb = lamb
        self.fit_intercept=fit_intercept
        self.normalize=normalize
        self.copy_X = copy_X

    def fit(self, X, y, A=None):
        X, y = self._validate_data(X, y, y_numeric=True)

        X, y, X_offset, y_offset, X_scale = self._preprocess_data(
            X, y, fit_intercept=self.fit_intercept, normalize=self.normalize,
            copy=self.copy_X, sample_weight=None,
            return_mean=True)

        if type(A) is not np.ndarray:
            A = A.values

        # Center A
        A = A - A.mean(axis=0)

        self.coef_ = \
            inv(X.T@X + self.lamb*X.T@A@inv(A.T@A)@A.T@X)@(
                X.T@y + self.lamb*X.T@A@inv(A.T@A)@A.T@y)

        self._set_intercept(X_offset, y_offset, X_scale)

        self.is_fitted_ = True
        return self

class CrossProxyAnchorRegression(LinearModel):
    def __init__(self, lamb=1, fit_intercept=False, normalize=False, copy_X=False):
        self.lamb = lamb
        self.fit_intercept=fit_intercept
        self.normalize=normalize
        self.copy_X = copy_X

    def fit(self, X, y, W, Z):
        X, y = self._validate_data(X, y, y_numeric=True)

        X, y, X_offset, y_offset, X_scale = self._preprocess_data(
            X, y, fit_intercept=self.fit_intercept, normalize=self.normalize,
            copy=self.copy_X, sample_weight=None,
            return_mean=True)

        if type(W) is not np.ndarray:
            W = W.values
        if type(Z) is not np.ndarray:
            Z = Z.values

        # Center W
        W = W - W.mean(axis=0)
        Z = Z - Z.mean(axis=0)

        # Transpose to align with formatting of synth experiments
        W = W.T; Z = Z.T; X = X.T; Y = y.T

        ZW = inv(Z@W.T)
        denom = 2*X@X.T + self.lamb*(X@W.T@ZW@Z@X.T + X@Z.T@ZW.T@W@X.T)
        num = 2*X@Y.T + self.lamb*(X@W.T@ZW@Z@Y.T + X@Z.T@ZW.T@W@Y.T)
        self.coef_ = inv(denom)@num

        self._set_intercept(X_offset, y_offset, X_scale)
        self.is_fitted_ = True
        return self

class TargetedAnchorRegression(LinearModel):
    def __init__(self, fit_intercept=False, normalize=False, copy_X=False):
        self.fit_intercept=fit_intercept
        self.normalize=normalize
        self.copy_X = copy_X

    def fit(self, X, y, A=None, nu=None):
        '''
        Targeted shift where nu is the shifted A
        '''
        X, y = self._validate_data(X, y, y_numeric=True)

        X, y, X_offset, y_offset, X_scale = self._preprocess_data(
            X, y, fit_intercept=self.fit_intercept, normalize=self.normalize,
            copy=self.copy_X, sample_weight=None,
            return_mean=True)

        if type(A) is not np.ndarray:
            A = A.values

        # Center columns of A and nu, with respect to A
        n, d_A = A.shape
        mean_A = A.mean(axis=0)
        A = A - mean_A
        nu = nu - mean_A

        Sig_A = np.cov(A.T, bias=True)
        Sig_nu = np.cov(nu.T, bias=True)
        mean_nu = np.mean(nu, axis=0).T

        if len(np.shape(Sig_A)) == 0:
            Sig_A = np.tile(Sig_A, d_A).reshape(d_A, 1)
            Sig_nu = np.tile(Sig_nu, d_A).reshape(d_A, 1)

        # Transpose to align with formatting of synth experiments
        A = A.T; X = X.T; Y = y.T

        Omega = inv(A@A.T)@(Sig_nu - Sig_A)@inv(A@A.T)

        gamma = inv(X@X.T/n + X@A.T @ Omega @ A@X.T)@(
                    X@Y.T/n + X@A.T @ Omega @ A@Y.T)
        alpha = (Y - gamma.T@X)@A.T@inv(A@A.T)@mean_nu

        self.coef_ = gamma
        self.intercept_ = y_offset + alpha

        self.is_fitted_ = True
        return self

class CrossTargetedAnchorRegression(LinearModel):
    def __init__(self, fit_intercept=False, normalize=False, copy_X=False):
        self.fit_intercept=fit_intercept
        self.normalize=normalize
        self.copy_X = copy_X

    def fit(self, X, y, W=None, nu=None, Z=None):
        '''
        Targeted shift where nu is the shifted W
        '''
        X, y = self._validate_data(X, y, y_numeric=True)

        X, y, X_offset, y_offset, X_scale = self._preprocess_data(
            X, y, fit_intercept=self.fit_intercept, normalize=self.normalize,
            copy=self.copy_X, sample_weight=None,
            return_mean=True)

        if type(W) is not np.ndarray:
            W = W.values

        # Center columns of W and nu, with respect to W
        n, d_W = W.shape
        mean_W = W.mean(axis=0)
        W = W - mean_W
        nu = nu - mean_W

        Sig_W = np.cov(W.T, bias=True)
        Sig_nu = np.cov(nu.T, bias=True)
        mean_nu = np.mean(nu, axis=0).T

        if len(np.shape(Sig_W)) == 0:
            Sig_W = np.tile(Sig_W, d_W).reshape(d_W, 1)
            Sig_nu = np.tile(Sig_nu, d_W).reshape(d_W, 1)

        # Transpose to align with formatting of synth experiments
        Z = Z.T; W = W.T; X = X.T; Y = y.T

        Omega = inv(W@Z.T)@(Sig_nu - Sig_W)@inv(Z@W.T)

        gamma = inv(X@X.T/n + X@W.T @ Omega @ W@X.T)@(
                    X@Y.T/n + X@W.T @ Omega @ W@Y.T)
        alpha = (Y - gamma.T@X)@Z.T@inv(W@Z.T)@mean_nu

        self.coef_ = gamma
        self.intercept_ = y_offset + alpha

        self.is_fitted_ = True
        return self


class MeanPredictor(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=True)
        self.mean_ = y.mean()

        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def predict(self, X):
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        return np.ones(X.shape[0], dtype=np.int64) * self.mean_
