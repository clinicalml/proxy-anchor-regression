import pandas as pd
import numpy as np
from scipy.stats import skew
from copy import deepcopy

from sklearn import linear_model as lm
from sklearn import preprocessing 
from sklearn import model_selection as ms
from sklearn import pipeline
from sklearn.model_selection import train_test_split as tt_split

import sys; sys.path.insert(0, '..')

from anchorRegression import AnchorRegression as AR
from anchorRegression import CrossProxyAnchorRegression as XAR
from anchorRegression import TargetedAnchorRegression as TAR
from anchorRegression import CrossTargetedAnchorRegression as XTAR
from anchorRegression import MeanPredictor

import itertools as it

LAMBDA_RANGE = np.linspace(0, 40, 100)

def process_df(df):
    # The dataset contains PM (pollution) readings across different posts in the city.  We average them here
    PM_vars = [f for f in df.columns if "PM" in f]

    avg_pm = 0
    for f in PM_vars: avg_pm += df[f]
    avg_pm = avg_pm / len(PM_vars)

    df = df.drop(PM_vars, axis=1)
    df['avg_pm'] = avg_pm

    # Create a date-time index
    time_vars = ['year', 'month', 'day', 'hour']
    time = pd.to_datetime(df[time_vars])
    df.index = time

    df['target'] = np.log1p(df['avg_pm'])

    # Construct Features
    drop_vars = ['year', 'month', 'day', 'hour', 'season']

    cat_feats = ['month', 'day', 'hour', 'season', 'cbwd']
    for f in cat_feats:
        df[f] = pd.Categorical(df[f])

    X = df.copy()
    X = X.drop(drop_vars, axis=1)
    X = X.drop(['target', 'avg_pm'], axis=1)

    # Better variable names
    X = X.rename(columns={'DEWP': 'DewPt', 
                          'HUMI': 'Humidity', 
                          'PRES': 'Press', 
                          'TEMP': 'TempC', 
                          'cbwd': 'WindDir', 
                          'Iws': 'WindSp', 
                          'precipitation': 'PrecipHr', 
                          'Iprec': 'PrecipCm'})

    # Note that we drop the first dummy variable, because we'll be using OLS
    X = pd.get_dummies(X, drop_first=True)
    y = df['target']

    Xy = pd.concat([X, y], axis=1)

    numeric_feats = [f for f in X.columns if f not in time_vars and
                     'WindDir' not in f and
                     'season' not in f]

    X['PrecipCm'] = X['PrecipCm'] - X['PrecipHr']

    #log transform skewed numeric features:
    skewed_feats = X[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.index
    #print(skewed_feats)

    X[skewed_feats] = np.log1p(X[skewed_feats])

    return df, X, y

def get_dev_train_test_data_anchors(df, X, y, test_season, dev_year,
        anchor, proxy_names, rho=0.9):

    dev_ids = np.logical_and(df.season != test_season, df.year == dev_year)
    train_ids = np.logical_and(df.season != test_season, df.year != dev_year)
    test_ids = np.logical_and(df.season == test_season, df.year != dev_year)

    rng = np.random.default_rng(0)

    data = {'dev': {}, 'train': {}, 'test': {}}

    # Construct proxies, so that stv in train is equal to rho
    A = X[[anchor]]
    var_A = A[train_ids].var()
    # rho = var_A / (var_A + var_eps)
    sigma_eps = np.sqrt((var_A / rho) - var_A)
    print(f"\t\t\t Sigma for proxies is: {sigma_eps[0]:.3f}")

    for prox in proxy_names:
        X[[prox]] = pd.DataFrame(rng.normal(A, sigma_eps),
                index=A.index, columns=A.columns)

    data['dev'] = {
            'G': df[dev_ids][['season']],
            'X': X[dev_ids].drop(anchor, axis=1),
            'y': y[dev_ids]
        }

    data['train'] = {
            'G': df[train_ids][['season']],
            'X': X[train_ids].drop(anchor, axis=1),
            'y': y[train_ids]
        }

    data['test'] = {
            'G': df[test_ids][['season']],
            'X': X[test_ids].drop(anchor, axis=1),
            'y': y[test_ids]
        }

    for prox in proxy_names:
        data['dev'][prox] = X[dev_ids][[prox]]
        data['train'][prox] = X[train_ids][[prox]]
        data['test'][prox] = X[test_ids][[prox]]

    return data

def get_dev_train_test_data(df, X, y, test_season, dev_year, proxies):
    dev_ids = np.logical_and(df.season != test_season, df.year == dev_year)
    train_ids = np.logical_and(df.season != test_season, df.year != dev_year)
    test_ids = np.logical_and(df.season == test_season, df.year != dev_year)

    data = {'dev': {}, 'train': {}, 'test': {}}

    data['dev'] = {
            'G': df[dev_ids][['season']],
            'X': X[dev_ids],
            'y': y[dev_ids]
        }

    data['train'] = {
            'G': df[train_ids][['season']],
            'X': X[train_ids],
            'y': y[train_ids]
        }

    data['test'] = {
            'G': df[test_ids][['season']],
            'X': X[test_ids],
            'y': y[test_ids]
        }

    for prox in proxies:
        data['dev'][prox] = X[dev_ids][[prox]]
        data['train'][prox] = X[train_ids][[prox]]
        data['test'][prox] = X[test_ids][[prox]]

    return data

def construct_baselines(data, proxies, drop_all=False):
    mp = pipeline.Pipeline([('pred', MeanPredictor())])

    lr = pipeline.Pipeline([('scaler', preprocessing.StandardScaler()), 
                            ('pred', lm.LinearRegression(fit_intercept=True, normalize=False))])

    baselines = {
        'Mean': 
            {'pipe': mp,
            'fit_params': None},
        'OLS (All)':
            {'pipe': deepcopy(lr),
            'fit_params': None},
        'OLS':
            {'pipe': deepcopy(lr),
             'fit_params': None,
             'drop_cols': proxies},
        }

    return baselines

def construct_tar_baseline(data, proxies, drop_all=False):
    tar = pipeline.Pipeline(
            [('scaler', preprocessing.StandardScaler()),
             ('pred', TAR(fit_intercept=True, normalize=False))])

    estimators = {}
    for prox in proxies:
        estimators[f"PA ({prox})"] = {
            'pipe': deepcopy(tar),
            'drop_cols': proxies if drop_all else [prox],
            'fit_params': {
                'pred__A': data['train'][prox],
                'pred__nu': np.ones_like(data['test'][prox]
                    )*data['test'][prox].mean().values[0]
                }
            }

    return estimators

def construct_tar(data, proxies, drop_all=False):
    tar = pipeline.Pipeline(
            [('scaler', preprocessing.StandardScaler()),
             ('pred', TAR(fit_intercept=True, normalize=False))])

    estimators = {}
    for prox in proxies:
        estimators[f"TAR ({prox})"] = {
            'pipe': deepcopy(tar),
            'drop_cols': proxies if drop_all else [prox],
            'fit_params': {
                'pred__A': data['train'][prox],
                'pred__nu': data['test'][prox]}
            }

    return estimators

def construct_xtar(data, proxies, drop_all=False):
    xtar = pipeline.Pipeline(
            [('scaler', preprocessing.StandardScaler()),
             ('pred', XTAR(fit_intercept=True, normalize=False))])

    estimators = {}
    for prox_combo in it.permutations(proxies, 2):
        estimators[f"xTAR ({prox_combo[0]}, {prox_combo[1]})"] = {
            'pipe': deepcopy(xtar),
            'drop_cols': proxies if drop_all else [p for p in prox_combo],
            'fit_params': {
                'pred__W': data['train'][prox_combo[0]],
                'pred__Z': data['train'][prox_combo[1]],
                'pred__nu': data['test'][prox_combo[0]]}
            }

    return estimators

def construct_ar(data, proxies, drop_all=False):
    ar = pipeline.Pipeline(
            [('scaler', preprocessing.StandardScaler()),
             ('pred', AR(lamb=0, fit_intercept=True, normalize=False))])

    estimators = {}
    for prox in proxies:
        estimators[f"AR ({prox})"] = {
            'pipe': deepcopy(ar),
            'drop_cols': proxies if drop_all else [prox],
            'fit_params_dev': {
                'pred__A': data['dev'][prox],
                },
            'fit_params': {
                'pred__A': data['train'][prox],
                },
            'tune_lambda': True
            }

    return estimators

def construct_xar(data, proxies, drop_all=False):
    xar = pipeline.Pipeline(
            [('scaler', preprocessing.StandardScaler()),
             ('pred', XAR(lamb=0, fit_intercept=True, normalize=False))])

    estimators = {}
    for prox_combo in it.combinations(proxies, 2):
        estimators[f"xAR ({prox_combo[0]}, {prox_combo[1]})"] = {
            'pipe': deepcopy(xar),
            'drop_cols': proxies if drop_all else [p for p in prox_combo],
            'fit_params_dev': {
                'pred__W': data['dev'][prox_combo[0]],
                'pred__Z': data['dev'][prox_combo[1]]
                },
            'fit_params': {
                'pred__W': data['train'][prox_combo[0]],
                'pred__Z': data['train'][prox_combo[1]]
                },
            'tune_lambda': True
            }

    return estimators

def get_estimator_X(X, est):
    if 'drop_cols' in est.keys():
        return X.copy().drop(est['drop_cols'], axis=1)
    else:
        return X.copy()

def get_best_lambda(est, data):
    X = data['dev']['X']
    y = data['dev']['y']
    G = data['dev']['G']

    X = get_estimator_X(X, est)
    fit_params = est['fit_params_dev']

    logo = ms.LeaveOneGroupOut()
    lamb_params = {'pred__lamb': LAMBDA_RANGE}

    est_cv = ms.GridSearchCV(
        est['pipe'], lamb_params, cv=logo,
        scoring='neg_root_mean_squared_error')
    est_cv = est_cv.fit(X, y, **fit_params, groups=np.ravel(G))

    return est_cv.best_params_
