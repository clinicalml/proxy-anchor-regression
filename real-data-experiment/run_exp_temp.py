#!/usr/bin/env python
# coding: utf-8

# Real-Data Experiments: Pollution in 5 Chinese Cities
# Link: https://archive.ics.uci.edu/ml/datasets/PM2.5+Data+of+Five+Chinese+Cities

# * Year / Month / Day / Hour
# * Season
# * DEWP: Dew Point (Celsius Degree)
# * TEMP: Temperature (Celsius Degree)
# * HUMI: Humidity (%)
# * PRES: Pressure (hPa)
# * cbwd: Combined wind direction
# * Iws: Cumulated wind speed (m/s)
# * precipitation: hourly precipitation (mm)
# * Iprec: Cumulated precipitation (mm)
# * (Target) PM2.5 concentration (ug/m^3)

import numpy as np
import pandas as pd
from copy import deepcopy
from numpy.random import default_rng
import pickle as pkl

import pdb

from sklearn import linear_model as lm
from sklearn import preprocessing
from sklearn import model_selection as ms
from sklearn import pipeline
from sklearn.model_selection import train_test_split as tt_split

import seaborn as sns
import matplotlib.pyplot as plt

from anchorRegression import AnchorRegression as AR
from anchorRegression import CrossProxyAnchorRegression as XAR
from anchorRegression import TargetedAnchorRegression as TAR
from anchorRegression import CrossTargetedAnchorRegression as XTAR
from anchorRegression import MeanPredictor
import utils

drop_all = True

proxies = ['TempC']

cities = np.arange(5)
seasons = np.arange(1, 5)

all_res_dfs = []
all_rmse_dfs = []
prox_info = {}

for CITY in cities:
    print(f"City: {CITY}")

    DATA_PATH = "data"

    files = [
        'BeijingPM20100101_20151231.csv',
        'GuangzhouPM20100101_20151231.csv',
        'ShenyangPM20100101_20151231.csv',
        'ChengduPM20100101_20151231.csv',
        'ShanghaiPM20100101_20151231.csv'
    ]

    dfs = [pd.read_csv(f"{DATA_PATH}/{f}") for f in files]

    raw_df = dfs[CITY].drop('No', axis=1)
    filt_df = raw_df.dropna()

    df, X, y = utils.process_df(filt_df)

    # Get Proxy Info for this city / season
    lr = pipeline.Pipeline([('scaler', preprocessing.StandardScaler()),
                            ('lr', lm.LinearRegression(fit_intercept=True))])

    this_city_prox_info = {}

    for prox in proxies:
        this_city_prox_info[prox] = {}

        # Get leave-one-out correlation with error
        this_X = X.drop([prox], axis=1)
        resid = y - lr.fit(this_X, y).predict(this_X)
        this_city_prox_info[prox]['corr_resid_lone'] = \
                np.corrcoef(resid, X[prox].values)[0, 1]

        # Get leave-all-out correlation with error
        this_X = X.drop(proxies, axis=1)
        resid = y - lr.fit(this_X, y).predict(this_X)
        this_city_prox_info[prox]['corr_resid_lall'] = \
                np.corrcoef(resid, X[prox].values)[0, 1]

    prox_info[CITY] = this_city_prox_info

    for test_season in seasons:
        print(f"\t Season: {test_season}")

        dev_year = 2013
        data = utils.get_dev_train_test_data(
                df, X, y, test_season, dev_year, proxies)

        baselines = utils.construct_baselines(
                data, proxies, drop_all=drop_all)
        tar_baselines = utils.construct_tar_baseline(
                data, proxies, drop_all=drop_all)
        tar_estimators = utils.construct_tar(
                data, proxies, drop_all=drop_all)
        ar_estimators = utils.construct_ar(
                data, proxies, drop_all=drop_all)

        if len(proxies) > 1:
            xtar_estimators = utils.construct_xtar(
                    data, proxies, drop_all=drop_all)
            xar_estimators = utils.construct_xar(
                    data, proxies, drop_all=drop_all)

            estimators = {
                    **baselines,
                    **tar_baselines,
                    **tar_estimators, **xtar_estimators,
                    **ar_estimators, **xar_estimators}
        else:
            estimators = {
                    **baselines,
                    **tar_baselines,
                    **tar_estimators,
                    **ar_estimators}

        for k, est in estimators.items():
            if 'tune_lambda' in est.keys() and est['tune_lambda']:
                best_lambda = utils.get_best_lambda(est, data)

                print(f"\t\t {k}: {best_lambda}")

                estimators[k]['pipe'] = estimators[k]['pipe'].set_params(
                    **best_lambda)

                estimators[k].update(best_lambda)

        for k, est in estimators.items():

            perf = {}

            # Get cross-validated training errors
            this_X_train = utils.get_estimator_X(data['train']['X'], est)
            this_X_test = utils.get_estimator_X(data['test']['X'], est)
            y_train = data['train']['y']
            y_test = data['test']['y']

            preds_train_cv = ms.cross_val_predict(est['pipe'],
                    this_X_train, y_train, fit_params=est['fit_params'], cv=10)
            resid_train_cv = preds_train_cv - y_train

            perf['Train (CV)'] = {
                'preds': preds_train_cv,
                'resid': resid_train_cv.values
            }

            # Train on the full training set    
            if est['fit_params'] is not None:
                est['fit'] = est['pipe'].fit(this_X_train, y_train, **est['fit_params'])
            else:
                est['fit'] = est['pipe'].fit(this_X_train, y_train)

            # Evaluate on the test set
            preds_test = est['fit'].predict(this_X_test)
            resid_test = preds_test - y_test

            perf['Test'] = {
                'preds': preds_test,
                'resid': resid_test.values
            }

            estimators[k]['perf'] = perf

        res_dfs = []

        for key, est in estimators.items():
            for env_name, perf in est['perf'].items():
                rs = pd.DataFrame(perf['resid'], columns=['Residual'])
                rs['City'] = CITY
                rs['Test_Season'] = test_season
                rs['Type'] = key.split()[0]
                rs['Estimator'] = key
                rs['Environment'] = env_name
                if 'lamb' in estimators[key]['fit']['pred'].get_params():
                    rs['Lambda'] = estimators[key]['fit']['pred'].get_params()['lamb']
                else:
                    rs['Lambda'] = np.nan

                res_dfs.append(rs)
                all_res_dfs.append(rs)

        res_df = pd.concat(res_dfs, axis=0)

        rmse = lambda v: np.sqrt(np.mean(v**2))

        rng = default_rng(0)

        # boostrap the RMSE
        n_boot_iter = 1000

        # For each estimator
        for key, est in estimators.items():
            # For each environment (train / test)
            for env_name, perf in est['perf'].items():
                # Bootstrap RMSE
                rmse_set = []
                for _ in range(n_boot_iter):
                    rmse_set.append(rmse(rng.choice(perf['resid'], size=len(perf['resid']))))
                rmse_set = np.array(rmse_set)

                # Save distribution of results
                estimators[key]['perf'][env_name]['rmse_boot'] = rmse_set

        rmse_dfs = []

        for key, est in estimators.items():
            for env_name, perf in est['perf'].items():
                rs = pd.DataFrame(perf['rmse_boot'], columns=['RMSE'])
                rs['City'] = CITY
                rs['Test_Season'] = test_season
                rs['Type'] = key.split()[0]
                rs['Estimator'] = key
                rs['Environment'] = env_name
                rmse_dfs.append(rs)
                all_rmse_dfs.append(rs)

        rmse_df = pd.concat(rmse_dfs, axis=0)

all_res_df = pd.concat(all_res_dfs, axis=0)
all_rmse_df = pd.concat(all_rmse_dfs, axis=0)
all_res_df.to_csv(f"results/all_res_test_{proxies[0]}.csv")
all_rmse_df.to_csv(f"results/all_rmse_test_{proxies[0]}.csv")
with open(f'results/prox_info_test_{proxies[0]}.pkl', 'wb') as f:
    pkl.dump(prox_info, f, protocol=pkl.HIGHEST_PROTOCOL)
