import numpy as np
import pandas as pd

from sklearn.covariance import LedoitWolf


def sample_cov(logret):
    return np.cov(logret, rowvar=False)


def ledoit_wolf_cov(logret):
    return LedoitWolf().fit(logret).covariance_


def q_n(values):
    raise NotImplementedError


def gk_corr(Z):
    """Create correlation matrix from Z matric
    Z = C.T @ C / number_observations
    """
    raise NotImplementedError


def power_spectral_density_corr(R, epsilon=1e-4):
    raise NotImplementedError


def robust_covar(X, epsilon=1e-4):
    raise NotImplementedError


def min_variance(S):
    one_vector = np.ones(S.shape[0])
    x = np.linalg.solve(S, one_vector)
    return x / x.sum()


def backtest(logret, dates, estimators, window: int = 504, hold: int = 21):
    T = logret.shape[0]
    simple_ret = np.expm1(logret)  # evaluation is done with simple returns

    rebalance_dates, holding_dates = [], []
    results = {
        name: {
            "weights": [],
            "realised_vol": [],
            "gross_leverage": [],
            "portfolio_ret": [],
        }
        for name in estimators
    }

    for start in range(window, T - hold + 1, hold):
        logret_estimation = logret[start - window : start]
        rebalance_dates.append(dates[start])
        holding_dates.append(dates[start : start + hold])

        for name, estimator in estimators.items():
            cov_estimate = estimator(logret_estimation)
            weights = min_variance(cov_estimate)
            portfolio_ret = (
                simple_ret[start : start + hold] @ weights
            )  # daily portfolio returns

            realised_vol = portfolio_ret.std(ddof=1) * np.sqrt(252)
            gross_leverage = np.abs(weights).sum()

            result = results[name]
            result["weights"].append(weights)
            result["realised_vol"].append(realised_vol)
            result["gross_leverage"].append(gross_leverage)
            result["portfolio_ret"].append(portfolio_ret)

    return rebalance_dates, holding_dates, results
