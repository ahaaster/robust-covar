import numpy as np
import pandas as pd

from robust_covar.core import sample_cov, min_variance, backtest


def test_min_variance_diag():
    cov = np.diag([1.0, 2.0, 4.0])
    weights = min_variance(cov)

    expected = np.array([1.0, 0.5, 0.25])
    expected = expected / expected.sum()

    assert np.isclose(weights.sum(), 1.0)
    assert np.allclose(weights, expected)


def test_backtest_rebalance_count():
    rng = np.random.default_rng(0)
    T, N = 1000, 5
    window, hold = 504, 21

    logret = rng.normal(0, 0.01, size=(T, N))
    dates = pd.bdate_range("2000-01-03", periods=T)
    rebalance_dates, *_ = backtest(logret, dates, {"sample": sample_cov}, window, hold)

    assert len(rebalance_dates) == 23


def test_backtest_no_lookahead():
    rng = np.random.default_rng(0)
    T, N = (
        1000,
        5,
    )
    window, hold = 504, 21

    logret = rng.normal(0, 0.01, size=(T, N))
    dates = pd.bdate_range("2000-01-03", periods=T)
    estimators = {"sample": sample_cov}

    *_, results = backtest(logret, dates, estimators, window, hold)

    logret_perturbed = logret.copy()
    logret_perturbed[window] = 0.5

    *_, results_perturbed = backtest(logret_perturbed, dates, estimators, window, hold)

    first_weights = results["sample"]["weights"][0]
    first_weights_perturbed = results_perturbed["sample"]["weights"][0]

    assert np.allclose(first_weights, first_weights_perturbed)
