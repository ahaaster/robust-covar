import pytest
import numpy as np
import pandas as pd

from sklearn.covariance import LedoitWolf

from robust_covar.core import sample_cov, min_variance, backtest
from robust_covar.core import ledoit_wolf_cov, calc_qn, gk_corr
from robust_covar.core import positive_semidef_corr, robust_covar


@pytest.fixture
def rng():
    return np.random.default_rng(144)


def test_min_variance_diag():
    cov = np.diag([1.0, 2.0, 4.0])
    weights = min_variance(cov)

    expected = np.array([1.0, 0.5, 0.25])
    expected = expected / expected.sum()

    assert np.isclose(weights.sum(), 1.0)
    assert np.allclose(weights, expected)


def test_backtest_rebalance_count(rng):
    T, N = 1000, 5
    window, hold = 504, 21

    logret = rng.normal(0, 0.01, size=(T, N))
    dates = pd.bdate_range("2000-01-03", periods=T)
    rebalance_dates, *_ = backtest(logret, dates, {"sample": sample_cov}, window, hold)

    assert len(rebalance_dates) == 23


def test_backtest_no_lookahead(rng):
    T, N = (1000, 5)
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


def test_ledoitwolf_matches_formula(rng):
    logret = rng.normal(0, 0.01, size=(200, 10))

    fitted = LedoitWolf().fit(logret)
    sample = np.cov(logret, rowvar=False, ddof=0)
    mu = np.trace(sample) / sample.shape[0]

    delta = fitted.shrinkage_
    expected = (1 - delta) * sample + delta * mu * np.eye(10)

    assert 0 <= delta, +1
    assert np.allclose(ledoit_wolf_cov(logret), expected)


def test_qn_on_normal_data(rng):
    sigma = 2.0
    values = rng.normal(0, sigma, size=2000)

    scale = calc_qn(values)

    assert scale.shape == (1,)
    assert np.abs(calc_qn(values) / 2 - 1) < 0.05


def test_qn_exact_small_sample():
    """Hand-computed: T=4 -> h=3, k=3. Pairwise distances of [0, 1, 3, 7] sorted
    are 1, 2, 3, 4, 6, 7, so the 3rd smallest is 3. Catches a wrong k
    (off-by-one gives 2 or 4) and a wrong consistency constant."""
    assert np.isclose(calc_qn(np.array([0.0, 1.0, 3.0, 7.0]))[0], 3 * 2.2191)


def test_gk_corr_estimates_rho(rng):
    """SE of GK rho ~ (1 - rho^2) / sqrt(0.82 * n) ~ 0.016 for n = 2000; 0.05 ~ 3 SE."""
    rho = 0.6
    data = rng.multivariate_normal([0.0, 0.0], [[1.0, rho], [rho, 1.0]], size=2000)

    corr, scales = gk_corr(data)

    assert abs(corr[0, 1] - rho) < 0.05
    assert corr[0, 1] == corr[1, 0]
    assert np.allclose(np.diag(corr), 1.0)


def test_psd_repair_of_invalid_corr():
    invalid_corr = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    assert np.linalg.eigvalsh(invalid_corr).min() < 0  # input really is invalid

    repaired = positive_semidef_corr(invalid_corr)

    assert np.allclose(repaired, repaired.T)
    assert np.allclose(np.diag(repaired), 1.0)
    assert np.linalg.eigvalsh(repaired).min() > 0
