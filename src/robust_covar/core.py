import jax
import numpy as np
import jax.numpy as jnp

from jax import lax, vmap
from sklearn.covariance import LedoitWolf

QN_CONSTANT = 2.21914  # = 1 / (sqrt(2) * Phi^-1(5/8))


def sample_cov(logret):
    return np.cov(logret, rowvar=False)


def ledoit_wolf_cov(logret):
    return LedoitWolf().fit(logret).covariance_


def qn_scale(series, bisection_steps: int = 32):
    series = jnp.sort(series)
    T = series.shape[0]
    halfway = T // 2 + 1  # Just one further over halfway
    target_pair_count = halfway * (halfway - 1) // 2

    def pairs_within(distance):
        upper = jnp.searchsorted(series, series + distance, side="right", method="sort")
        return jnp.sum(upper - jnp.arange(T) - 1)

    def bisection_step(_, bounds):
        low, high = bounds
        mid = (low + high) / 2
        enough_pairs = pairs_within(mid) >= target_pair_count
        return jnp.where(enough_pairs, low, mid), jnp.where(enough_pairs, mid, high)

    low, high = lax.fori_loop(
        0, bisection_steps, bisection_step, (0.0, series[-1] - series[0])
    )
    return high


batched_qn = jax.jit(vmap(qn_scale, in_axes=1))


def calc_qn(logret):
    logret = jnp.asarray(logret).reshape(logret.shape[0], -1)
    return np.asarray(QN_CONSTANT * batched_qn(logret))


def gk_corr(logret) -> tuple[np.ndarray, np.ndarray]:
    T, N = logret.shape
    scales = calc_qn(logret)
    assert (scales > 0).all()  # Divide by zero precaution
    standardised = logret / scales
    pair_i, pair_j = np.triu_indices(N, k=1)

    sum_series = standardised[:, pair_i] + standardised[:, pair_j]
    diff_series = standardised[:, pair_i] - standardised[:, pair_j]
    squared_qn_sum = calc_qn(sum_series) ** 2
    squared_qn_diff = calc_qn(diff_series) ** 2

    corr = np.eye(N)
    corr[pair_i, pair_j] = (squared_qn_sum - squared_qn_diff) / (
        squared_qn_sum + squared_qn_diff
    )
    corr[pair_j, pair_i] = corr[pair_i, pair_j]
    return corr, scales


def positive_semidef_corr(corr, epsilon=1e-4):
    eigval, eigvec = np.linalg.eigh(corr)
    diag_floored = np.diag(np.maximum(eigval, epsilon))
    arr = eigvec @ diag_floored @ eigvec.T

    inverse_sqrt_diag = 1 / np.sqrt(np.diag(arr))
    return arr * np.outer(inverse_sqrt_diag, inverse_sqrt_diag)


def robust_covar(logret, epsilon=1e-4):
    corr, scales = gk_corr(logret)
    psd = positive_semidef_corr(corr, epsilon)
    return psd * np.outer(scales, scales)


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
