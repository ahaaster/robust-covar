from .core import sample_cov, ledoit_wolf_cov, robust_covar

ESTIMATORS = {
    "sample": sample_cov,
    "ledoit_wolf": ledoit_wolf_cov,
    "robust": robust_covar,
}
MARKETS = ["euronext", "sp500"]
