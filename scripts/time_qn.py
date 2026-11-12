import time
import numpy as np
import pandas as pd
from pathlib import Path

from robust_covar.core import robust_covar, min_variance

PROJECT_DIR = Path(__file__).resolve().parents[1]  # scripts/ -> repo root
PATH_DATA = PROJECT_DIR / "data"
PATH_UNIVERSE = PROJECT_DIR / "universe"

logret_df = pd.read_csv(
    PATH_DATA / "logret_sp500_daily.csv", index_col=0, parse_dates=True
)
tickers = [
    t
    for t in (PATH_UNIVERSE / "sp500.csv").read_text().strip().rstrip(",").split(",")
    if t
]
window_logret = logret_df[tickers].to_numpy()[:504]

robust_covar(window_logret)  # One warm up call
start_time = time.perf_counter()
cov_robust = robust_covar(window_logret)
seconds = time.perf_counter() - start_time

weights = min_variance(cov_robust)
print(f"{seconds:.1f} s per call -> ~{seconds * 229 / 60:.0f} min per market")
print(
    f"min eigenvalue {np.linalg.eigvalsh(cov_robust).min():.2e}, leverage {np.abs(weights).sum():.2f}"
)
