import matplotlib

matplotlib.use("Agg")  # only render to file
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sqlite3 import Connection

from robust_covar.plots import plot_relative_volatility, plot_volatility
from robust_covar.storage import connect, read_daily_returns

PROJECT_DIR = Path(__file__).resolve().parents[1]
PATH_FIG = PROJECT_DIR / "figures"
PATH_DB = PROJECT_DIR / "results" / "backtest.sqlite"

MARKETS = ("sp500", "euronext")
BASE_ESTIMATOR = "sample"


def available_estimators(con: Connection, market: str) -> list[str]:
    query = (
        "SELECT DISTINCT estimator FROM rebalances WHERE market = ? ORDER BY estimator"
    )
    return [row[0] for row in con.execute(query, (market,))]


def load_daily_returns_wide(
    con: Connection, market: str, estimators: list[str]
) -> pd.DataFrame:
    """Long table (estimator, date, ret) -> wide table: index date, one column per estimator"""
    daily_long = read_daily_returns(con, market, estimators, "1900-01-01", "2100-01-01")
    daily_wide = daily_long.pivot(index="date", columns="estimator", values="ret")
    daily_wide.index = pd.to_datetime(daily_wide.index)
    return daily_wide


def plot_market(con: Connection, market: str) -> Path | None:
    estimators = available_estimators(con, market)
    if not estimators:
        print(f"{market}: no results in database, skipped")
        return None

    daily_ret = load_daily_returns_wide(con, market, estimators)
    has_relative = BASE_ESTIMATOR in daily_ret.columns and daily_ret.shape[1] > 1

    n_panels = 2 if has_relative else 1
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(10, 3.5 * n_panels), sharex=True, squeeze=False
    )
    axes = axes[:, 0]

    plot_volatility(daily_ret, axes[0])
    axes[0].set_title(f"{market}: minimum-variance portfolio")
    if has_relative:
        plot_relative_volatility(daily_ret, axes[1], base=BASE_ESTIMATOR)

    fig.tight_layout()
    fig_path = PATH_FIG / f"volatility_{market}.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path


def main():
    PATH_FIG.mkdir(parents=True, exist_ok=True)
    con = connect(PATH_DB)
    try:
        for market in MARKETS:
            fig_path = plot_market(con, market)
            if fig_path is not None:
                print(f"{market}: saved {fig_path.relative_to(PROJECT_DIR)}")
    finally:
        con.close()


if __name__ == "__main__":
    main()
