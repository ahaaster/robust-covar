import matplotlib

matplotlib.use("Agg")  # only render to file
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sqlite3 import Connection

from robust_covar.plots import THEMES, make_volatility_figure, summary_table
from robust_covar.storage import connect, read_daily_returns, read_rebalance

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

    # One figure per theme; the README serves the one matching the reader's GitHub theme
    for theme in THEMES:
        fig = make_volatility_figure(
            daily_ret,
            window=63,
            base=BASE_ESTIMATOR,
            theme=theme,
            title=f"{market}: minimum-variance portfolio",
        )
        suffix = "" if theme == "light" else f"_{theme}"
        fig_path = PATH_FIG / f"volatility_{market}{suffix}.png"
        fig.savefig(fig_path, dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)

    start, end = daily_ret.index[0], daily_ret.index[-1]
    rebalances = read_rebalance(con, market, estimators, start, end)
    summary = summary_table(daily_ret, rebalances)
    summary_md = summary.to_markdown()
    # Save the summary to markdown file
    (PATH_FIG / f"summary_{market}.md").write_text(summary_md)

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
