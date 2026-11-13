"""Streamlit dashboard: reads cached backtest results from SQLite, never reruns the backtest.

Run with:  uv run --extra app streamlit run app/dashboard.py
"""

import datetime as dt
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from robust_covar import ESTIMATORS, MARKETS
from robust_covar.plots import make_volatility_figure, summary_table
from robust_covar.storage import (
    connect,
    read_daily_returns,
    read_rebalance,
    read_weights,
)

PATH_DB = Path(__file__).resolve().parents[1] / "results" / "backtest.sqlite"
ESTIMATOR_NAMES = list(ESTIMATORS)  # only the names; the app never runs an estimator
BASE_ESTIMATOR = "sample"


# --- Data access ------------------------------------------------------------
# Streamlit reruns this whole script on every widget change. st.cache_data stores
# each function's result per combination of arguments, so a query runs only once
# per distinct selection. Each function opens and closes its own connection,
# because sqlite3 connections may not be shared between Streamlit's threads.
# Arguments must be hashable, hence tuples instead of lists.
# ----------------------------------------------------------------------------


@st.cache_data
def load_date_bounds(market: str) -> tuple[dt.date, dt.date]:
    con = connect(PATH_DB)
    try:
        query = "SELECT MIN(date), MAX(date) FROM daily_returns WHERE market = ?"
        first_day, last_day = con.execute(query, (market,)).fetchone()
    finally:
        con.close()
    return dt.date.fromisoformat(first_day), dt.date.fromisoformat(last_day)


@st.cache_data
def load_daily_returns(
    market: str, estimators: tuple, start: dt.date, end: dt.date
) -> pd.DataFrame:
    """Wide table: index date, one column of daily portfolio returns per estimator."""
    con = connect(PATH_DB)
    try:
        daily_long = read_daily_returns(con, market, list(estimators), start, end)
    finally:
        con.close()
    daily_wide = daily_long.pivot(index="date", columns="estimator", values="ret")
    daily_wide.index = pd.to_datetime(daily_wide.index)
    return daily_wide


@st.cache_data
def load_rebalances(
    market: str, estimators: tuple, start: dt.date, end: dt.date
) -> pd.DataFrame:
    """Long table: estimator, date, realised_vol, gross_leverage (as summary_table expects)."""
    con = connect(PATH_DB)
    try:
        return read_rebalance(con, market, list(estimators), start, end)
    finally:
        con.close()


@st.cache_data
def load_top_weights(
    market: str, estimator: str, date: dt.date, top: int = 15
) -> pd.DataFrame:
    """Largest |weight| holdings at the last rebalance on or before `date`."""
    con = connect(PATH_DB)
    try:
        return read_weights(con, market, estimator, date, top)
    finally:
        con.close()


# --- Controls -----------------------------------------------------------------

st.set_page_config(page_title="robust-covar", layout="wide")
st.title("Does a better covariance estimate give a lower-risk portfolio?")
st.caption(
    "Minimum-variance portfolios of 67 stocks, re-estimated every 21 trading days on a "
    "504-day window. All numbers are out of sample and read from cached results."
)

with st.sidebar:
    market = st.selectbox("Market", MARKETS, index=MARKETS.index("sp500"))
    estimators = st.multiselect("Estimators", ESTIMATOR_NAMES, default=ESTIMATOR_NAMES)
    first_day, last_day = load_date_bounds(market)
    start, end = st.slider(
        "Date range",
        min_value=first_day,
        max_value=last_day,
        value=(first_day, last_day),
        format="YYYY-MM",
    )
    window = st.select_slider(
        "Rolling window (trading days)", options=[21, 63, 252], value=63
    )

if not estimators:
    st.info("Select at least one estimator.")
    st.stop()  # ends this script run; nothing below is drawn

# Summary ratios need the base estimator, so it is always loaded and only hidden
# from the table when it was not selected.
loaded = tuple(sorted(set(estimators) | {BASE_ESTIMATOR}))
daily_ret = load_daily_returns(market, loaded, start, end)
rebalances = load_rebalances(market, loaded, start, end)

if len(daily_ret) <= window:
    st.warning(f"Choose a date range longer than the {window}-day rolling window.")
    st.stop()

# --- Volatility figure ------------------------------------------------------------

selected_ret = daily_ret[estimators]


def active_theme() -> str:
    """'dark' or 'light': the theme the viewer actually sees (st.context.theme, Streamlit
    >= 1.46), falling back to the configured base theme, then to light."""
    context_theme = getattr(getattr(st, "context", None), "theme", None)
    theme_type = getattr(context_theme, "type", None) or st.get_option("theme.base")
    return "dark" if theme_type == "dark" else "light"


fig = make_volatility_figure(
    selected_ret, window=window, base=BASE_ESTIMATOR, theme=active_theme()
)
st.subheader("Out-of-sample volatility")
st.pyplot(fig)
plt.close(fig)  # free memory: a new figure is drawn on every rerun
st.caption("Shaded: GFC, COVID crash and 2022 sell-off.")

# --- Summary table ----------------------------------------------------------------

st.subheader(f"Summary, {start:%Y-%m} to {end:%Y-%m}")
summary = summary_table(daily_ret, rebalances, base=BASE_ESTIMATOR)
st.dataframe(summary.loc[estimators])
st.caption(
    "pooled_vol: annualised std of all daily returns in the range. vol_ratio and "
    "median_monthly_ratio: relative to the sample-covariance portfolio. "
    "Leverage: sum of |weights|."
)

# --- Holdings -----------------------------------------------------------------------

st.subheader("Top 15 holdings at a rebalance")
holdings_estimator = st.selectbox("Estimator", estimators)
rebalance_dates = sorted(
    pd.to_datetime(
        rebalances.loc[rebalances["estimator"] == holdings_estimator, "date"]
    ).dt.date
)
holdings_date = st.select_slider(
    "Rebalance date", options=rebalance_dates, value=rebalance_dates[-1]
)
top_weights = load_top_weights(market, holdings_estimator, holdings_date)

used_date = top_weights["date"].iloc[0]
leverage_row = rebalances[
    (rebalances["estimator"] == holdings_estimator) & (rebalances["date"] == used_date)
]
st.caption(
    f"Weights set on {used_date}; gross leverage {leverage_row['gross_leverage'].iloc[0]:.2f}, "
    f"realised volatility over the next 21 days {leverage_row['realised_vol'].iloc[0]:.1%}."
)
st.dataframe(
    top_weights[["ticker", "weight"]].set_index("ticker").style.format("{:+.3f}")
)
