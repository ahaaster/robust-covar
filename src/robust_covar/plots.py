import numpy as np
import pandas as pd

from matplotlib.ticker import PercentFormatter, FixedLocator, NullFormatter

# import matplotlib.pyplot as plt

ESTIMATOR_COLOURS = {"sample": "#2a78d6", "ledoit_wolf": "#eb6834", "robust": "#1baf7a"}
CRISES = {
    "GFC": ("2008-09-15", "2009-03-09"),
    "COVID": ("2020-02-20", "2020-04-30"),
    "2022": ("2022-01-03", "2022-10-12"),
}


def shade_crises(ax):
    for name, (start, end) in CRISES.items():
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        ax.axvspan(start, end, color="grey", alpha=0.15, linewidth=0)


def plot_volatility(df: pd.DataFrame, ax, window: int = 252):
    rolling_vol = df.rolling(window).std() * np.sqrt(252)
    for estimator in rolling_vol.columns:
        ax.plot(
            rolling_vol.index,
            rolling_vol[estimator],
            label=estimator,
            color=ESTIMATOR_COLOURS[estimator],
            linewidth=1.5,
        )

    ax.set_ylabel(f"Annualised volatility ({window}-day window)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    shade_crises(ax)
    ax.legend()


def plot_relative_volatility(df: pd.DataFrame, ax, base="sample", window: int = 252):
    rolling_vol = df.rolling(window).std() * np.sqrt(252)
    relative = rolling_vol.drop(columns=base).div(rolling_vol[base], axis=0)  # - 1

    for estimator in relative.columns:
        ax.plot(
            relative.index,
            relative[estimator],
            label=estimator,
            color=ESTIMATOR_COLOURS[estimator],
            linewidth=1.5,
        )

    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_ylabel(f"Volatility relative to {base}")
    ax.grid(alpha=0.3)
    shade_crises(ax)

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(
        PercentFormatter(1.0)
    )  # volatility panel: 10%, 20%, ...
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(
        FixedLocator([0.5, 0.8, 1, 1.25, 2, 4, 8])
    )  # ratio panel
    ax.yaxis.set_major_formatter("{x:g}×")
    ax.legend()


def summary_table(df: pd.DataFrame, rebalances: pd.DataFrame, base="sample"):
    pooled_vol = df.std() * np.sqrt(252)
    monthly_vol = rebalances.pivot(
        index="date", columns="estimator", values="realised_vol"
    )

    leverage = rebalances.groupby("estimator")["gross_leverage"]

    summary = pd.DataFrame(
        {
            "pooled_vol": pooled_vol,
            "vol_ratio": pooled_vol / pooled_vol[base],
            "median_monthly_ratio": monthly_vol.div(monthly_vol[base], axis=0).median(),
            "median_leverage": leverage.median(),
            "max_leverage": leverage.max(),
        }
    )
    return summary.round(3)
