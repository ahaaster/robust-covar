import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, FixedLocator, NullFormatter

# One validated categorical palette, stepped per background (colour-blind safe on both)
ESTIMATOR_COLOURS = {"sample": "#2a78d6", "ledoit_wolf": "#eb6834", "robust": "#1baf7a"}
ESTIMATOR_COLOURS_DARK = {
    "sample": "#3987e5",
    "ledoit_wolf": "#d95926",
    "robust": "#199e70",
}

THEMES = {
    "light": {
        "style": "default",
        "colours": ESTIMATOR_COLOURS,
        "background": "#ffffff",
        "shade_alpha": 0.15,
    },
    "dark": {
        "style": "dark_background",
        "colours": ESTIMATOR_COLOURS_DARK,
        "background": "#0e1117",
        "shade_alpha": 0.25,
    },  # Streamlit / GitHub dark surface
}
CRISES = {
    "GFC": ("2008-09-15", "2009-03-09"),
    "COVID": ("2020-02-20", "2020-04-30"),
    "2022": ("2022-01-03", "2022-10-12"),
}


def shade_crises(ax, alpha: float = 0.15):
    for name, (start, end) in CRISES.items():
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        ax.axvspan(start, end, color="grey", alpha=alpha, linewidth=0)


def plot_volatility(
    df: pd.DataFrame,
    ax,
    window: int = 252,
    colours: dict = ESTIMATOR_COLOURS,
    shade_alpha: float = 0.15,
):
    rolling_vol = df.rolling(window).std() * np.sqrt(252)
    for estimator in rolling_vol.columns:
        ax.plot(
            rolling_vol.index,
            rolling_vol[estimator],
            label=estimator,
            color=colours[estimator],
            linewidth=1.5,
        )
    shade_crises(ax, shade_alpha)

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([0.05, 0.1, 0.2, 0.4, 0.8]))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel(f"Annualised volatility ({window}-day window)")
    ax.grid(alpha=0.3)
    ax.legend()


def plot_relative_volatility(
    df: pd.DataFrame,
    ax,
    base="sample",
    window: int = 252,
    colours: dict = ESTIMATOR_COLOURS,
    shade_alpha: float = 0.15,
):
    rolling_vol = df.rolling(window).std() * np.sqrt(252)
    relative = rolling_vol.drop(columns=base).div(rolling_vol[base], axis=0)  # - 1

    for estimator in relative.columns:
        ax.plot(
            relative.index,
            relative[estimator],
            label=estimator,
            color=colours[estimator],
            linewidth=1.5,
        )
    shade_crises(ax, shade_alpha)

    ax.axhline(1, color="grey", linewidth=0.8)
    ax.set_ylabel(f"Volatility relative to {base}")
    ax.grid(alpha=0.3)

    ax.set_yscale("log")
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(FixedLocator([0.5, 0.8, 1, 1.25, 2, 4, 8]))
    ax.yaxis.set_major_formatter("{x:g}×")
    ax.legend()


def make_volatility_figure(
    daily_ret: pd.DataFrame,
    window: int = 63,
    base: str = "sample",
    theme: str = "light",
    title: str | None = None,
):
    """Volatility panel, plus the relative panel when `base` and another estimator are present.
    Used by both the README figures and the dashboard, so they cannot disagree."""
    settings = THEMES[theme]
    has_relative = base in daily_ret.columns and daily_ret.shape[1] > 1
    n_panels = 2 if has_relative else 1

    # style.context changes matplotlib's defaults only inside this block, not globally
    with plt.style.context(settings["style"]):
        fig, axes = plt.subplots(
            n_panels, 1, figsize=(10, 3.5 * n_panels), sharex=True, squeeze=False
        )
        fig.patch.set_facecolor(settings["background"])
        for ax in axes[:, 0]:
            ax.set_facecolor(settings["background"])

        plot_volatility(
            daily_ret,
            axes[0, 0],
            window=window,
            colours=settings["colours"],
            shade_alpha=settings["shade_alpha"],
        )
        if title is not None:
            axes[0, 0].set_title(title)
        if has_relative:
            plot_relative_volatility(
                daily_ret,
                axes[1, 0],
                base=base,
                window=window,
                colours=settings["colours"],
                shade_alpha=settings["shade_alpha"],
            )
        fig.tight_layout()
    return fig


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
