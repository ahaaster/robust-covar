import numpy as np
import pandas as pd

# import matplotlib.pyplot as plt


def plot_volatility(df: pd.DataFrame, ax, window: int = 252):
    rolling_vol = df.rolling(window).std() * np.sqrt(252)
    for estimator in rolling_vol.columns:
        ax.plot(rolling_vol.index, rolling_vol[estimator], label=estimator)

    ax.set_ylabel("Annualised volatility (1 year window)")
    ax.legend()


def plot_relative_volatility(df: pd.DataFrame, ax, base="sample", window: int = 252):
    rolling_vol = df.rolling(window).std() * np.sqrt(252)
    relative = rolling_vol.drop(columns=base).div(rolling_vol[base], axis=0) - 1

    for estimator in relative.columns:
        ax.plot(relative.index, relative[estimator], label=estimator)

    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_ylabel(f"Volatility relative to {base}")
    ax.legend()
