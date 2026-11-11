import numpy as np
import pandas as pd
from pathlib import Path


def clean_structurally(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the price values structurally"""
    df = df.sort_index()  # Make sure everything has ascending dates
    df = df[~df.index.duplicated(keep="first")]  # Remove duplicated dates
    df = df.loc[:, ~df.columns.duplicated()]  # same for duplicated tickers
    df = df.apply(pd.to_numeric, errors="coerce").astype(
        "float64"
    )  # Make everything a float value
    df = df.where(df > 0)  # Non-positive prices are errors
    return df


def clean_nans(
    df: pd.DataFrame,
    max_missing_per_date: float = 0.5,
    max_missing_per_share: float = 0.02,
) -> pd.DataFrame:
    """Remove dates with too many NaNs, then remove tickers with too high of a proportion of NaNs"""

    missing_proportion_per_date = df.isna().mean(axis=1)
    df = df[missing_proportion_per_date < max_missing_per_date]

    missing_proportion_per_ticker = df.isna().mean(axis=0)
    df = df.loc[:, missing_proportion_per_ticker < max_missing_per_share]

    df = df.ffill(limit=3)  # Limit the forward fills to just 3 days
    df = df.dropna(axis=1)  # remove remaining tickers with missing data
    assert not df.isna().any().any()

    return df


def clean_stale_tickers(df: pd.DataFrame, stale_ratio: float = 0.05) -> pd.DataFrame:
    """Find stale stocks where too many times there was 0 return"""
    zero_return_share = (df == 0).mean()
    stale = zero_return_share[zero_return_share > stale_ratio].index
    return df.drop(columns=stale)


def validate_logreturns(df: pd.DataFrame):
    """Final checks to make sure I didn't mess up"""

    market_return = df.mean(axis=1)
    corr_with_market = df.corrwith(
        market_return
    )  # Median corr_with_market near 0 -> something wrong!

    assert np.isfinite(df.to_numpy()).all()
    assert df.index.is_monotonic_increasing and df.index.is_unique

    repeats = ((df.diff() == 0) & (df != 0)).sum()
    zero_ratio = (df == 0).mean()

    # print(f"{corr_with_market.median() = }\n{df.std().median() * np.sqrt(252)}")
    # print(f"largest moves:\n{df.abs().stack().nlargest(10)}")
    print(f"Dim={df.shape} | Daterange: {df.index[0]}--{df.index[-1]}")
    print(f"Repeats = {repeats[repeats > 0].sort_values(ascending=False).head(5)}")
    print(f"ZeroRatios = {zero_ratio.sort_values(ascending=False).head(5)}")
    # print(
    #     f"Kept {log_returns.shape[1]} out of {raw_close_data.shape[1]} tickers for {log_returns.shape[0]} days"
    # )


if __name__ == "__main__":
    market = ("sp500", "euronext")
    interval = ("daily", "5min")

    data_path = Path(__file__).resolve().parent.parent / "data"
    target_name = f"{market[0]}_{interval[0]}.csv"
    raw_data = pd.read_csv(
        data_path / f"raw_{target_name}",
        header=[0, 1],  # raw data columns are MultiIndex with two levels
        index_col=0,
        parse_dates=True,
    )
    raw_close_data = raw_data["Close"]

    data = clean_structurally(raw_close_data)
    data = clean_nans(
        data, max_missing_per_date=0.2, max_missing_per_share=0.02  # arbitrary number
    )

    log_prices = np.log(data.values)
    log_returns = np.diff(log_prices, axis=0)
    log_returns = pd.DataFrame(log_returns, index=data.index[1:], columns=data.columns)
    log_returns = clean_stale_tickers(log_returns, stale_ratio=0.05)

    validate_logreturns(log_returns)

    # log_returns.to_csv(data_path / f"logret_{target_name}")
