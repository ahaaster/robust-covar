import sqlite3
import numpy as np
import pandas as pd

from robust_covar.storage import create_schema, write_results
from robust_covar.storage import read_rebalance, read_daily_returns, read_weights


def test_round_trip():
    """This test checks:
    - Correct picking of holding day starts and ends (no overlap)
    - Picking the correct date for weights when a non-rebalance day is queried
    - Check correct ordering of absolute weights
    """
    con = sqlite3.connect(":memory:")
    create_schema(con)

    holding_dates = pd.bdate_range("2009-01-02", periods=21)
    start, end = holding_dates[0], holding_dates[-1]
    rebalance_date = holding_dates[0]
    tickers = ["ABC", "IJK", "XYZ"]
    weights = np.array([0.5, -0.2, 0.7])
    portfolio_ret = np.linspace(-0.01, 0.01, 21)

    write_results(
        con,
        "sp500",
        "sample",
        rebalance_date,
        tickers,
        weights,
        holding_dates,
        portfolio_ret,
        realised_vol=0.12,
        gross_leverage=1.4,
    )

    rebalance = read_rebalance(con, "sp500", "sample", start, end)
    daily = read_daily_returns(con, "sp500", ["sample"], start, end)
    top = read_weights(con, "sp500", "sample", holding_dates[5], top=2)

    assert len(rebalance) == 1 and np.isclose(rebalance["realised_vol"][0], 0.12)
    # all 21 days, in order
    assert np.allclose(daily["ret"], portfolio_ret)
    # larget absolute weight is first
    assert list(top["ticker"]) == ["XYZ", "ABC"]
