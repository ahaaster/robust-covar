import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sqlite3 import Connection

from robust_covar.core import sample_cov, backtest, ledoit_wolf_cov, robust_covar
from robust_covar.storage import connect, write_results, create_schema, read_rebalance

PROJECT_DIR = Path(__file__).resolve().parents[1]  # scripts/ -> repo root
PATH_DB = PROJECT_DIR / "results" / "backtest.sqlite"
MARKETS = {"euronext": "euronext", "sp500": "sp500"}
ESTIMATORS = {"sample": sample_cov, "ledoit_wolf": ledoit_wolf_cov, "robust": robust_covar}


def run(market: str, estimator_name: str, con):
    logret = pd.read_csv(
        PROJECT_DIR / f"data/logret_{market}_daily.csv", index_col=0, parse_dates=True
    )
    tickers = PROJECT_DIR / f"universe/{market}.csv"
    tickers = tickers.read_text().strip().rstrip(",").split(",")
    logret = logret[tickers]
    # print(f"Market date check: {logret.index[0], logret.index[-1], len(logret)}")

    rebalance_dates, holding_dates, results = backtest(
        logret.to_numpy(), logret.index, {estimator_name: ESTIMATORS[estimator_name]}
    )
    result = results[estimator_name]

    for i, rebalance_date in enumerate(rebalance_dates):
        write_results(
            con,
            market,
            estimator_name,
            rebalance_date,
            tickers,
            result["weights"][i],
            holding_dates[i],
            result["portfolio_ret"][i],
            result["realised_vol"][i],
            result["gross_leverage"][i],
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", nargs="+", default=list(MARKETS))
    parser.add_argument("--estimator", nargs="+", default=list(ESTIMATORS))
    args = parser.parse_args()

    con = connect(PATH_DB)
    create_schema(con)

    try:
        for market in args.market:
            for estimator_name in args.estimator:
                run(market, estimator_name, con)
                results = read_rebalance(
                    con, market, estimator_name, "1900-01-01", "2100-01-01"
                )
                print(
                    f"{market:9} {estimator_name:8} n={len(results):4}  "
                    f"vol={results['realised_vol'].mean():.3f}  "
                    f"lev={results['gross_leverage'].mean():.2f}"
                )
    finally:
        con.close()


if __name__ == "__main__":
    main()
