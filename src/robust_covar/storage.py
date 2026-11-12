import sqlite3
import pandas as pd
from pathlib import Path
from sqlite3 import Connection

ISO_DAY_FORMAT = "%Y-%m-%d"
SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_returns (
    market TEXT, estimator TEXT, date TEXT, ret REAL,
    PRIMARY KEY (market, estimator, date)
);
CREATE TABLE IF NOT EXISTS weights (
    market TEXT, estimator TEXT, date TEXT,
    ticker TEXT, weight REAL,
    PRIMARY KEY (market, estimator, date, ticker)
);
CREATE TABLE IF NOT EXISTS rebalances (
    market TEXT, estimator TEXT, date TEXT,
    realised_vol REAL, gross_leverage REAL,
    PRIMARY KEY (market, estimator, date)
);
"""


def connect(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def create_schema(con: Connection):
    con.executescript(SCHEMA)


def write_results(
    con: Connection,
    market,
    estimator,
    rebalance_date: pd.Timestamp,
    tickers,
    weights,
    holding_dates: pd.DatetimeIndex,
    portfolio_ret,
    realised_vol,
    gross_leverage,
):
    key = (market, estimator)
    rebalance_day = rebalance_date.strftime(ISO_DAY_FORMAT)
    with con:
        con.execute(
            "INSERT OR REPLACE INTO rebalances (market, estimator, date, realised_vol, gross_leverage) VALUES (?, ?, ?, ?, ?)",
            (*key, rebalance_day, float(realised_vol), float(gross_leverage)),
        )
        con.executemany(
            "INSERT OR REPLACE INTO weights (market, estimator, date, ticker, weight) VALUES (?, ?, ?, ?, ?)",
            [
                (*key, rebalance_day, ticker, float(weight))
                for ticker, weight in zip(tickers, weights)
            ],
        )
        con.executemany(
            "INSERT OR REPLACE INTO daily_returns (market, estimator, date, ret) VALUES (?, ?, ?, ?)",
            [
                (*key, day.strftime(ISO_DAY_FORMAT), float(ret))
                for day, ret in zip(holding_dates, portfolio_ret)
            ],
        )


def read_rebalance(con: Connection, market, estimators, start, end):
    start = pd.Timestamp(start).strftime(ISO_DAY_FORMAT)
    end = pd.Timestamp(end).strftime(ISO_DAY_FORMAT)

    estimators = [estimators] if isinstance(estimators, str) else estimators
    placeholder = ", ".join("?" * len(estimators))

    query = f"""SELECT estimator, date, realised_vol, gross_leverage
                FROM rebalances
                WHERE market = ? AND estimator IN ({placeholder}) 
                    AND date BETWEEN ? AND ?
                ORDER BY estimator, date
            """

    return pd.read_sql_query(query, con, params=[market, *estimators, start, end])


def read_weights(con: Connection, market, estimator, date, top: int = 15):
    date = pd.Timestamp(date).strftime(ISO_DAY_FORMAT)

    query = """SELECT estimator, date, ticker, weight
                FROM weights
                WHERE market = ? AND estimator = ?
                    AND date = (SELECT MAX(date) FROM weights
                                WHERE market = ? AND estimator = ? 
                                AND date <= ?)
                ORDER BY ABS(weight) DESC
                LIMIT ?
            """

    return pd.read_sql_query(
        query, con, params=[market, estimator, market, estimator, date, top]
    )


def read_daily_returns(con: Connection, market, estimators, start, end):
    start = pd.Timestamp(start).strftime(ISO_DAY_FORMAT)
    end = pd.Timestamp(end).strftime(ISO_DAY_FORMAT)

    estimators = [estimators] if isinstance(estimators, str) else estimators
    placeholder = ", ".join("?" * len(estimators))

    query = f"""SELECT estimator, date, ret
                FROM daily_returns
                WHERE market = ? AND estimator in ({placeholder})
                    AND date BETWEEN ? AND ?
                ORDER BY estimator, date
            """

    return pd.read_sql_query(query, con, params=[market, *estimators, start, end])
