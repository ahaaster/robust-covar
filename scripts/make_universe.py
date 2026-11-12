import random
import pandas as pd
from pathlib import Path

SEED = 144
MAX_N = 100


def save_the_universe(file_path: Path, tickers: list) -> None:
    with open(file_path, "w") as save_file:
        for ticker in tickers:
            save_file.write(f"{ticker},")


if __name__ == "__main__":
    work_dir = Path(__file__).resolve().parents[1]
    data_path = work_dir / "data"
    universe_path = work_dir / "universe"
    random.seed(SEED)

    # Smallest market must be the first entry!
    markets = ("euronext", "sp500")

    for market in markets:
        target_file = f"logret_{market}_daily.csv"
        save_file = universe_path / f"{market}.csv"

        df = pd.read_csv(
            data_path / target_file,
            header=0,
            index_col=0,
        )
        tickers = df.columns.to_list()

        # Euronext has the lowest number of tickers, so dictates draw count
        if market == markets[0]:
            # Detrmine amount of draws to randomly pick from for other markets, with a maximum
            draws = min(df.shape[1], MAX_N)

            save_the_universe(save_file, tickers)
            continue

        tickers = random.sample(tickers, draws)
        save_the_universe(save_file, tickers)
