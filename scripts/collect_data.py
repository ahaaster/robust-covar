import io
import time
import requests
import pandas as pd
import yfinance as yf

from tqdm import tqdm
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

url_sp500 = (
    "https://en.wikipedia.org/wiki/List_of_S&P_500_companies#S&P_500_component_stocks"
)
url_aex = "https://en.wikipedia.org/wiki/AEX_index#Composition"
url_euronext = "https://en.wikipedia.org/wiki/Euronext_100#Components"
urls = [url_aex, url_euronext, url_sp500]


def scrape_tickers(url: str, table_idx: int = 0, symbol_str: str = None) -> pd.Series:
    webpage = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).text
    ticker_table = pd.read_html(io.StringIO(webpage))[table_idx]

    tickers = ticker_table[symbol_str]
    # Fix some wrongly spelled ticker data and the like
    tickers = tickers.mask(
        tickers == "SHEEL.AS", "SHELL.AS"
    )  # Wikipedia's table misspells Shell's ticker
    tickers = tickers.mask(tickers == "BF.B", "BF-B")  # Yahoo finance uses '-'

    return tickers


def download_data(tickers: pd.Series, period: str, interval: str) -> pd.Series:
    tickers = " ".join(tickers)
    data = yf.download(tickers, period=period, interval=interval)
    return data


def fetch_sectors(
    tickers,
    target_path: Path = PROJECT_DIR / "data" / "sectors.csv",
):
    df = (
        pd.read_csv(target_path, index_col=0)
        if target_path.exists()
        else pd.DataFrame()
    )
    rows = {}
    for ticker in tqdm(tickers):
        if ticker in df.index and pd.notna(df.loc[ticker, "sector"]):
            continue
        try:
            info = yf.Ticker(ticker).info
            rows[ticker] = {
                "sector": info.get("sector"),
                "industry": info.get("industry"),
            }
        except Exception as e:
            rows[ticker] = {"sector": None, "industry": None}
        time.sleep(1)

    new_df = pd.concat([df, pd.DataFrame.from_dict(rows, orient="index")])
    new_df = new_df[~new_df.index.duplicated(keep="last")]
    new_df.to_csv(target_path)
    return new_df


if __name__ == "__main__":
    # aex_args = (url_aex, 3, "Ticker")
    sp500_args = (url_sp500, 0, "Symbol")
    euronext_args = (url_euronext, 3, "Ticker")

    tickers = scrape_tickers(*euronext_args)
    # tickers = scrape_tickers(*sp500_args)
    # data = download_data(tickers, period="10y", interval="1d")

    sectors = fetch_sectors(tickers)

    # data.to_csv("data/raw_euronext_5min_max.csv")
    # data.to_csv("data/raw_sp500_5min_max.csv")
