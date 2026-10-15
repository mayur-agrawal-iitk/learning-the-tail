"""Download daily index prices from Yahoo Finance and build clean log-return series.

Returns are daily log returns in percent: r_t = 100 * log(P_t / P_{t-1}).
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from .config import RAW_DIR

_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
_HEADERS = {"User-Agent": "Mozilla/5.0"}


def _epoch(date_str: str) -> int:
    return int(datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def download_ticker(ticker: str, start: str, end: str, retries: int = 3) -> pd.DataFrame:
    """Download daily OHLC prices for one Yahoo ticker via the public chart API."""
    params = {
        "period1": _epoch(start),
        "period2": _epoch(end) + 86400,
        "interval": "1d",
        "events": "history",
    }
    for attempt in range(retries):
        try:
            resp = requests.get(_CHART_URL.format(ticker=ticker), params=params,
                                headers=_HEADERS, timeout=30)
            resp.raise_for_status()
            result = resp.json()["chart"]["result"][0]
            break
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2.0 * (attempt + 1))

    tz = result["meta"].get("exchangeTimezoneName", "UTC")
    stamps = pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_convert(tz)
    quote = result["indicators"]["quote"][0]
    df = pd.DataFrame(
        {k: quote.get(k) for k in ("open", "high", "low", "close", "volume")},
        index=pd.DatetimeIndex(stamps.date, name="date"),
    )
    adj = result["indicators"].get("adjclose", [{}])[0].get("adjclose")
    df["adjclose"] = adj if adj is not None else df["close"]
    return df.loc[start:end]


def download_all(cfg: dict) -> None:
    """Download every ticker in the config to ``data/raw/<name>.csv``."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    start, end = cfg["data"]["start"], cfg["data"]["end"]
    for name, ticker in cfg["data"]["tickers"].items():
        df = download_ticker(ticker, start, end)
        df.to_csv(RAW_DIR / f"{name}.csv")
        print(f"{name:8s} {ticker:10s} {len(df):6d} rows  {df.index[0]} -> {df.index[-1]}")


def clean_returns(prices: pd.Series, name: str = "", verbose: bool = False) -> pd.Series:
    """Turn a close-price series into clean daily log returns (percent).

    Cleaning steps (each documented in the paper's data section):
      1. drop missing / non-positive prices and duplicated dates;
      2. compute log returns;
      3. drop exact-zero returns, which on Yahoo index data are stale prints on
         local holidays rather than genuine no-change days.
    """
    p = prices.dropna()
    p = p[p > 0]
    p = p[~p.index.duplicated(keep="last")].sort_index()
    r = 100.0 * np.log(p).diff().dropna()
    n_zero = int((r == 0.0).sum())
    r = r[r != 0.0]
    if verbose:
        big = r[r.abs() > 15]
        print(f"{name:8s} n={len(r):5d} dropped_zero={n_zero:4d} "
              f"|r|>15%: {[(d.strftime('%Y-%m-%d'), round(v, 1)) for d, v in big.items()]}")
    r.name = name
    return r


def load_returns(cfg: dict, verbose: bool = False) -> dict[str, pd.Series]:
    """Load cleaned percent log returns for all assets in the config."""
    out = {}
    start, end = cfg["data"]["start"], cfg["data"]["end"]
    for name in cfg["data"]["tickers"]:
        path = RAW_DIR / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing - run scripts/download_data.py first")
        df = pd.read_csv(path, index_col="date", parse_dates=True)
        r = clean_returns(df["adjclose"], name, verbose)
        out[name] = r.loc[start:end]
    return out
