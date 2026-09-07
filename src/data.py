"""Download, cache and transform index price data.

Prices come from Yahoo Finance via yfinance. Each index is cached as its own
CSV in data/raw/ so the rest of the project runs offline after the first pull.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

from . import config as C


def _cache_path(ticker: str) -> "C.Path":
    return C.DATA_RAW / f"{ticker.replace('^', '')}.csv"


def download_index(
    ticker: str,
    start: str = C.START_DATE,
    end: str | None = C.END_DATE,
    force: bool = False,
) -> pd.Series:
    """Return the daily adjusted close for one index, using a local cache.

    Yahoo's index tickers carry no dividends, so `auto_adjust=True` returns the
    same close price; it is kept on so the code also works for equities.
    """
    path = _cache_path(ticker)
    if path.exists() and not force:
        s = pd.read_csv(path, index_col=0, parse_dates=True)["close"]
        s.name = ticker
        return s

    raw = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
    )
    if raw.empty:
        raise RuntimeError(f"No data returned for {ticker}")

    close = raw["Close"]
    if isinstance(close, pd.DataFrame):  # yfinance returns MultiIndex columns
        close = close.iloc[:, 0]
    close = close.dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    close.name = ticker

    close.rename("close").to_frame().to_csv(path, index_label="date")
    return close


def load_prices(
    tickers: list[str] | None = None,
    start: str = C.START_DATE,
    end: str | None = C.END_DATE,
    force: bool = False,
) -> pd.DataFrame:
    """Adjusted closes for every index, outer-joined on the calendar date.

    An outer join keeps each market's own holidays as NaN rather than silently
    forward-filling a price that never traded. Column order follows
    config.INDICES so colour assignment is stable.
    """
    tickers = tickers or list(C.INDICES)
    series = [download_index(t, start=start, end=end, force=force) for t in tickers]
    prices = pd.concat(series, axis=1).sort_index()
    return prices.loc[start:] if start else prices


def log_returns(prices: pd.DataFrame | pd.Series, scale: float = 100.0):
    """Daily log returns in percent.

    The arch package's optimiser is much better behaved when returns are of
    order 1 rather than 0.01, so the project scales by 100 throughout. Returns
    are computed per column on that column's own observed days, so a holiday in
    Tokyo never creates a fake zero return or a fake two-day return elsewhere.
    """
    if isinstance(prices, pd.Series):
        return (np.log(prices.dropna()).diff().dropna() * scale).rename(prices.name)

    out = {}
    for col in prices.columns:
        s = prices[col].dropna()
        out[col] = np.log(s).diff().dropna() * scale
    return pd.DataFrame(out).sort_index()


def clean_returns(returns: pd.DataFrame | pd.Series, max_abs: float = 60.0):
    """Drop absurd observations that indicate a bad price print.

    A daily move beyond +/-60% in a major index is a data error, not a market
    event: the worst genuine day in this sample is Black-Monday-scale and well
    inside that bound. Returns the cleaned object and a report of what went.
    """
    if isinstance(returns, pd.Series):
        bad = returns.abs() > max_abs
        return returns[~bad], returns[bad]

    flagged = {}
    cleaned = returns.copy()
    for col in returns.columns:
        s = returns[col]
        bad = s.abs() > max_abs
        if bad.any():
            flagged[col] = s[bad]
        cleaned.loc[bad.fillna(False), col] = np.nan
    return cleaned, flagged


def series_for_model(returns: pd.DataFrame, ticker: str) -> pd.Series:
    """The single-index return series a GARCH model is actually fitted to."""
    return returns[ticker].dropna()


def save_processed(prices: pd.DataFrame, returns: pd.DataFrame) -> None:
    """Persist the cleaned panels so the notebook and report share one source."""
    prices.to_csv(C.DATA_PROCESSED / "prices.csv", index_label="date")
    returns.to_csv(C.DATA_PROCESSED / "log_returns_pct.csv", index_label="date")


def coverage_table(prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """One row per index: sample window, observation count, missing days."""
    rows = []
    for t in prices.columns:
        p = prices[t].dropna()
        r = returns[t].dropna()
        rows.append(
            {
                "Index": C.name(t),
                "Ticker": t,
                "Region": C.REGION.get(t, ""),
                "Start": p.index.min().date(),
                "End": p.index.max().date(),
                "Price obs": len(p),
                "Return obs": len(r),
                "Years": round((p.index.max() - p.index.min()).days / 365.25, 1),
            }
        )
    return pd.DataFrame(rows).set_index("Index")
