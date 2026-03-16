#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 10:41:23 2026

@author: malhotran
"""
from __future__ import annotations 
"""
This Python script can fetch the required data. We fetch the required datasets of S&P500 (using proxy SPY) 
    and Gold (using proxy GLD). Additionally, we fetch VIX and add it to the same dataframe since we will
    be using it as a volatility proxy. 
    
We fetch daily data (and not a finer timeframe) since the assumption is that this is sufficient for weekly 
    regime classification. A dataset finer than this might add more noise, but perhaps a 12H timeframe 
    could be another option. 
    
Disclaimer: Parts of the code are AI-assisted. 

Outputs (daily, indexed by date):
- spy_open, spy_high, spy_low, spy_close, spy_volume
- gld_close
- vix_close
- spy_logret, gld_logret

Notes:
- SPY/GLD use auto_adjust=True (total-return adjusted OHLC).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

DEFAULT_START = "2006-01-01"


def _flatten_yf_cols(df: pd.DataFrame) -> pd.DataFrame:
    #yfinance sometimes returns MultiIndex cols; flatten to single-level lowercase 
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = ["_".join([str(x) for x in tup if x]) for tup in out.columns.to_list()]
    out.columns = out.columns.str.lower()
    return out


def _pick_close_col(df: pd.DataFrame, ticker: str) -> str:
    """
    Find the close column from yfinance output after flattening.
    Accepts either:
      - 'close' (single ticker download)
      - 'close_{ticker}' (multi-ticker style)
    """
    t = ticker.lower()
    candidates = [f"close_{t}", "close"]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Couldnt find close column for {ticker}. Columns: {list(df.columns)[:10]}...")


def fetch_spy_vix_gld(start: str = DEFAULT_START) -> pd.DataFrame:
    spy = _flatten_yf_cols(yf.download("SPY", start=start, auto_adjust=True, progress=False))
    gld = _flatten_yf_cols(yf.download("GLD", start=start, auto_adjust=True, progress=False))
    vix = _flatten_yf_cols(yf.download("^VIX", start=start, auto_adjust=False, progress=False))

    spy_close = _pick_close_col(spy, "spy")
    gld_close = _pick_close_col(gld, "gld")
    vix_close = _pick_close_col(vix, "^vix")

    # Build output with stable column names
    out = pd.DataFrame(index=spy.index)
    # SPY OHLCV (prefer close naming stability)
    out["spy_open"] = spy.get("open_spy", spy.get("open"))
    out["spy_high"] = spy.get("high_spy", spy.get("high"))
    out["spy_low"] = spy.get("low_spy", spy.get("low"))
    out["spy_close"] = spy[spy_close]
    out["spy_volume"] = spy.get("volume_spy", spy.get("volume"))

    # GLD/VIX close only
    out["gld_close"] = gld[gld_close].reindex(out.index)
    out["vix_close"] = vix[vix_close].reindex(out.index)

    # Returns
    out["spy_logret"] = np.log(out["spy_close"] / out["spy_close"].shift(1))
    out["gld_logret"] = np.log(out["gld_close"] / out["gld_close"].shift(1))

    # Final cleanup
    out = out.dropna()
    out.index.name = "date"
    return out


if __name__ == "__main__":
    df = fetch_spy_vix_gld()
    path = DATA_DIR / "spy_vix_gold_daily.csv"
    df.to_csv(path)
    print(f"Saved {path.name}")
    print("Message: Data loaded. Proceed to run features.py to compute features from this dataset.")
