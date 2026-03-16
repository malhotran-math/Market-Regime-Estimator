#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 11:19:00 2026

@author: malhotran
"""
# source_codes/features.py 
from __future__ import annotations 
"""
features.py computes and returns features from the data file (.csv type). This file essentially builds the variables we will use for our calculations. 

The features computed in this code are: 
    1. Returns (which is the daily price change), including logret. We need this to compute volatility, trend, and drawdowns.  
    2. ATR (Average True Range), a classic metric for capturing volatility. Used in risk regime classification.
    3. EWMA (Exponential Weighted Volatility Estimate), reacts faster than standard rolling estimate and can capture sudden spikes. 
    4. Drawdown % from rolling peak (1Y). 
    5. Signal to noise ratio (as defined in the documentation, essentialy displacement scaled by how a random walk would behave). 
    6. Trend scores required for Trend vs Range detection (over 20, 60, 200 windows). We map this to [0,1]. Derived from SNR. 
    7. Net log return over horizons h (20. 60, 200). 
    8. Optional: slope t-state, should one wish to use this instead of SNR for structure.
    9. Optional: VIX change/return

Notes:
  - All computations are causal (rolling windows)
  - Warmup is trimmed 
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    atr_window: int = 14
    ewma_vol_span: int = 20
    dd_lookback: int = 252  # 1y rolling peak
    trend_horizons: tuple[int, ...] = (20, 60, 200)

    price_col: str = "spy_close"
    ohlc_cols: tuple[str, str, str, str] = ("spy_open", "spy_high", "spy_low", "spy_close")
    vix_col: str = "vix_close"

    # switches (avoid computing things you are not using)
    compute_snr: bool = True
    compute_tstat: bool = True
    tstat_scale: float = 10.0  # larger => trend_tstat closer to 0 for same |t|


def _read_daily_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True).sort_index()
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")]
    return df


def compute_atr(df: pd.DataFrame, window: int, high: str, low: str, close: str) -> pd.Series:
    """ 
    ATR computation. 
    At time t, ATR_t = mean_{i = t-window+1 to t} (TR_i), 
        where TR_i = max(High_t - Low_t, |High_t - Close_{t-1}|, |Low_t - Close_{t-1}|) 
    """
    prev_close = df[close].shift(1)
    tr = pd.concat(
        [
            (df[high] - df[low]).abs(),
            (df[high] - prev_close).abs(),
            (df[low] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Simple rolling mean ATR (Wilder EMA also common; keep simple here)
    atr = tr.rolling(window=window, min_periods=window).mean()
    return atr.rename(f"atr_{window}")


def compute_ewma_vol(df: pd.DataFrame, span: int, ret_col: str = "logret") -> pd.Series:
    """ 
    EWMA conputation. 
    At time t, EWMA_var_t = (1-λ) * r_t^2 + λ * EWMA_var_{t-1}, 
    and EWMA_vol_t = sqrt(EWMA_var_t) 
    where λ = exp(-2 / (span + 1)) (pandas equivalent) 
    """
    # EWMA variance of returns, then sqrt
    ewma_var = (df[ret_col] ** 2).ewm(span=span, adjust=False, min_periods=span).mean()
    ewma_vol = np.sqrt(ewma_var)
    return ewma_vol.rename(f"ewma_vol_{span}")


def compute_drawdown(df: pd.DataFrame, price_col: str, lookback: int) -> pd.Series:
    """ 
    drawdown computer. 
    At time t, DD_t = P_t/max(P_{t-lookback to t}) - 1. 
    """
    rolling_peak = df[price_col].rolling(window=lookback, min_periods=lookback).max()
    dd = df[price_col] / rolling_peak - 1.0
    return dd.rename(f"dd_{lookback}")


def compute_trend_scores(
    df: pd.DataFrame,
    horizons: Iterable[int],
    price_col: str,
    ret_col: str = "logret",
) -> pd.DataFrame:
    """
    R_h(t)     = log(P_t / P_{t-h})
    σ_h(t)     = stddev(r_{t-h+1 to t}) for window h
    SNR_h(t)   = |R_h(t)| / (σ_h(t) * sqrt(h))
    trend_score_h(t) = 1 - exp(-SNR_h(t)) ∈ [0,1]
    """
    out: dict[str, pd.Series] = {}

    for h in horizons:
        R = np.log(df[price_col] / df[price_col].shift(h)).rename(f"net_logret_{h}")
        out[f"net_logret_{h}"] = R

        sigma = df[ret_col].rolling(window=h, min_periods=h).std()
        denom = sigma * np.sqrt(h)

        S = (R.abs() / denom)
        S = S.where(denom > 0)  # avoid inf when denom=0
        S = S.replace([np.inf, -np.inf], np.nan).rename(f"snr_{h}")
        out[f"snr_{h}"] = S

        trend_score = (1.0 - np.exp(-S)).rename(f"trend_score_{h}")
        out[f"trend_score_{h}"] = trend_score

    return pd.DataFrame(out, index=df.index)


def compute_slope_tstat_features(
    df: pd.DataFrame,
    horizons: Iterable[int],
    price_col: str,
    tstat_scale: float = 10.0,
) -> pd.DataFrame:
    """
    Rolling OLS: log(price) ~ time over each horizon h.
    Returns:
      slope_{h}         : slope per day (on log-price)
      tstat_{h}         : t-statistic of slope
      trend_tstat_{h}   : mapped to [0,1] via 1-exp(-|t|/scale)
    """
    logp = np.log(df[price_col].astype(float))
    out: dict[str, pd.Series] = {}

    for h in horizons:
        x = np.arange(h, dtype=float)
        x_mean = x.mean()
        x_cent = x - x_mean
        Sxx = float(np.sum(x_cent**2))
        if Sxx <= 0:
            raise ValueError("Sxx must be > 0")

        y = logp
        sum_y = y.rolling(h, min_periods=h).sum()
        sum_y2 = (y * y).rolling(h, min_periods=h).sum()

        # Σ x_i * y_{t-h+1+i}
        sum_xy = y.rolling(h, min_periods=h).apply(lambda w: float(np.dot(x, w)), raw=True)

        y_mean = sum_y / h
        Sxy = sum_xy - h * x_mean * y_mean

        beta = (Sxy / Sxx).rename(f"slope_{h}")

        a = (y_mean - beta * x_mean)

        sum_x = float(np.sum(x))
        sum_x2 = float(np.sum(x * x))

        SSE = (
            sum_y2
            - 2 * a * sum_y
            - 2 * beta * sum_xy
            + h * (a * a)
            + 2 * a * beta * sum_x
            + (beta * beta) * sum_x2
        )

        dof = h - 2
        sigma2 = SSE / dof
        se_beta = np.sqrt(sigma2 / Sxx)

        tstat = (beta / se_beta).replace([np.inf, -np.inf], np.nan).rename(f"tstat_{h}")

        trend_t = (1.0 - np.exp(-tstat.abs() / float(tstat_scale))).rename(f"trend_tstat_{h}")

        out[f"slope_{h}"] = beta
        out[f"tstat_{h}"] = tstat
        out[f"trend_tstat_{h}"] = trend_t

    return pd.DataFrame(out, index=df.index)


def compute_features(df_daily: pd.DataFrame, cfg: FeatureConfig = FeatureConfig()) -> pd.DataFrame:
    df = df_daily.copy()

    # returns
    if "logret" not in df.columns:
        df["logret"] = np.log(df[cfg.price_col] / df[cfg.price_col].shift(1))
    if "ret" not in df.columns:
        df["ret"] = df[cfg.price_col].pct_change()

    o, h, l, c = cfg.ohlc_cols

    # ATR
    atr = compute_atr(df, cfg.atr_window, high=h, low=l, close=c)
    df[atr.name] = atr

    # EWMA vol
    ew = compute_ewma_vol(df, cfg.ewma_vol_span, ret_col="logret")
    df[ew.name] = ew

    # Drawdown
    dd = compute_drawdown(df, cfg.price_col, cfg.dd_lookback)
    df[dd.name] = dd

    # SNR / trend_score / net_logret
    if cfg.compute_snr:
        trend_df = compute_trend_scores(df, cfg.trend_horizons, price_col=cfg.price_col, ret_col="logret")
        df = df.join(trend_df, how="left")

    # slope t-stat features
    if cfg.compute_tstat:
        tstat_df = compute_slope_tstat_features(
            df, cfg.trend_horizons, price_col=cfg.price_col, tstat_scale=cfg.tstat_scale
        )
        df = df.join(tstat_df, how="left")

    # VIX transforms
    if cfg.vix_col in df.columns:
        df["vix_chg_1d"] = df[cfg.vix_col].diff(1)
        df["vix_ret_1d"] = df[cfg.vix_col].pct_change()

    # Trim warmup (ensure all rolling windows have time to populate)
    min_warmup = max(cfg.atr_window, cfg.ewma_vol_span, cfg.dd_lookback, *cfg.trend_horizons)
    df = df.iloc[min_warmup:].copy()

    return df


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    ROOT = project_root()
    data_path = ROOT / "data" / "spy_vix_gold_daily.csv"
    out_path = ROOT / "data" / "spy_vix_features_daily.csv"

    df_daily = _read_daily_csv(data_path)
    df_feat = compute_features(df_daily, FeatureConfig())

    out_path.parent.mkdir(exist_ok=True)
    df_feat.to_csv(out_path)
    print(f"Saved: {out_path}")
    print("Tail of the dataframe (sanity check):")
    print(df_feat.tail())
    print("Message: Features computed. You can now run weekly.py for as-of week estimates.")
