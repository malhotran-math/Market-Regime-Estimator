#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 15:12:32 2026

@author: malhotran
"""
# source_codes/weekly.py 
from __future__ import annotations 


"""
weekly.py ingests features computed using features.py. The goal of the project is the following: 
    ,,,At each weekly rebalance timestamp 𝑇𝑤 (week close), compute signals using only data up to 𝑇𝑤. 
    Those signals are the effective regime for week 𝑤+1.''' 
    
It's best to use the last trading day as week close (usually a Friday). 
We start with a daily csv for the purpose of computing rolling features (ATR for e.g.). 
Now, we use weekly.py to aggregrate these features. It defines the rebalancing clock and groups rows into weeks. 
It then picks the last trading day 𝑇𝑤, creates weekly as-of features by taking the last of all features in that week, 
    and that is then the features you know at the close of the week before rebalance. 
    
To keep timestamps consistent and avoiding messing up public holidays, AI assistance was used. 

Implementation:
  - Resample to week-end (default W-FRI).
  - For all feature columns: take last() observation in the week (as-of Tw).
  - Compute realized same-week return from daily log returns (diagnostic).
  - Compute forward-week return for evaluation (shift -1).

Notes:
  - W-FRI aligns week ends to Fridays; if Friday is a holiday, the last available trading day <= Friday is used.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WeeklyConfig:
    week_rule: str = "W-FRI"
    logret_col: str = "logret"
    feature_cols: tuple[str, ...] | None = None


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_daily_features(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True).sort_index()
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")]
    return df


def to_weekly_asof(df_daily: pd.DataFrame, cfg: WeeklyConfig = WeeklyConfig()) -> pd.DataFrame:
    if cfg.logret_col not in df_daily.columns:
        raise KeyError(f"Missing '{cfg.logret_col}' in daily dataframe.")

    df = df_daily.copy()

    # Columns to carry forward as "as-of" features using last() each week
    if cfg.feature_cols is None:
        # exclude anything we create below
        exclude = {"week_logret", "week_ret", "fwd_week_logret", "fwd_week_ret"}
        feature_cols = [c for c in df.columns if c not in exclude]
    else:
        feature_cols = list(cfg.feature_cols)

    # 1) Weekly "as-of" features at Tw: last observation in the week
    weekly_features = df[feature_cols].resample(cfg.week_rule).last()

    # 2) Weekly realized return (same-week) from daily log returns (diagnostic)
    week_logret = df[cfg.logret_col].resample(cfg.week_rule).sum().rename("week_logret")
    week_ret = (np.exp(week_logret) - 1.0).rename("week_ret")

    out = weekly_features.join([week_logret, week_ret], how="inner")

    # 3) Forward week return for evaluation of regimes computed at Tw
    out["fwd_week_logret"] = out["week_logret"].shift(-1)
    out["fwd_week_ret"] = out["week_ret"].shift(-1)

    # Drop last row (no forward return)
    out = out.dropna(subset=["fwd_week_ret"])

    out.index.name = "asof_date"
    return out


if __name__ == "__main__":
    ROOT = project_root()
    in_path = ROOT / "data" / "spy_vix_features_daily.csv"
    out_path = ROOT / "data" / "spy_vix_features_weekly.csv"

    df_daily = read_daily_features(in_path)
    df_weekly = to_weekly_asof(df_daily, WeeklyConfig())

    out_path.parent.mkdir(exist_ok=True)
    df_weekly.to_csv(out_path)
    print(f"Saved: {out_path}")
    print("Tail of df for sanity check:")
    print(df_weekly.tail())
    print("Message: All features loaded. Any model can now be used on these features. You can proceed with threshold_model.py for stats.")
