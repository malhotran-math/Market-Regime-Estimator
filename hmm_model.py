#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 16:39:05 2026

@author: malhotran
"""
from __future__ import annotations

"""
HMM Risk Regime (CALM vs STRESSED) + Structure Regime (TREND vs RANGE) on weekly features.

Design goals:
- Causal / walk-forward: fit on data up to Tw, label week w+1
- Stationary emissions (no raw price levels)
- Diagonal covariance for stability
- Robust state->label mapping (STRESSED = higher realized vol)
- Confidence from posterior probability (optionally hysteresis on posterior)

Outputs:
- CSV compatible with existing plot_risk.py and plot_structure.py


This model is incomplete and in-progress.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

try:
    from hmmlearn.hmm import GaussianHMM
except Exception as e:
    GaussianHMM = None

#cfg 
@dataclass(frozen=True)
class HMMConfig:
    fwd_ret_col: str = "fwd_week_ret"
    px_col: str = "spy_close"
    ewma_col: str = "ewma_vol_20"
    atr_col: str = "atr_14"
    dd_col: str = "dd_252"
    vix_col: str = "vix_close"
    week_logret_col: str = "week_logret"   
    week_ret_col: str = "week_ret"        

    #riskHMM
    n_states: int = 2
    covariance_type: str = "diag"
    n_iter: int = 300
    tol: float = 1e-3
    random_state: int = 7

    
    min_train_weeks: int = 156        # 3 years is min; you can set 104 if you must
    refit_every: int = 4              # refit HMM every N weeks (1 = refit weekly)

    #rolling z-score window for emissions (stationarity)
    z_window: int = 260               # ~5y of weekly data
    z_min_periods: int = 52

    #optional posterior hysteresis to reduce micro flips
    use_posterior_hysteresis: bool = True
    post_upper: float = 0.60          # enter STRESSED when P(stress) >= upper
    post_lower: float = 0.40          # return to CALM when P(stress) <= lower

    #structure (keep deterministic vote from your threshold model) ---
    horizons: Tuple[int, ...] = (20, 60, 200)
    snr_prefix: str = "snr_"
    trend_prefix: str = "trend_score_"
    structure_vote_k: float = 0.25
    use_structure_hysteresis: bool = True
    structure_upper: float = 0.6
    structure_lower: float = 0.4
    weights: Dict[int, float] = field(default_factory=lambda: {20: 0.15, 60: 0.30, 200: 0.55})


#helpers------------------------

def _clip01(x):
    return np.clip(x, 0.0, 1.0)

def rolling_zscore(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    mu = s.rolling(window=window, min_periods=min_periods).mean()
    sd = s.rolling(window=window, min_periods=min_periods).std()
    z = (s - mu) / sd
    return z.replace([np.inf, -np.inf], np.nan)

def project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def read_weekly(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True).sort_index()
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")]
    return df


#struct--------------------------

def compute_structure_vote(df: pd.DataFrame, cfg: HMMConfig) -> tuple[pd.Series, pd.Series, pd.Series]:
    w_sum = sum(cfg.weights.values())
    weights = {k: v / w_sum for k, v in cfg.weights.items()} if abs(w_sum - 1.0) > 1e-9 else cfg.weights

    vote = pd.Series(0.0, index=df.index)
    for h, w in weights.items():
        col = f"{cfg.trend_prefix}{h}"
        if col not in df.columns:
            raise KeyError(f"Missing '{col}' for structure vote.")
        vote += w * df[col].astype(float)

    vote = pd.Series(_clip01(vote), index=df.index).rename("structure_vote")
    conf = pd.Series(_clip01(((vote - 0.5).abs() / cfg.structure_vote_k)), index=df.index).rename("structure_confidence")

    if not cfg.use_structure_hysteresis:
        state = pd.Series(np.where(vote >= 0.5, "TREND", "RANGE"), index=df.index, name="structure_state")
        return vote, state, conf

    states = []
    prev = "RANGE"
    for v in vote.values:
        if v >= cfg.structure_upper:
            prev = "TREND"
        elif v <= cfg.structure_lower:
            prev = "RANGE"
        states.append(prev)

    state = pd.Series(states, index=df.index, name="structure_state")
    return vote, state, conf


#risk--------------------------

def build_risk_emissions(df: pd.DataFrame, cfg: HMMConfig) -> pd.DataFrame:
    """
    Stationary emissions for HMM.
    All based on weekly features, z-scored causally.
    """

    # Weekly realized vol proxy: abs(week_logret) or rolling std of weekly logrets
    if cfg.week_logret_col not in df.columns:
        raise KeyError(f"Missing '{cfg.week_logret_col}' (weekly.py should create it).")

    # ATR% (weekly as-of): atr_14 / spy_close
    if cfg.atr_col not in df.columns or cfg.px_col not in df.columns:
        raise KeyError("Missing ATR or price for atr_pct.")

    atr_pct = (df[cfg.atr_col] / df[cfg.px_col]).rename("atr_pct")

    # Raw “level” proxies -> z-score them
    z_ewma = rolling_zscore(df[cfg.ewma_col], cfg.z_window, cfg.z_min_periods).rename("z_ewma")
    z_atr = rolling_zscore(atr_pct, cfg.z_window, cfg.z_min_periods).rename("z_atr")

    if cfg.dd_col not in df.columns:
        raise KeyError(f"Missing '{cfg.dd_col}' for drawdown.")
    # drawdown is negative; more negative = worse => use -dd then z
    z_dd = rolling_zscore(-df[cfg.dd_col], cfg.z_window, cfg.z_min_periods).rename("z_dd")

    # VIX z-score (optional but strongly recommended)
    if cfg.vix_col in df.columns:
        z_vix = rolling_zscore(df[cfg.vix_col], cfg.z_window, cfg.z_min_periods).rename("z_vix")
    else:
        z_vix = pd.Series(np.nan, index=df.index, name="z_vix")

    # Weekly return sign helps HMM not call “high vol rally” the same as “high vol crash”
    # Use week_logret (already stationary)
    r = df[cfg.week_logret_col].rename("week_logret")

    X = pd.concat([r, z_ewma, z_atr, z_dd, z_vix], axis=1)

    # Drop rows where we can’t compute zscores yet
    X = X.dropna()

    return X


def fit_predict_walkforward_hmm(
    X: pd.DataFrame,
    cfg: HMMConfig
) -> pd.DataFrame:
    """
    Walk-forward:
    - At time t, fit HMM on X[:t] (train)
    - Predict posterior for X[t] (this corresponds to asof_date=t)
    """
    if GaussianHMM is None:
        raise ImportError("hmmlearn is not installed. Install with: pip install hmmlearn")

    idx = X.index
    n = len(X)

    # store posteriors for state 0/1
    post = np.full((n, cfg.n_states), np.nan, dtype=float)

    model: Optional[GaussianHMM] = None
    last_fit_i = -10**9

    X_values = X.values.astype(float)

    for i in range(n):
        # Need enough training history
        if i < cfg.min_train_weeks:
            continue

        # Refit on schedule
        if (i - last_fit_i) >= cfg.refit_every or model is None:
            train = X_values[:i, :]
            model = GaussianHMM(
                n_components=cfg.n_states,
                covariance_type=cfg.covariance_type,
                n_iter=cfg.n_iter,
                tol=cfg.tol,
                random_state=cfg.random_state,
            )
            model.fit(train)
            last_fit_i = i

        # posterior for current point i (one-step “asof” regime)
        # hmmlearn expects sequence; feed single observation as length-1
        _, gamma = model.score_samples(X_values[i:i+1, :])
        post[i, :] = gamma[0]

    out = pd.DataFrame(post, index=idx, columns=[f"post_state_{k}" for k in range(cfg.n_states)])
    return out


def map_states_to_risk_labels(
    df_joined: pd.DataFrame,
    post_cols: list[str],
) -> tuple[int, Dict[int, str]]:
    """
    Decide which hidden state is STRESSED by looking at realized vol proxy.
    We use abs(week_logret) as the simplest weekly vol proxy.
    """
    # If you have posteriors but no classification yet, map via mean abs return per state assignment
    probs = df_joined[post_cols].values
    hard = np.argmax(probs, axis=1)

    # Use abs weekly logret as vol proxy
    if "week_logret" not in df_joined.columns:
        raise KeyError("Need 'week_logret' to map states by riskiness.")

    absr = df_joined["week_logret"].abs().values

    means = {}
    for k in range(len(post_cols)):
        m = np.nanmean(absr[hard == k]) if np.any(hard == k) else -np.inf
        means[k] = m

    stressed_state = max(means, key=means.get)  # higher abs return => riskier
    risk_map = {0: "CALM", 1: "STRESSED"}

    # If stressed_state != 1, we’ll swap downstream by converting to binary labels
    return stressed_state, risk_map


def apply_hmm_model(df_weekly: pd.DataFrame, cfg: HMMConfig = HMMConfig()) -> dict:
    df = df_weekly.copy()

    # --- emissions (stationary, z-scored)
    X = build_risk_emissions(df, cfg)

    # align df to X (drop early NaNs)
    df = df.loc[X.index].copy()

    # --- walk-forward posteriors
    post = fit_predict_walkforward_hmm(X, cfg)
    df = df.join(post, how="left")

    post_cols = [c for c in df.columns if c.startswith("post_state_")]
    if df[post_cols].dropna().empty:
        raise ValueError("HMM produced no posteriors. Reduce min_train_weeks or check NaNs.")

    # --- map hidden states -> stressed/calm
    stressed_state, risk_map = map_states_to_risk_labels(df.dropna(subset=post_cols), post_cols)

    # Compute stressed posterior P(stressed)
    p_stress = df[f"post_state_{stressed_state}"].rename("p_stressed")

    # Optional posterior hysteresis for state label stability
    if not cfg.use_posterior_hysteresis:
        risk_state = (p_stress >= 0.5).astype(int)
    else:
        states = []
        prev = 0  # start CALM
        for p in p_stress.values:
            if np.isnan(p):
                states.append(np.nan)
                continue
            if p >= cfg.post_upper:
                prev = 1
            elif p <= cfg.post_lower:
                prev = 0
            states.append(prev)
        risk_state = pd.Series(states, index=df.index, name="risk_state")

    df["risk_state"] = risk_state
    df["risk_state_label"] = df["risk_state"].map({0: "CALM", 1: "STRESSED"})
    df["risk_gate"] = "hmm"

    # Confidence = posterior prob of current chosen label (hysteresis-safe)
    # If label==1 use p_stress else use 1-p_stress
    df["risk_confidence"] = np.where(df["risk_state"] == 1, p_stress, 1.0 - p_stress)
    df["risk_confidence"] = pd.Series(_clip01(df["risk_confidence"]), index=df.index)

    # --- Structure: deterministic vote
    vote, state, conf = compute_structure_vote(df, cfg)
    df["structure_vote"] = vote
    df["structure_state"] = state
    df["structure_confidence"] = conf

    # --- Stats
    risk_stats = pd.DataFrame()
    structure_stats = pd.DataFrame()

    if cfg.fwd_ret_col in df.columns:
        g = df.groupby("risk_state")[cfg.fwd_ret_col]
        risk_stats = g.agg(count="count", mean="mean", median="median", std="std", min="min")
        risk_stats["q05"] = g.quantile(0.05)
        risk_stats["hit_rate_pos"] = (df[cfg.fwd_ret_col] > 0).groupby(df["risk_state"]).mean()

        g2 = df.groupby("structure_state")[cfg.fwd_ret_col]
        structure_stats = g2.agg(count="count", mean="mean", median="median", std="std", min="min")
        structure_stats["q05"] = g2.quantile(0.05)
        structure_stats["hit_rate_pos"] = (df[cfg.fwd_ret_col] > 0).groupby(df["structure_state"]).mean()

    # Switching diagnostic
    valid_rs = df["risk_state"].dropna()
    risk_switches = valid_rs.ne(valid_rs.shift(1)).sum()
    risk_switch_rate = float(risk_switches) / float(len(valid_rs)) if len(valid_rs) else np.nan

    return {
        "df": df,
        "risk_map": {0: "CALM", 1: "STRESSED"},
        "risk_stats": risk_stats,
        "structure_stats": structure_stats,
        "risk_switches": risk_switches,
        "risk_switch_rate": risk_switch_rate,
        "coverage": {
            "start": df.index.min(),
            "end": df.index.max(),
            "weeks": len(df),
        },
    }


if __name__ == "__main__":
    ROOT = project_root()
    in_path = ROOT / "data" / "spy_vix_features_weekly.csv"
    out_path = ROOT / "outputs" / "weekly_regimes_hmm.csv"

    dfw = read_weekly(in_path)
    res = apply_hmm_model(dfw, HMMConfig())

    out_path.parent.mkdir(exist_ok=True)
    res["df"].to_csv(out_path)

    print(f"Saved: {out_path}")

    print("\nRisk map:")
    print(res["risk_map"])

    print("\nRisk stats (forward week returns):")
    print(res["risk_stats"].to_string())

    print("\nStructure stats (forward week returns):")
    print(res["structure_stats"].to_string())

    print("\nSwitching diagnostics (risk):")
    print(f"Risk switches: {res['risk_switches']} | Risk switch rate: {res['risk_switch_rate']:.4f}")

    cov = res["coverage"]
    print("\nCoverage:")
    print(f"Start: {cov['start']} | End: {cov['end']} | Weeks: {cov['weeks']}")