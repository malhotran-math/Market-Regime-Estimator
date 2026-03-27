#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 16:00:21 2026

@author: malhotran
"""

# source_codes/models/threshold_model.py 
from __future__ import annotations 


"""
This is the first model that we use to estimate the regimes, called the "Threshold model". 
As the name suggests, we simply use the features we computed to estimate the regimes for 
    the following week, and determine thresholds that classify Calm/Stressed and Ranging/Trending. 

We additionally use our features and scores to compute the confidence, 
    and use risk stats to determine the overall accurace of this model. 

As a preliminary comment, note that threshold models can be inaccurate on quite a few instances 
    and are highly dependent on the threshold parameters. 
    
Outputs:
  - risk_state: 0=CALM, 1=STRESSED
  - structure_state: TREND / RANGE
  - risk_confidence, structure_confidence in [0,1]
  - evaluation stats on forward-week returns (if fwd_week_ret exists)
"""


from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

#Configuration dataclass. This is the parameter storage, tune here or apply ML layer here
@dataclass(frozen=True)
class ThresholdConfig:
    # ---- Risk axis inputs ----
    ewma_col: str = "ewma_vol_20"
    atr_col: str = "atr_14"
    px_col: str = "spy_close"
    vix_col: str = "vix_close"
    dd_col: str = "dd_252"

    # ---- Weekly evaluation target ----
    fwd_ret_col: str = "fwd_week_ret"

    # ---- Normalization (causal) ----
    z_window: int = 260
    z_min_periods: int = 52

    # ---- Risk rule ----
    risk_theta: float = 1.0
    w_vol: float = 0.3
    w_atr: float = 0.3
    w_dd: float = 0.4

    # Optional VIX gate (override)
    use_vix_gate: bool = True
    vix_theta: float = 30.0
    vix_k: float = 10.0

    # Optional drawdown gate
    use_dd_gate: bool = False
    dd_theta: float = -0.10
    dd_k: float = 0.10

    # Confidence scaling for risk_score distance-to-threshold
    risk_k: float = 1.0

    # ---- Structure rule ----
    horizons: Tuple[int, ...] = (20, 60, 200)
    snr_prefix: str = "snr_"                 # expects snr_20, snr_60, snr_200
    trend_prefix: str = "trend_score_"       # expects trend_score_20, ...
    trend_tstat_prefix: str = "trend_tstat_" # optional alt signal
    use_tstat_for_structure: bool = False

    use_continuous_structure_vote: bool = True

    # Confidence from vote-distance
    structure_vote_k: float = 0.20

    # Structure hysteresis
    use_structure_hysteresis: bool = True
    structure_upper: float = 0.60
    structure_lower: float = 0.40

    # Risk hysteresis
    use_risk_hysteresis: bool = True
    risk_upper: float = 1.15
    risk_lower: float = 0.85
    risk_hyst_k: float = 1.0

    # Per-horizon thresholds
    snr_theta: Dict[int, float] = field(default_factory=lambda: {20: 0.7, 60: 0.8, 200: 0.9})
    snr_k: Dict[int, float] = field(default_factory=lambda: {20: 0.5, 60: 0.5, 200: 0.5})

    # Structure vote weights
    weights: Dict[int, float] = field(default_factory=lambda: {20: 0.15, 60: 0.35, 200: 0.50})


# ----------------------------
#helper fns
# ----------------------------

def _clip01(x: pd.Series | np.ndarray | float) -> pd.Series | float:
    return np.clip(x, 0.0, 1.0)


def rolling_zscore(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    mu = s.rolling(window=window, min_periods=min_periods).mean()
    sd = s.rolling(window=window, min_periods=min_periods).std()
    z = (s - mu) / sd
    return z.replace([np.inf, -np.inf], np.nan)


def distance_confidence(x: pd.Series, theta: float, k: float) -> pd.Series:
    return pd.Series(_clip01((x - theta).abs() / k), index=x.index)


def hysteresis_confidence(
    score: pd.Series,
    state: pd.Series,
    lower: float,
    upper: float,
    k: float,
) -> pd.Series:
    """
    Confidence = how deep we are inside the current hysteresis state.
    Uses midpoint as a symmetric reference.
    """
    mid = (lower + upper) / 2.0
    calm_conf = (mid - score) / k
    stress_conf = (score - mid) / k
    conf = np.where(state.astype(int).values == 0, calm_conf.values, stress_conf.values)
    return pd.Series(_clip01(conf), index=score.index)


def _normalize_weights(weights: Dict[int, float]) -> Dict[int, float]:
    w_sum = float(sum(weights.values()))
    if w_sum <= 0:
        raise ValueError("weights must sum to a positive value")
    if abs(w_sum - 1.0) < 1e-12:
        return weights
    return {k: v / w_sum for k, v in weights.items()}


# ----------------------------
#structure computation
# ----------------------------

def compute_structure_combined(
    df: pd.DataFrame,
    cfg: ThresholdConfig,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Returns:
      vote: in [0,1]
      state: "TREND"/"RANGE" (with optional hysteresis)
      confidence: in [0,1] based on vote-distance

    If cfg.use_continuous_structure_vote:
      vote = Σ w_h * trend_score_h (or trend_tstat_h)
    Else:
      vote = Σ w_h * 1[state_h == "TREND"]   (state_h computed from snr thresholds)
    """
    weights = _normalize_weights(cfg.weights)
    vote = pd.Series(0.0, index=df.index)

    if cfg.use_continuous_structure_vote:
        for h, w in weights.items():
            col = f"{cfg.trend_tstat_prefix}{h}" if cfg.use_tstat_for_structure else f"{cfg.trend_prefix}{h}"
            if col not in df.columns:
                raise KeyError(f"Missing '{col}' for continuous structure vote.")
            vote += w * df[col].astype(float)
    else:
        for h, w in weights.items():
            state_col = f"structure_state_{h}"
            if state_col not in df.columns:
                raise KeyError(f"Missing '{state_col}' for discrete structure vote.")
            vote += w * (df[state_col] == "TREND").astype(float)

    vote = pd.Series(_clip01(vote), index=df.index).rename("structure_vote")

    conf = ((vote - 0.5).abs() / cfg.structure_vote_k)
    conf = pd.Series(_clip01(conf), index=df.index).rename("structure_confidence")

    if not cfg.use_structure_hysteresis:
        state = pd.Series(np.where(vote >= 0.5, "TREND", "RANGE"), index=df.index, name="structure_state")
        return vote, state, conf

    states: list[str] = []
    prev = "RANGE"
    for v in vote.values:
        if v >= cfg.structure_upper:
            prev = "TREND"
        elif v <= cfg.structure_lower:
            prev = "RANGE"
        states.append(prev)

    state = pd.Series(states, index=df.index, name="structure_state")
    return vote, state, conf


# ----------------------------
#threshold model
# ----------------------------

def apply_threshold_model(df_weekly: pd.DataFrame, cfg: ThresholdConfig = ThresholdConfig()) -> dict:
    df = df_weekly.copy()

    # --- Basic checks
    for c in (cfg.ewma_col, cfg.atr_col, cfg.px_col, cfg.dd_col):
        if c not in df.columns:
            raise KeyError(f"Missing required column '{c}' in weekly df.")

    # --- Risk features
    df["atr_pct"] = df[cfg.atr_col] / df[cfg.px_col]

    z_vol = rolling_zscore(df[cfg.ewma_col], cfg.z_window, cfg.z_min_periods)
    z_atr = rolling_zscore(df["atr_pct"], cfg.z_window, cfg.z_min_periods)
    z_dd = rolling_zscore(-df[cfg.dd_col], cfg.z_window, cfg.z_min_periods)  # negate: deeper dd => higher risk

    df["risk_score"] = cfg.w_vol * z_vol + cfg.w_atr * z_atr + cfg.w_dd * z_dd

    # Base risk state
    if not cfg.use_risk_hysteresis:
        df["risk_state"] = (df["risk_score"] > cfg.risk_theta).astype(int)
    else:
        states: list[int] = []
        prev = 0
        for v in df["risk_score"].values:
            if v > cfg.risk_upper:
                prev = 1
            elif v < cfg.risk_lower:
                prev = 0
            states.append(prev)
        df["risk_state"] = states

    df["risk_state_label"] = df["risk_state"].map({0: "CALM", 1: "STRESSED"})
    df["risk_gate"] = "score"

    # Risk confidence
    if cfg.use_risk_hysteresis:
        df["risk_confidence"] = hysteresis_confidence(
            score=df["risk_score"],
            state=df["risk_state"],
            lower=cfg.risk_lower,
            upper=cfg.risk_upper,
            k=cfg.risk_hyst_k,
        )
    else:
        df["risk_confidence"] = distance_confidence(df["risk_score"], cfg.risk_theta, cfg.risk_k)

    # VIX gate
    if cfg.use_vix_gate:
        if cfg.vix_col not in df.columns:
            raise KeyError(f"use_vix_gate=True but missing '{cfg.vix_col}' in weekly df.")
        vix_hit = df[cfg.vix_col] > cfg.vix_theta
        if vix_hit.any():
            df.loc[vix_hit, "risk_state"] = 1
            df.loc[vix_hit, "risk_state_label"] = "STRESSED"
            df.loc[vix_hit, "risk_gate"] = "vix"
            df.loc[vix_hit, "risk_confidence"] = distance_confidence(
                df.loc[vix_hit, cfg.vix_col], cfg.vix_theta, cfg.vix_k
            )

    # Drawdown gate
    if cfg.use_dd_gate:
        dd_hit = df[cfg.dd_col] < cfg.dd_theta
        if dd_hit.any():
            df.loc[dd_hit, "risk_state"] = 1
            df.loc[dd_hit, "risk_state_label"] = "STRESSED"
            df.loc[dd_hit, "risk_gate"] = "dd"
            df.loc[dd_hit, "risk_confidence"] = distance_confidence(
                df.loc[dd_hit, cfg.dd_col], cfg.dd_theta, cfg.dd_k
            )

    # --- Structure per horizon (needed for diagnostics / discrete vote fallback)
    for h in cfg.horizons:
        snr_col = f"{cfg.snr_prefix}{h}"
        if snr_col not in df.columns:
            raise KeyError(f"Missing '{snr_col}' in weekly df.")

        theta_h = cfg.snr_theta[h]
        k_h = cfg.snr_k[h]

        state_col = f"structure_state_{h}"
        conf_col = f"structure_confidence_{h}"

        df[state_col] = np.where(df[snr_col] > theta_h, "TREND", "RANGE")
        df[conf_col] = distance_confidence(df[snr_col], theta_h, k_h)

        # ensure the continuous vote columns exist if you intend to use them
        cont_col = f"{cfg.trend_tstat_prefix}{h}" if cfg.use_tstat_for_structure else f"{cfg.trend_prefix}{h}"
        if cont_col not in df.columns:
            raise KeyError(f"Missing '{cont_col}' in weekly df (required for structure vote).")

    # --- Combined structure output
    vote, state, conf = compute_structure_combined(df, cfg)
    df["structure_vote"] = vote
    df["structure_state"] = state
    df["structure_confidence"] = conf

    # --- Maps
    risk_map = {0: "CALM", 1: "STRESSED"}

    # --- Evaluation stats
    if cfg.fwd_ret_col in df.columns:
        g = df.groupby("risk_state")[cfg.fwd_ret_col]
        risk_stats = g.agg(count="count", mean="mean", median="median", std="std", min="min")
        risk_stats["q05"] = g.quantile(0.05)
        risk_stats["hit_rate_pos"] = (df[cfg.fwd_ret_col] > 0).groupby(df["risk_state"]).mean()

        g2 = df.groupby("structure_state")[cfg.fwd_ret_col]
        structure_stats = g2.agg(count="count", mean="mean", median="median", std="std", min="min")
        structure_stats["q05"] = g2.quantile(0.05)
        structure_stats["hit_rate_pos"] = (df[cfg.fwd_ret_col] > 0).groupby(df["structure_state"]).mean()
    else:
        risk_stats = pd.DataFrame()
        structure_stats = pd.DataFrame()

    return {
        "df": df,
        "risk_map": risk_map,
        "risk_stats": risk_stats,
        "structure_stats": structure_stats,
    }


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[2]  # project root (source_codes/)
    in_path = ROOT / "data" / "spy_vix_features_weekly.csv"
    out_path = ROOT / "outputs" / "weekly_regimes_threshold.csv"

    dfw = pd.read_csv(in_path, index_col=0, parse_dates=True).sort_index()
    cfg = ThresholdConfig()
    res = apply_threshold_model(dfw, cfg)

    out_path.parent.mkdir(exist_ok=True)
    res["df"].to_csv(out_path)
    print(f"Saved: {out_path}")

    print("\nRisk map:")
    print(res["risk_map"])

    print("\nRisk stats (forward week returns):")
    print(res["risk_stats"].to_string())

    print("\nStructure stats (forward week returns):")
    print(res["structure_stats"].to_string())

    df_out = res["df"]

    df_out["joint"] = df_out["risk_state_label"] + "_" + df_out["structure_state"]
    joint_stats = df_out.groupby("joint")["fwd_week_ret"].agg(
        count="count", mean="mean", median="median", std="std", min="min"
    )
    joint_stats["q05"] = df_out.groupby("joint")["fwd_week_ret"].quantile(0.05)

    print("\nJoint stats (risk x structure; forward week returns):")
    print(joint_stats.to_string())

    risk_switches = df_out["risk_state"].ne(df_out["risk_state"].shift(1)).sum()
    risk_switch_rate = risk_switches / len(df_out)

    structure_switches = df_out["structure_state"].ne(df_out["structure_state"].shift(1)).sum()
    structure_switch_rate = structure_switches / len(df_out)

    print("\nSwitching diagnostics:")
    print(f"Risk switches: {risk_switches} | Risk switch rate: {risk_switch_rate:.4f}")
    print(f"Structure switches: {structure_switches} | Structure switch rate: {structure_switch_rate:.4f}")

    corr_mat = pd.crosstab(df_out["risk_state_label"], df_out["structure_state"], normalize="all")
    print("\nRisk x Structure association matrix (joint probabilities):")
    print(corr_mat.to_string())

    print("\nTrend rate per horizon (overall):")
    for h in cfg.horizons:
        col = f"structure_state_{h}"
        print(f"{h}: {(df_out[col] == 'TREND').mean():.6f}")

    print("\nTrend rate per horizon by joint regime:")
    for h in cfg.horizons:
        col = f"structure_state_{h}"
        trend_rate_by_joint = (df_out[col] == "TREND").groupby(df_out["joint"]).mean().sort_index()
        print(f"\nHorizon {h}:")
        print(trend_rate_by_joint.to_string())

    print("\nRisk confidence by risk_state:")
    print(df_out.groupby("risk_state_label")["risk_confidence"].agg(["mean", "std"]).to_string())

    print("\nStructure confidence by structure_state:")
    print(df_out.groupby("structure_state")["structure_confidence"].agg(["mean", "std"]).to_string())
    
    corr_mat = pd.crosstab(df_out["risk_state_label"], df_out["structure_state"], normalize="all")

    print("\nRisk x Structure association matrix (joint probabilities):")
    print(corr_mat.to_string())
    
    # Binary correlation (phi coefficient)
    risk_bin = df_out["risk_state"].astype(int)
    struct_bin = (df_out["structure_state"] == "TREND").astype(int)

    phi = risk_bin.corr(struct_bin)
    
    print("\nPhi coefficient (binary correlation between Risk and Structure):")
    print(f"phi = {phi:.4f}")


    
    print("Message: Threshold model successfully compiled. Please run plot_risk.py and plot_structure.py to visualise risk and structure regimes.")
