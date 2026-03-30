#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 10:37:02 2026

@author: malhotran
"""
# source_codes/plots/plot_structure.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt


def project_root() -> Path:
    # plots/ -> source_codes/ -> project root
    return Path(__file__).resolve().parents[2]


def _require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in CSV: {missing}")


def shade_trend_blocks(
    ax,
    df: pd.DataFrame,
    base_alpha: float = 0.12,
    alpha_scale: float = 0.60,
) -> None:
    """
    Shade contiguous TREND blocks (structure_state == "TREND"),
    with alpha scaled by mean structure_confidence in the block.
    """
    is_trend = df["structure_state"].astype(str) == "TREND"
    block_id = is_trend.ne(is_trend.shift()).cumsum()

    for _, block in df.groupby(block_id):
        if str(block["structure_state"].iloc[0]) != "TREND":
            continue

        start = block.index[0]
        end = block.index[-1]

        # extend to next timestamp if available (prevents last-bar truncation)
        try:
            end_pos = df.index.get_loc(end)
            if isinstance(end_pos, int) and end_pos + 1 < len(df.index):
                end = df.index[end_pos + 1]
        except Exception:
            pass

        mean_conf = float(block["structure_confidence"].mean())
        alpha = base_alpha + alpha_scale * mean_conf
        alpha = min(max(alpha, 0.0), 1.0)

        ax.axvspan(start, end, color="orange", alpha=alpha, linewidth=0)


def plot_structure(
    csv_path: Optional[Path] = None,
    out_path: Optional[Path] = None,
    show: bool = True,
) -> Path:
    ROOT = project_root()
    csv_path = csv_path or (ROOT / "outputs" / "weekly_regimes_threshold.csv")
    out_path = out_path or (ROOT / "outputs" / "plot_structure_multihorizon.png")

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True).sort_index()

    _require_cols(
        df,
        [
            "spy_close",
            "structure_state",
            "structure_confidence",
            "structure_state_20",
            "structure_state_60",
            "structure_state_200",
        ],
    )

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(16, 9), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.2, 1.0]},
    )

    # --------------------------
    # Panel 1 — Price (GLD left, SPY right) + structure shading
    # --------------------------
    ax1.plot(df.index, df["gld_close"], color="red", linewidth=1.2, label="GLD (left)")
    shade_trend_blocks(ax1, df)
    ax1.set_title("SPY + GLD with Combined Structure Regime Overlay")
    ax1.set_ylabel("GLD close")
    ax1.grid(True, alpha=0.3)
    
    ax1b = ax1.twinx()
    ax1b.plot(df.index, df["spy_close"], color="black", linewidth=1.2, label="SPY (right)")
    ax1b.set_ylabel("SPY close")
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")


    # --------------------------
    # Panel 2 — Multi-horizon structure states (20/60/200)
    # Encode TREND=1, RANGE=0; plot as three offset step lines
    # --------------------------
    state_map = {"RANGE": 0.0, "TREND": 1.0}

    s20 = df["structure_state_20"].astype(str).map(state_map)
    s60 = df["structure_state_60"].astype(str).map(state_map)
    s200 = df["structure_state_200"].astype(str).map(state_map)

    if s20.isna().any() or s60.isna().any() or s200.isna().any():
        bad = sorted(
            set(pd.concat([df["structure_state_20"], df["structure_state_60"], df["structure_state_200"]]).dropna().unique())
            - set(state_map.keys())
        )
        raise ValueError(
            "Unexpected values in structure_state_{20,60,200}. "
            f"Expected only {list(state_map.keys())}. Found extras: {bad}"
        )

    ax2.step(df.index, s20 + 0.00, where="post", linewidth=1.2, color="blue", label="20d (0/1)")
    ax2.step(df.index, s60 + 1.25, where="post", linewidth=1.2, color="red", label="60d (0/1, offset)")
    ax2.step(df.index, s200 + 2.50, where="post", linewidth=1.2, color="black", label="200d (0/1, offset)")

    shade_trend_blocks(ax2, df)

    ax2.set_title("Multi-horizon Structure States (TREND=1, RANGE=0) — offsets for readability")
    ax2.set_yticks([0, 1, 1.25, 2.25, 2.5, 3.5])
    ax2.set_yticklabels(["0", "1", "0", "1", "0", "1"])
    ax2.set_ylabel("state (offset)")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    # --------------------------
    # Panel 3 — Combined structure confidence
    # --------------------------
    ax3.plot(df.index, df["structure_confidence"], color="purple", linewidth=1.2)
    shade_trend_blocks(ax3, df)

    ax3.set_title("Combined Structure Confidence (0–1)")
    ax3.set_ylabel("confidence")
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)

    print(f"Saved plot: {out_path}")
    return out_path


if __name__ == "__main__":
    plot_structure()
    print("Message: Hopefully you liked the structure plots!") 
