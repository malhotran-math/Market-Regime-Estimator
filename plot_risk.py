#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 22:07:40 2026

@author: malhotran
"""


# source_codes/plots/plot_risk.py
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


def shade_stress_blocks(
    ax,
    df: pd.DataFrame,
    base_alpha: float = 0.12,
    alpha_scale: float = 0.60,
) -> None:
    """
    Shade contiguous STRESSED blocks (risk_state == 1), with alpha scaled by mean risk_confidence.
    """
    # be robust to float risk_state after CSV read
    stress = df["risk_state"].astype(int) == 1
    block_id = stress.ne(stress.shift()).cumsum()

    for _, block in df.groupby(block_id):
        if int(block["risk_state"].iloc[0]) != 1:
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

        mean_conf = float(block["risk_confidence"].mean())
        alpha = base_alpha + alpha_scale * mean_conf
        alpha = min(max(alpha, 0.0), 1.0)

        ax.axvspan(start, end, color="red", alpha=alpha, linewidth=0)


def plot_risk(
    csv_path: Optional[Path] = None,
    out_path: Optional[Path] = None,
    show: bool = True,
) -> Path:
    ROOT = project_root()
    csv_path = csv_path or (ROOT / "outputs" / "weekly_regimes_threshold.csv")
    out_path = out_path or (ROOT / "outputs" / "plot_risk_3panel.png")

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True).sort_index()

    _require_cols(
        df,
        ["spy_close", "risk_state", "risk_confidence", "ewma_vol_20", "vix_close"],
    )

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(16, 9), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.5, 1]},
    )

    # --------------------------
    # Panel 1 — Price
    # --------------------------
    ax1.plot(df.index, df["gld_close"], color="orange", linewidth=1.2, label="GLD (left)")
    shade_stress_blocks(ax1, df)
    ax1.set_title("SPY + GLD with Risk Regime Overlay")
    ax1.set_ylabel("GLD close")
    ax1.grid(True, alpha=0.3)

    ax1b = ax1.twinx()
    ax1b.plot(df.index, df["spy_close"], color="black", linewidth=1.2, label="SPY (right)")
    ax1b.set_ylabel("SPY close")
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    # --------------------------
    # Panel 2 — Volatility
    # --------------------------
    ax2.plot(df.index, df["ewma_vol_20"], color="blue", linewidth=1.2, label="EWMA vol (20)")
    ax2.set_ylabel("EWMA vol", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")

    ax2b = ax2.twinx()
    ax2b.plot(df.index, df["vix_close"], color="orange", linewidth=1.0, label="VIX")
    ax2b.set_ylabel("VIX", color="orange")
    ax2b.tick_params(axis="y", labelcolor="orange")

    shade_stress_blocks(ax2, df)

    ax2.set_title("Volatility Proxies (EWMA & VIX)")
    ax2.grid(True, alpha=0.3)

    # combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    # --------------------------
    # Panel 3 — Confidence
    # --------------------------
    ax3.plot(df.index, df["risk_confidence"], color="purple", linewidth=1.2)
    shade_stress_blocks(ax3, df)

    ax3.set_title("Risk Confidence (0–1)")
    ax3.set_ylabel("Confidence")
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
    plot_risk()
    print("Message: Hopefully you liked the risk plots!") 
