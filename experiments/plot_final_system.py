"""
plot_final_system.py - the honest, per-sensor picture of the final system.

Reads results/final_system_per_sensor.csv and writes
results/per_sensor_performance.png: every sensor, cal1 vs calibrated, for both R²
and RMSE. This view is the point - calibration lifts essentially every sensor -
and it makes plain why a *mean* over sensors is misleading (a couple of cal1
baselines are badly behaved, so the per-sensor view, not the average, is the
right summary).

    python experiments/plot_final_system.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path("results")
RAW_C, CAL_C = "#e0a06a", "#2a72b5"


def main():
    df = pd.read_csv(RESULTS / "final_system_per_sensor.csv")
    df = df.sort_values("cal_R2", ascending=False).reset_index(drop=True)
    x = np.arange(len(df))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    # ---- R² panel (raw clipped to -1 so the broken baselines don't crush it) ----
    raw_r2 = df["raw_R2"].clip(lower=-1.0)
    ax1.axhline(0.0, color="0.6", lw=0.8)
    ax1.axhline(0.9, color="green", ls="--", lw=0.9, label="reference-grade (R²=0.9)")
    ax1.bar(x - 0.2, raw_r2, width=0.4, color=RAW_C, label="cal1 (clipped at -1)")
    ax1.bar(x + 0.2, df["cal_R2"], width=0.4, color=CAL_C, label="calibrated")
    ax1.set_ylim(-1.1, 1.0)
    ax1.set_ylabel("R²  (held-out future test)")
    ax1.set_title(f"AQCN per-sensor performance - cal1 vs calibrated  (n={len(df)} sensors)")
    ax1.legend(loc="lower left", fontsize=8)

    # ---- RMSE panel (log scale: cal1 spans ~2 to 360 ppb) ----
    ax2.bar(x - 0.2, df["raw_RMSE"], width=0.4, color=RAW_C, label="cal1")
    ax2.bar(x + 0.2, df["cal_RMSE"], width=0.4, color=CAL_C, label="calibrated")
    ax2.set_yscale("log")
    ax2.set_ylabel("RMSE  ppb  (log scale)")
    ax2.set_xlabel("sensor  (sorted by calibrated R²)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(df["sensor_id"], rotation=90, fontsize=6)
    ax2.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    out = RESULTS / "per_sensor_performance.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}  ({len(df)} sensors)")


if __name__ == "__main__":
    main()
