"""
run_drift_comparison.py - Compare drift detectors inside the full pipeline.

Keeps the Random Forest calibrator fixed; swaps only the drift detector:
  CUSUM  (classic, hand-tuned threshold)
  ADWIN  (self-tuning, false-alarm guarantee)
  BOCPD  (Bayesian, probability of changepoint)

For each, runs the complete pipeline on all 40 sensors and reports the
adaptive-calibration accuracy (vs held-out reference) plus how many automatic
retrains each detector triggered. Better drift handling -> lower RMSE / higher R2.
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from aqcn.features import build_feature_frames
from aqcn.pipeline import AQCNPipeline
from aqcn.drift_monitor import CusumDriftMonitor
from aqcn.adwin_monitor import AdwinDriftMonitor
from aqcn.bocpd_monitor import BocpdDriftMonitor
from aqcn.ks_monitor import KsDriftMonitor
from aqcn.uncertainty_monitor import UncertaintyDriftMonitor

RESULTS = Path("results")


def agg(results, block, key):
    vals = [r[block][key] for r in results.values()
            if not r.get("skipped") and not np.isnan(r[block].get(key, np.nan))]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    RESULTS.mkdir(exist_ok=True)
    frames = build_feature_frames(use_cache=True)

    detectors = {
        "CUSUM": CusumDriftMonitor,           # watches residual mean (needs ref)
        "ADWIN": AdwinDriftMonitor,           # self-tuning window (needs ref)
        "BOCPD": BocpdDriftMonitor,           # Bayesian changepoint (needs ref)
        "KS": KsDriftMonitor,                 # distribution shift (NO ref needed)
        "Uncertainty": UncertaintyDriftMonitor,  # model confidence (NO ref needed)
    }

    summary = {}
    per_sensor_rows = []

    for name, factory in detectors.items():
        print(f"\n{'='*60}\nDRIFT DETECTOR: {name}\n{'='*60}")
        pipe = AQCNPipeline(frames, use_gap_fill=False, drift_monitor_factory=factory)
        results = pipe.run()
        ok = {s: r for s, r in results.items() if not r.get("skipped")}

        total_retrains = sum(r["n_retrains"] for r in ok.values())
        summary[name] = {
            "mean_raw_RMSE": agg(results, "raw", "RMSE"),
            "mean_adapt_RMSE": agg(results, "adaptive", "RMSE"),
            "mean_static_RMSE": agg(results, "static", "RMSE"),
            "mean_adapt_R2": agg(results, "adaptive", "R2"),
            "mean_adapt_d": agg(results, "adaptive", "d"),
            "total_retrains": total_retrains,
            "n_improved": sum(1 for r in ok.values()
                              if r["adaptive"]["RMSE"] < r["raw"]["RMSE"]),
            "n_sensors": len(ok),
        }
        for sid, r in ok.items():
            per_sensor_rows.append({
                "sensor_id": sid, "detector": name,
                "adapt_RMSE": round(r["adaptive"]["RMSE"], 4),
                "adapt_R2": round(r["adaptive"]["R2"], 4),
                "n_retrains": r["n_retrains"],
            })

    # Save per-sensor
    pd.DataFrame(per_sensor_rows).to_csv(RESULTS / "drift_comparison_per_sensor.csv",
                                         index=False)

    # Report
    L = []
    L.append("=" * 68)
    L.append("DRIFT DETECTOR COMPARISON  (Random Forest calibrator fixed)")
    L.append("Metrics: adaptive calibration on held-out reference, all 40 sensors")
    L.append("=" * 68)
    L.append(f"\n{'Detector':<10}{'rawRMSE':>9}{'staticRMSE':>12}{'adaptRMSE':>11}"
             f"{'adaptR2':>9}{'adaptD':>8}{'retrains':>10}{'improved':>10}")
    L.append("-" * 79)
    for name, s in summary.items():
        L.append(f"{name:<10}{s['mean_raw_RMSE']:>9.3f}{s['mean_static_RMSE']:>12.3f}"
                 f"{s['mean_adapt_RMSE']:>11.3f}{s['mean_adapt_R2']:>9.3f}"
                 f"{s['mean_adapt_d']:>8.3f}{s['total_retrains']:>10}"
                 f"{s['n_improved']:>8}/{s['n_sensors']}")

    # Determine winner
    best = min(summary.items(), key=lambda kv: kv[1]["mean_adapt_RMSE"])
    L.append(f"\nLowest mean adaptive RMSE: {best[0]} ({best[1]['mean_adapt_RMSE']:.3f})")
    base = summary["CUSUM"]["mean_adapt_RMSE"]
    for name, s in summary.items():
        if name == "CUSUM":
            continue
        delta = (base - s["mean_adapt_RMSE"]) / base * 100
        L.append(f"  {name} vs CUSUM: {delta:+.1f}% RMSE "
                 f"({'better' if delta > 0 else 'worse'})")

    report = "\n".join(L)
    (RESULTS / "drift_comparison.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\nSaved results/drift_comparison.txt + drift_comparison_per_sensor.csv")


if __name__ == "__main__":
    main()
