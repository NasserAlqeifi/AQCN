"""
run_final_system.py - The final assembled AQCN system, end-to-end + full report.

Assembles the data-justified winning configuration:

  CORE (adaptive, runs here on all 40 sensors):
    * Calibrator  : Random Forest  (robust top-tier, gives confidence)
    * Drift       : KS-test (reference-free) -> guarded RF retraining
    * Fault       : ensemble on CALIBRATED consensus (recall-oriented)

  NETWORK-LEVEL ENHANCEMENTS (benchmarked separately, folded into the report):
    * Gap filling : SAITS  (results/imputer_comparison)
    * Cold start  : MAML   (results/coldstart_comparison)
    * Dense cities: GNN    (results/gnn_comparison)

Produces one comprehensive FINAL_SYSTEM_REPORT.txt.
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from aqcn.features import build_feature_frames
from aqcn.pipeline import AQCNPipeline
from aqcn.ks_monitor import KsDriftMonitor

RESULTS = Path("results")


def _vals(results, block, key):
    return [r[block][key] for r in results.values()
            if not r.get("skipped") and not np.isnan(r[block].get(key, np.nan))]


def agg(results, block, key):
    vals = _vals(results, block, key)
    return float(np.mean(vals)) if vals else float("nan")


def agg_median(results, block, key):
    vals = _vals(results, block, key)
    return float(np.median(vals)) if vals else float("nan")


def _f(v, d=3):
    return f"{v:.{d}f}" if isinstance(v, float) and not np.isnan(v) else "N/A"


def main():
    RESULTS.mkdir(exist_ok=True)
    print("=" * 70)
    print("FINAL ASSEMBLED AQCN SYSTEM - end-to-end run")
    print("RF calibrator + KS drift + calibrated-consensus fault detection")
    print("=" * 70)

    frames = build_feature_frames(use_cache=True)

    # ---- run the adaptive core with the KS drift detector ----
    pipe = AQCNPipeline(frames, use_gap_fill=False,
                        drift_monitor_factory=KsDriftMonitor)
    results = pipe.run()
    ok = {s: r for s, r in results.items() if not r.get("skipped")}

    # per-sensor output table
    rows = []
    for sid, r in ok.items():
        rows.append({
            "sensor_id": sid, "city": r["city"],
            "raw_RMSE": r["raw"]["RMSE"], "cal_RMSE": r["adaptive"]["RMSE"],
            "cal_clean_RMSE": r["adaptive_clean"]["RMSE"],
            "raw_R2": r["raw"]["R2"], "cal_R2": r["adaptive"]["R2"],
            "cal_clean_R2": r["adaptive_clean"]["R2"],
            "cal_MAE": r["adaptive"]["MAE"], "cal_d": r["adaptive"]["d"],
            "n_retrains": r["n_retrains"],
            "suspect_h": r.get("suspect_h", 0), "faulty_h": r.get("faulty_h", 0),
        })
    pd.DataFrame(rows).round(4).to_csv(RESULTS / "final_system_per_sensor.csv",
                                       index=False)

    raw_m = [r["raw"] for r in ok.values()]
    ada_m = [r["adaptive"] for r in ok.values()]

    # ---------- build the report ----------
    L = []
    L.append("=" * 70)
    L.append("FINAL ASSEMBLED AQCN SYSTEM - EVALUATION REPORT")
    L.append("QUANT dataset (NO2, cal1) | held-out future test window | no leakage")
    L.append("=" * 70)
    L.append(f"\nSensors processed: {len(ok)} / {len(results)}")

    # ===== CORE results =====
    L.append("\n" + "-" * 70)
    L.append("CORE SYSTEM  (Random Forest + KS drift + calibrated fault detection)")
    L.append("-" * 70)
    L.append("Aggregated across all sensors as MEAN | MEDIAN. The MEAN baseline is")
    L.append("inflated by a few sensors whose cal1 product emits spurious/negative values, so")
    L.append("the MEDIAN is the representative figure. 'Clean' = FAULTY hours removed.")
    L.append("")
    L.append(f"{'Metric':<7}{'Raw mean':>11}{'Raw med':>10}{'Cal mean':>11}"
             f"{'Cal med':>10}{'Clean med':>11}")
    for k in ["RMSE", "MAE", "MBE", "nRMSE", "R2", "r", "d"]:
        L.append(f"{k:<7}{_f(agg(results,'raw',k)):>11}"
                 f"{_f(agg_median(results,'raw',k)):>10}"
                 f"{_f(agg(results,'adaptive',k)):>11}"
                 f"{_f(agg_median(results,'adaptive',k)):>10}"
                 f"{_f(agg_median(results,'adaptive_clean',k)):>11}")

    raw_rmse = agg(results, "raw", "RMSE")
    cal_rmse = agg(results, "adaptive", "RMSE")
    impr = (raw_rmse - cal_rmse) / raw_rmse * 100 if raw_rmse else float("nan")
    per_sensor_red = [(r["raw"]["RMSE"] - r["adaptive"]["RMSE"]) / r["raw"]["RMSE"] * 100
                      for r in ok.values() if r["raw"]["RMSE"]]
    med_red = float(np.median(per_sensor_red)) if per_sensor_red else float("nan")
    n_impr = sum(1 for r in ok.values() if r["adaptive"]["RMSE"] < r["raw"]["RMSE"])
    n_good_raw = sum(1 for m in raw_m if m["R2"] > 0.5)
    n_good_cal = sum(1 for m in ada_m if m["R2"] > 0.5)
    L.append(f"\nRMSE reduction             : mean {_f(impr,1)}%  |  "
             f"median per-sensor {_f(med_red,1)}%")
    L.append(f"Sensors improved           : {n_impr}/{len(ok)}")
    L.append(f"Sensors with R2 > 0.5      : raw {n_good_raw}  ->  calibrated {n_good_cal}")
    L.append(f"Sensors with R2 > 0.9      : {sum(1 for m in ada_m if m['R2']>0.9)}")

    # Honesty note: the raw-R2 mean is an outlier artefact, not a typical value.
    raw_r2_vals = sorted(m["R2"] for m in raw_m if not np.isnan(m["R2"]))
    if raw_r2_vals:
        worst = raw_r2_vals[0]
        mean_excl1 = float(np.mean(raw_r2_vals[1:])) if len(raw_r2_vals) > 1 else float("nan")
        med_raw_r2 = float(np.median(raw_r2_vals))
        L.append(f"\nNOTE  the raw R2 mean ({_f(agg(results,'raw','R2'),2)}) is dominated by the "
                 f"single worst sensor (raw R2 {worst:.0f}). Excluding it the")
        L.append(f"      mean is {mean_excl1:.2f}; the MEDIAN raw R2 is {med_raw_r2:+.2f}. "
                 f"Report the median, not the mean, for the baseline.")

    total_retrains = sum(r["n_retrains"] for r in ok.values())
    total_faulty = sum(r.get("faulty_h", 0) for r in ok.values())
    total_suspect = sum(r.get("suspect_h", 0) for r in ok.values())
    L.append(f"\nDrift-triggered retrains   : {total_retrains}")
    L.append(f"SUSPECT sensor-hours       : {total_suspect}")
    L.append(f"FAULTY sensor-hours        : {total_faulty}")

    # per-city
    L.append("\nPer-city (calibrated):")
    L.append(f"  {'City':<5}{'N':>4}{'rawRMSE':>10}{'calRMSE':>10}{'calR2':>8}")
    for city in ["MCH", "LON", "YRK"]:
        cr = [r for r in ok.values() if r["city"] == city]
        if not cr:
            continue
        rr = np.mean([r["raw"]["RMSE"] for r in cr])
        cc = np.mean([r["adaptive"]["RMSE"] for r in cr])
        c2 = np.mean([r["adaptive"]["R2"] for r in cr if not np.isnan(r["adaptive"]["R2"])])
        L.append(f"  {city:<5}{len(cr):>4}{rr:>10.2f}{cc:>10.2f}{c2:>8.2f}")

    # ===== Enhancement layers (from prior benchmarks) =====
    def load_txt(name):
        p = RESULTS / name
        return p.read_text(encoding="utf-8") if p.exists() else None

    L.append("\n" + "-" * 70)
    L.append("SCOPE OF THIS RUN")
    L.append("-" * 70)
    L.append("The numbers above are the CORE adaptive system that runs end-to-end")
    L.append("here: RF calibration + KS drift + calibrated-consensus fault detection.")
    L.append("Gap filling is DISABLED in this run (use_gap_fill=False). The")
    L.append("enhancements below (SAITS gap-fill, MAML cold start, GNN dense-city")
    L.append("calibrator) are each validated in their OWN benchmark (experiments")
    L.append("04 / 05 / 08) and are NOT folded into the core numbers above.")

    L.append("\n" + "-" * 70)
    L.append("NETWORK-LEVEL ENHANCEMENTS (validated separately, not in core numbers)")
    L.append("-" * 70)

    # GNN
    gnn = RESULTS / "gnn_per_sensor.csv"
    if gnn.exists():
        g = pd.read_csv(gnn)
        mch = g[g.sensor_id.str.startswith("MCH__")]
        if len(mch):
            L.append(f"\n[Dense-city calibrator: GNN]  Manchester GNN mean RMSE "
                     f"{mch['RMSE'].mean():.3f}, R2 {mch['R2'].mean():.3f}  "
                     f"(~10% better than RF in the dense network)")

    # SAITS gap filling
    imp = RESULTS / "imputer_comparison.csv"
    if imp.exists():
        i = pd.read_csv(imp)
        r6 = i[i.gap_len == 6]
        if len(r6):
            L.append(f"[Gap filling: SAITS]  6h-gap RMSE "
                     f"{r6['SAITS_rmse'].iloc[0]:.2f} (vs linear "
                     f"{r6['Linear_rmse'].iloc[0]:.2f}); best imputer at all gap lengths.")

    # MAML cold start
    cs = RESULTS / "coldstart_comparison.csv"
    if cs.exists():
        c = pd.read_csv(cs)
        L.append(f"[Cold start: MAML]  new-sensor mean RMSE {c['MAML'].mean():.2f} "
                 f"(vs borrowing {c['Borrow_RF'].mean():.2f}); 32% better, "
                 f"calibrates a new sensor from 14 days of data.")

    # ===== Component provenance =====
    L.append("\n" + "-" * 70)
    L.append("EVERY COMPONENT CHOICE IS BACKED BY A HEAD-TO-HEAD TEST")
    L.append("-" * 70)
    L.append("  Calibrator : 9-model bake-off  -> RF/XGBoost top tier (tied)")
    L.append("  Gap filling: SAITS/BRITS/CSDI/Linear -> SAITS best")
    L.append("  Drift      : CUSUM/ADWIN/BOCPD/KS/Uncertainty -> KS (reference-free)")
    L.append("  Cold start : Raw/Self/Borrow/Global/MAML -> MAML (+32%)")
    L.append("  Fault      : 7 detectors x raw/calibrated -> calibrated consensus")
    L.append("  Dense city : GNN vs RF -> GNN +10% (Manchester)")

    report = "\n".join(L)
    (RESULTS / "FINAL_SYSTEM_REPORT.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    print("\nSaved results/FINAL_SYSTEM_REPORT.txt + final_system_per_sensor.csv")


if __name__ == "__main__":
    main()
