"""
run_fault_comparison_cal.py - Fault-detector benchmark on CALIBRATED signals.

The raw-signal benchmark was unfair to the consensus/ensemble detectors: raw
cheap sensors disagree wildly (unit-to-unit gain/offset), so consensus flagged
everything. Here we first CALIBRATE every sensor (Random Forest), so healthy
sensors agree, then:
  * inject the same synthetic faults into each test sensor's CALIBRATED signal,
  * build consensus from PEERS' CALIBRATED signals,
  * run all 7 detectors and score precision / recall / F1.

This isolates the real question: when the healthy baseline agrees, how well does
each detector catch genuine faults?
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from aqcn.features import build_feature_frames
from aqcn.calibrator import SensorCalibrator, temporal_split
from aqcn.fault_injection import inject_faults
from aqcn import fault_detectors as fd

RESULTS = Path("results")

DETECTORS = {
    "Z-score": lambda sig, ref, cons: fd.detect_zscore(sig),
    "IQR": lambda sig, ref, cons: fd.detect_iqr(sig),
    "Consensus": lambda sig, ref, cons: fd.detect_consensus(sig, cons),
    "Ensemble(ours)": lambda sig, ref, cons: fd.detect_ensemble(sig, ref, cons),
    "IsolationForest": lambda sig, ref, cons: fd.detect_iforest(sig),
    "OneClassSVM": lambda sig, ref, cons: fd.detect_ocsvm(sig),
    "Autoencoder": lambda sig, ref, cons: fd.detect_autoencoder(sig),
}


def main():
    RESULTS.mkdir(exist_ok=True)
    frames = build_feature_frames(use_cache=True)
    rng = np.random.default_rng(42)

    by_city = {}
    for s in frames:
        by_city.setdefault(s.split("__")[0], []).append(s)

    # --- calibrate every sensor once (RF on first 60%, predict full) ---
    print("Calibrating all sensors (for calibrated consensus)...")
    calibrated = {}
    for s, df in frames.items():
        tr, te = temporal_split(df, 0.6)
        cal = SensorCalibrator(s)
        if cal.fit(df, tr) >= 50:
            pred, _ = cal.predict(df)
            calibrated[s] = pd.Series(pred, index=df.index)
    print(f"Calibrated {len(calibrated)} sensors.")

    paired = {s: int((d["no2_raw"].notna() & d["ref_no2"].notna()).sum())
              for s, d in frames.items()}
    test_sensors = [s for s in calibrated
                    if paired[s] > 5000 and len(by_city[s.split("__")[0]]) >= 3]
    print(f"Test sensors: {len(test_sensors)}")

    agg = {name: {"tp": 0, "fp": 0, "fn": 0} for name in DETECTORS}
    per_rows = []

    for sid in test_sensors:
        df = frames[sid]
        city = sid.split("__")[0]
        sig = calibrated[sid].values.astype(float)             # CALIBRATED signal
        ref = df["ref_no2"].values.astype(float)
        peers = [p for p in by_city[city] if p != sid and p in calibrated]
        if not peers:
            continue
        cons = pd.concat([calibrated[p].reindex(df.index) for p in peers],
                         axis=1).median(axis=1).values          # CALIBRATED consensus

        faulty, labels, events = inject_faults(sig, rng, n_events=4)
        if labels.sum() == 0:
            continue

        for name, fn_det in DETECTORS.items():
            try:
                pred = fn_det(faulty, ref, cons)
            except Exception as e:
                print(f"  {name} failed on {sid}: {e}")
                continue
            sc = fd.score(pred, labels)
            agg[name]["tp"] += sc["tp"]; agg[name]["fp"] += sc["fp"]
            agg[name]["fn"] += sc["fn"]
            per_rows.append({"sensor_id": sid, "detector": name,
                             "precision": round(sc["precision"], 3),
                             "recall": round(sc["recall"], 3),
                             "f1": round(sc["f1"], 3)})
        print(f"  injected into {sid} ({labels.sum()} fault hours)")

    pd.DataFrame(per_rows).to_csv(
        RESULTS / "fault_comparison_calibrated_per_sensor.csv", index=False)

    L = ["=" * 60,
         "FAULT DETECTOR COMPARISON - CALIBRATED signals (fair test)",
         "Consensus/ensemble now use calibrated peers that agree when healthy.",
         "=" * 60,
         f"\n{'Detector':<18}{'Precision':>11}{'Recall':>9}{'F1':>9}"]
    summary = {}
    for name, c in agg.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = (2 * prec * rec / (prec + rec)
              if prec and rec and not np.isnan(prec + rec) else float("nan"))
        summary[name] = f1
        L.append(f"{name:<18}{prec:>11.3f}{rec:>9.3f}{f1:>9.3f}")
    best = max(summary, key=lambda k: summary[k] if not np.isnan(summary[k]) else -1)
    L.append(f"\nBest detector by F1: {best} ({summary[best]:.3f})")
    report = "\n".join(L)
    (RESULTS / "fault_comparison_calibrated.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    print("\nSaved results/fault_comparison_calibrated.{csv,txt}")


if __name__ == "__main__":
    main()
