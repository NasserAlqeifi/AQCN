"""
cold_start.py - Calibration transfer for new sensors with little reference data.

A freshly deployed sensor has not yet accumulated enough paired (sensor,
reference) hours to train its own reliable model. Following the calibration-
transfer / propagation literature, it BORROWS the calibrator of the most
suitable already-calibrated donor (same city, most training data, same
instrument family preferred), then progressively refits its own model as its
reference data accumulates.

This module provides:
  pick_donor()         choose the best donor sensor for a target
  evaluate_cold_start() simulate a new sensor: hide its history, show that the
                        borrowed model calibrates it from hour one, and that a
                        self-trained model overtakes the borrowed one as data grows
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aqcn.calibrator import SensorCalibrator, temporal_split


def _instrument_family(sensor_id: str) -> str:
    # e.g. MCH__AQY872_S1 -> AQY ; MCH__Prax1_S1 -> Prax
    base = sensor_id.split("__")[-1]
    fam = "".join(ch for ch in base.split("_")[0] if not ch.isdigit())
    return fam


def pick_donor(target_id: str, frames: dict[str, pd.DataFrame],
               trained: dict[str, SensorCalibrator]) -> str | None:
    """
    Choose the best donor: same city, prefer same instrument family,
    then most training data.
    """
    t_city = target_id.split("__")[0]
    t_fam = _instrument_family(target_id)

    candidates = []
    for sid, cal in trained.items():
        if sid == target_id or cal.model is None:
            continue
        if sid.split("__")[0] != t_city:
            continue
        paired = int((frames[sid]["no2_raw"].notna()
                      & frames[sid]["ref_no2"].notna()).sum())
        same_fam = _instrument_family(sid) == t_fam
        candidates.append((same_fam, paired, sid))

    if not candidates:
        # relax: any city
        for sid, cal in trained.items():
            if sid == target_id or cal.model is None:
                continue
            paired = int((frames[sid]["no2_raw"].notna()
                          & frames[sid]["ref_no2"].notna()).sum())
            candidates.append((False, paired, sid))

    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][2]


def evaluate_cold_start(target_id: str, frames: dict[str, pd.DataFrame],
                        trained: dict[str, SensorCalibrator],
                        warmup_days: int = 14) -> dict:
    """
    Simulate target_id as brand-new: it may use a donor model immediately,
    and trains its own model once `warmup_days` of reference have accumulated.
    Compare borrowed-vs-self RMSE on the period AFTER warmup (held-out).
    """
    df = frames[target_id]
    donor_id = pick_donor(target_id, frames, trained)
    if donor_id is None:
        return {"target": target_id, "skipped": True, "reason": "no donor"}

    donor = trained[donor_id]

    # Define the "new sensor" timeline: first paired row onward
    paired_idx = np.where((df["no2_raw"].notna() & df["ref_no2"].notna()).values)[0]
    if len(paired_idx) < 24 * (warmup_days + 7):
        return {"target": target_id, "skipped": True, "reason": "too little data"}

    warmup_end_pos = paired_idx[min(int(warmup_days * 24), len(paired_idx) - 1)]

    warmup_mask = np.zeros(len(df), dtype=bool)
    warmup_mask[:warmup_end_pos] = True
    eval_mask = ~warmup_mask

    # Borrowed model applied to eval period
    borrowed_pred, _ = donor.predict(df)

    # Self model trained on warmup window only, applied to eval period
    self_cal = SensorCalibrator(target_id)
    n_self = self_cal.fit(df, warmup_mask)
    self_pred, _ = self_cal.predict(df)

    y = df["ref_no2"].values
    raw = df["no2_raw"].values

    def rmse(o, p, m):
        mask = m & ~np.isnan(o) & ~np.isnan(p)
        return float(np.sqrt(np.mean((o[mask] - p[mask]) ** 2))) if mask.sum() else float("nan")

    return {
        "target": target_id,
        "donor": donor_id,
        "skipped": False,
        "warmup_days": warmup_days,
        "n_self_train": n_self,
        "raw_rmse": rmse(y, raw, eval_mask),
        "borrowed_rmse": rmse(y, borrowed_pred, eval_mask),
        "self_rmse": rmse(y, self_pred, eval_mask),
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from aqcn.features import build_feature_frames

    frames = build_feature_frames(use_cache=True)

    # Train calibrators on all sensors (full history) as potential donors
    trained = {}
    for sid, df in frames.items():
        tr, te = temporal_split(df, 0.6)
        cal = SensorCalibrator(sid)
        if cal.fit(df, tr) >= 50:
            trained[sid] = cal

    print("\nCOLD-START TRANSFER (borrowed donor vs self-trained on 14d warmup)")
    print(f"{'Target':<22}{'Donor':<22}{'RawRMSE':>9}{'Borrow':>9}{'Self':>9}")
    print("-" * 71)
    for target in ["MCH__Prax2_S1", "MCH__AQM390_S1", "MCH__IMB2_S3", "MCH__NS1_S1"]:
        if target not in frames:
            continue
        r = evaluate_cold_start(target, frames, trained)
        if r.get("skipped"):
            print(f"{target:<22} skipped: {r['reason']}")
            continue
        print(f"{r['target']:<22}{r['donor']:<22}{r['raw_rmse']:>9.3f}"
              f"{r['borrowed_rmse']:>9.3f}{r['self_rmse']:>9.3f}")
