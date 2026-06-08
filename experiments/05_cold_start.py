"""
run_coldstart_comparison.py - Compare cold-start calibration methods.

Scenario: a brand-new sensor has only `WARMUP_DAYS` of reference data, then must
calibrate itself. We compare five strategies, all measured by RMSE on the
held-out period AFTER warmup (vs the ratified reference).

  1. Raw          - uncalibrated sensor (floor)
  2. Self-RF      - Random Forest trained on the new sensor's warmup data only
  3. Borrow-RF    - borrow the nearest DONOR sensor's full Random Forest
  4. Global-RF    - one Random Forest trained on ALL donor sensors pooled
  5. MAML         - meta-learned model, adapted on the warmup data

Leakage-free design: sensors are split into DONORS (used to build Borrow/Global/
MAML) and TEST-NEW sensors (never seen during donor training). Every method is
evaluated only on the TEST-NEW sensors.
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from aqcn.features import build_feature_frames, feature_columns
from aqcn.calibrator import SensorCalibrator
from aqcn.maml_calibrator import MamlCalibrator, matrix

RESULTS = Path("results")
WARMUP_DAYS = 14
N_TEST = 10          # number of held-out "new" sensors


def rmse(o, p):
    m = ~(np.isnan(o) | np.isnan(p))
    return float(np.sqrt(np.mean((o[m] - p[m]) ** 2))) if m.sum() else float("nan")


def main():
    RESULTS.mkdir(exist_ok=True)
    frames = build_feature_frames(use_cache=True)

    # choose TEST-NEW sensors: well-covered, spread across cities
    paired = {s: int((d["no2_raw"].notna() & d["ref_no2"].notna()).sum())
              for s, d in frames.items()}
    # sort by coverage, take a spread for test
    ranked = sorted(paired, key=lambda s: -paired[s])
    test_sensors = ranked[5:5 + N_TEST]      # skip the very top to keep donors strong
    donor_sensors = [s for s in frames if s not in test_sensors]
    print(f"Donors: {len(donor_sensors)}  Test-new: {len(test_sensors)}")

    # shared feature set (intersection so every method uses same columns)
    feats = feature_columns(frames[donor_sensors[0]])

    # ---- build donor-based models ONCE ----
    # Global-RF: pool all donor rows
    print("Training Global-RF on pooled donors...")
    Xg, yg = [], []
    donor_cals = {}
    for s in donor_sensors:
        X, y = matrix(frames[s], feats)
        if len(X) >= 50:
            Xg.append(X); yg.append(y)
            # also train each donor's own RF for the borrow option
            cal = SensorCalibrator(s)
            n = len(frames[s])
            cal.fit(frames[s], np.ones(n, dtype=bool))
            donor_cals[s] = cal
    Xg = np.concatenate(Xg); yg = np.concatenate(yg)
    from sklearn.ensemble import RandomForestRegressor
    # impute NaN in pooled features with column medians
    med = np.nanmedian(Xg, axis=0)
    Xg_i = np.where(np.isnan(Xg), med, Xg)
    global_rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=3,
                                      random_state=42, n_jobs=-1).fit(Xg_i, yg)

    # MAML: meta-train across donor tasks
    print("Meta-training MAML across donors...")
    tasks = []
    for s in donor_sensors:
        X, y = matrix(frames[s], feats)
        if len(X) >= 40:
            tasks.append((X, y))
    maml = MamlCalibrator(feats)
    maml.fit_scaler(Xg, yg)
    maml.meta_train(tasks, iterations=2000, inner_steps=5)

    def nearest_donor(target):
        tcity = target.split("__")[0]
        same = [s for s in donor_cals if s.split("__")[0] == tcity]
        pool = same or list(donor_cals)
        return max(pool, key=lambda s: paired[s])

    # ---- evaluate every method on each TEST-NEW sensor ----
    print("\nEvaluating methods on held-out new sensors...\n")
    rows = []
    for sid in test_sensors:
        df = frames[sid]
        pidx = np.where((df["no2_raw"].notna() & df["ref_no2"].notna()).values)[0]
        if len(pidx) < 24 * (WARMUP_DAYS + 7):
            continue
        warm_end = pidx[min(WARMUP_DAYS * 24, len(pidx) - 1)]
        warm = np.zeros(len(df), dtype=bool); warm[:warm_end] = True
        evalm = ~warm

        y = df["ref_no2"].values
        raw = df["no2_raw"].values
        Xall = df[feats].values.astype(float)

        # 1 Raw
        r_raw = rmse(y[evalm], raw[evalm])

        # 2 Self-RF (warmup only)
        self_cal = SensorCalibrator(sid); self_cal.fit(df, warm)
        sp, _ = self_cal.predict(df)
        r_self = rmse(y[evalm], sp[evalm])

        # 3 Borrow-RF (nearest donor full model)
        donor = nearest_donor(sid)
        bp, _ = donor_cals[donor].predict(df)
        r_borrow = rmse(y[evalm], bp[evalm])

        # 4 Global-RF
        Xi = np.where(np.isnan(Xall), med, Xall)
        gp = global_rf.predict(Xi)
        r_global = rmse(y[evalm], gp[evalm])

        # 5 MAML (adapt on warmup, predict eval)
        Xs, ys = Xall[warm & ~np.isnan(raw) & ~np.isnan(y)], None
        ws = warm & ~np.isnan(raw) & ~np.isnan(y)
        mp_eval = maml.adapt_and_predict(Xall[ws], y[ws], Xall, steps=30)
        r_maml = rmse(y[evalm], mp_eval[evalm])

        rows.append({"sensor_id": sid, "donor": donor,
                     "Raw": r_raw, "Self_RF": r_self, "Borrow_RF": r_borrow,
                     "Global_RF": r_global, "MAML": r_maml})
        print(f"  {sid:<22} raw={r_raw:6.2f}  self={r_self:6.2f}  "
              f"borrow={r_borrow:6.2f}  global={r_global:6.2f}  maml={r_maml:6.2f}")

    df_res = pd.DataFrame(rows)
    df_res.to_csv(RESULTS / "coldstart_comparison.csv", index=False)

    methods = ["Raw", "Self_RF", "Borrow_RF", "Global_RF", "MAML"]
    L = ["=" * 64,
         "COLD-START COMPARISON (RMSE vs reference on held-out new sensors)",
         f"{WARMUP_DAYS}-day warmup | {len(df_res)} new sensors | donors excluded",
         "=" * 64,
         f"\n{'Method':<12}{'mean RMSE':>12}{'median RMSE':>13}{'best/worst sensors':>22}"]
    means = {}
    for m in methods:
        vals = df_res[m].values
        means[m] = float(np.nanmean(vals))
        L.append(f"{m:<12}{np.nanmean(vals):>12.3f}{np.nanmedian(vals):>13.3f}")
    best = min(methods, key=lambda m: means[m])
    L.append(f"\nBest method (lowest mean RMSE): {best} ({means[best]:.3f})")
    base = means["Raw"]
    for m in methods:
        if m == "Raw":
            continue
        L.append(f"  {m:<10} vs Raw: {(base-means[m])/base*100:+.1f}% RMSE")
    report = "\n".join(L)
    (RESULTS / "coldstart_comparison.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    print("\nSaved results/coldstart_comparison.{csv,txt}")


if __name__ == "__main__":
    main()
