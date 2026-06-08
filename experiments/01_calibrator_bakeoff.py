"""
run_calibrator_comparison.py - Bake-off of calibration models.

We had kept Random Forest fixed; this finally tests it against alternatives on
the SAME setup as the main pipeline: per-sensor, strict temporal 60/40 split,
identical features, evaluated on the held-out future window vs the reference.

Models:
  Linear     - ordinary least squares (simplest baseline)
  Ridge      - regularised linear
  kNN        - k-nearest-neighbours regression
  SVR        - support vector regression (RBF kernel)
  MLP        - feed-forward neural network
  GradBoost  - histogram gradient boosting (the XGBoost/LightGBM family)
  RandomForest - our current backbone

Tree models use raw (imputed) features; the others use standardised features.
Reported: mean RMSE, MAE, R2, Willmott d across all sensors (held-out test).
"""

from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from aqcn.features import build_feature_frames, feature_columns
from aqcn.calibrator import temporal_split
from aqcn.evaluator import all_metrics

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

RESULTS = Path("results")

# models that work on standardised features
NEEDS_SCALING = {"Linear", "Ridge", "kNN", "SVR", "MLP"}


def make_models():
    return {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "kNN": KNeighborsRegressor(n_neighbors=10, weights="distance"),
        "SVR": SVR(C=10.0, gamma="scale"),
        "MLP": MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=400,
                            early_stopping=True, random_state=42),
        "GradBoost": HistGradientBoostingRegressor(max_iter=300, random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=200, min_samples_leaf=3,
                                              random_state=42, n_jobs=-1),
    }


def main():
    RESULTS.mkdir(exist_ok=True)
    frames = build_feature_frames(use_cache=True)
    model_names = list(make_models().keys())

    # accumulate per-sensor metrics per model
    metrics_by_model = {m: [] for m in model_names}
    per_rows = []
    timing = {m: 0.0 for m in model_names}

    for i, (sid, df) in enumerate(frames.items()):
        feats = feature_columns(df)
        tr, te = temporal_split(df, 0.6)
        X = df[feats].values.astype(float)
        y = df["ref_no2"].values.astype(float)

        core = ~np.isnan(df["no2_raw"].values) & ~np.isnan(y)
        tr_ok = tr & core
        te_ok = te & core
        if tr_ok.sum() < 50 or te_ok.sum() < 20:
            continue

        # median-impute features using TRAIN medians (no leakage)
        med = np.nanmedian(X[tr_ok], axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        Ximp = np.where(np.isnan(X), med, X)

        # standardised copy for non-tree models (fit on train)
        scaler = StandardScaler().fit(Ximp[tr_ok])
        Xstd = scaler.transform(Ximp)

        for name, model in make_models().items():
            Xtr = Xstd if name in NEEDS_SCALING else Ximp
            t0 = time.time()
            try:
                model.fit(Xtr[tr_ok], y[tr_ok])
                pred = model.predict(Xtr[te_ok])
            except Exception as e:
                print(f"  {name} failed on {sid}: {e}")
                continue
            timing[name] += time.time() - t0
            m = all_metrics(y[te_ok], pred)
            metrics_by_model[name].append(m)
            per_rows.append({"sensor_id": sid, "model": name,
                             "RMSE": round(m["RMSE"], 4), "R2": round(m["R2"], 4),
                             "MAE": round(m["MAE"], 4), "d": round(m["d"], 4)})

        if (i + 1) % 10 == 0:
            print(f"  processed {i+1}/{len(frames)} sensors")

    pd.DataFrame(per_rows).to_csv(RESULTS / "calibrator_comparison_per_sensor.csv",
                                  index=False)

    def agg(ms, k):
        v = [m[k] for m in ms if not np.isnan(m.get(k, np.nan))]
        return float(np.mean(v)) if v else float("nan")

    L = ["=" * 70,
         "CALIBRATION MODEL BAKE-OFF",
         "Per-sensor, strict temporal 60/40 split, held-out test vs reference",
         f"{len(metrics_by_model[model_names[0]])} sensors evaluated",
         "=" * 70,
         f"\n{'Model':<14}{'RMSE':>8}{'MAE':>8}{'R2':>8}{'d':>8}{'R2>0.5':>9}{'fit time(s)':>12}"]
    summary = {}
    for name in model_names:
        ms = metrics_by_model[name]
        rmse = agg(ms, "RMSE"); mae = agg(ms, "MAE")
        r2 = agg(ms, "R2"); d = agg(ms, "d")
        n_good = sum(1 for m in ms if m["R2"] > 0.5)
        summary[name] = rmse
        L.append(f"{name:<14}{rmse:>8.3f}{mae:>8.3f}{r2:>8.3f}{d:>8.3f}"
                 f"{n_good:>7}/{len(ms)}{timing[name]:>12.1f}")

    best = min(summary, key=lambda k: summary[k] if not np.isnan(summary[k]) else 1e9)
    L.append(f"\nLowest mean RMSE: {best} ({summary[best]:.3f})")
    rf = summary.get("RandomForest", float("nan"))
    L.append(f"\nvs RandomForest (our backbone, {rf:.3f}):")
    for name in model_names:
        if name == "RandomForest":
            continue
        delta = (rf - summary[name]) / rf * 100
        tag = "better" if delta > 0 else "worse"
        L.append(f"  {name:<12}: {delta:+.1f}% RMSE ({tag})")

    report = "\n".join(L)
    (RESULTS / "calibrator_comparison.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    print("\nSaved results/calibrator_comparison.{csv,txt}")


if __name__ == "__main__":
    main()
