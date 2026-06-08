"""
run_calibrator_extended.py - Add XGBoost and a 1D-CNN to the calibrator bake-off.

Same setup as run_calibrator_comparison.py (per-sensor, temporal 60/40 split,
identical features, held-out test vs reference). Computes two extra models:

  XGBoost  - gradient-boosted trees (the named library; same family as the
             HistGradientBoosting already tested, included by request)
  CNN1D    - a 1-D convolutional network that reads the last 24 hours of
             features to predict the current value (deep learning, captures
             short-term temporal patterns the single-hour models cannot)

At the end it MERGES these with the 7 sklearn models from
calibrator_comparison_per_sensor.csv into one final 9-model table.
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

RESULTS = Path("results")
WIN = 24  # hours of context for the CNN


# ---------------- 1D-CNN ----------------
def cnn_predict(Ximp, y, tr_ok, te_ok, feats_n, scaler_mean, scaler_std, device):
    """Train a small 1D-CNN on windows of the standardised features."""
    import torch, torch.nn as nn
    torch.manual_seed(42)

    Xs = (Ximp - scaler_mean) / scaler_std
    n = len(Xs)

    # build [n, feats_n, WIN] windows (channels=features, length=time)
    Xw = np.zeros((n, feats_n, WIN), dtype=np.float32)
    for t in range(n):
        lo = max(0, t - WIN + 1)
        seg = Xs[lo:t + 1].T                      # [feats_n, len]
        Xw[t, :, WIN - seg.shape[1]:] = seg

    ys = (y - np.nanmean(y[tr_ok])) / (np.nanstd(y[tr_ok]) or 1.0)
    y_mean = float(np.nanmean(y[tr_ok])); y_std = float(np.nanstd(y[tr_ok]) or 1.0)

    tr_idx = np.where(tr_ok)[0]
    te_idx = np.where(te_ok)[0]

    Xt = torch.tensor(Xw[tr_idx], device=device)
    yt = torch.tensor(ys[tr_idx], dtype=torch.float32, device=device)

    model = nn.Sequential(
        nn.Conv1d(feats_n, 32, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv1d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    bs = 256
    for epoch in range(60):
        perm = torch.randperm(len(Xt), device=device)
        for b in range(0, len(Xt), bs):
            idx = perm[b:b + bs]
            opt.zero_grad()
            loss = lossf(model(Xt[idx]).squeeze(-1), yt[idx])
            loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        Xte = torch.tensor(Xw[te_idx], device=device)
        pred = model(Xte).squeeze(-1).cpu().numpy()
    return pred * y_std + y_mean


def main():
    import torch
    from xgboost import XGBRegressor
    device = "cuda" if torch.cuda.is_available() else "cpu"

    frames = build_feature_frames(use_cache=True)
    metrics_by_model = {"XGBoost": [], "CNN1D": []}
    per_rows = []
    timing = {"XGBoost": 0.0, "CNN1D": 0.0}

    for i, (sid, df) in enumerate(frames.items()):
        feats = feature_columns(df)
        tr, te = temporal_split(df, 0.6)
        X = df[feats].values.astype(float)
        y = df["ref_no2"].values.astype(float)
        core = ~np.isnan(df["no2_raw"].values) & ~np.isnan(y)
        tr_ok = tr & core; te_ok = te & core
        if tr_ok.sum() < 50 or te_ok.sum() < 20:
            continue

        med = np.nanmedian(X[tr_ok], axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        Ximp = np.where(np.isnan(X), med, X)
        mean = Ximp[tr_ok].mean(0); std = Ximp[tr_ok].std(0); std[std == 0] = 1.0

        # XGBoost
        t0 = time.time()
        xgb = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                           subsample=0.8, colsample_bytree=0.8, random_state=42,
                           n_jobs=-1, verbosity=0)
        xgb.fit(Ximp[tr_ok], y[tr_ok])
        pred = xgb.predict(Ximp[te_ok])
        timing["XGBoost"] += time.time() - t0
        m = all_metrics(y[te_ok], pred)
        metrics_by_model["XGBoost"].append(m)
        per_rows.append({"sensor_id": sid, "model": "XGBoost",
                         "RMSE": round(m["RMSE"], 4), "R2": round(m["R2"], 4),
                         "MAE": round(m["MAE"], 4), "d": round(m["d"], 4)})

        # CNN
        t0 = time.time()
        try:
            cpred = cnn_predict(Ximp, y, tr_ok, te_ok, len(feats), mean, std, device)
            timing["CNN1D"] += time.time() - t0
            mc = all_metrics(y[te_ok], cpred)
            metrics_by_model["CNN1D"].append(mc)
            per_rows.append({"sensor_id": sid, "model": "CNN1D",
                             "RMSE": round(mc["RMSE"], 4), "R2": round(mc["R2"], 4),
                             "MAE": round(mc["MAE"], 4), "d": round(mc["d"], 4)})
        except Exception as e:
            print(f"  CNN failed on {sid}: {e}")

        print(f"  [{i+1}/{len(frames)}] {sid:<22} "
              f"XGB R2={m['R2']:.3f}  CNN R2="
              f"{metrics_by_model['CNN1D'][-1]['R2']:.3f}"
              if metrics_by_model['CNN1D'] else f"  [{i+1}] {sid} XGB R2={m['R2']:.3f}")

    # save extended per-sensor
    ext = pd.DataFrame(per_rows)
    ext.to_csv(RESULTS / "calibrator_extended_per_sensor.csv", index=False)

    # ---- merge with the 7-model results for one final table ----
    def agg(ms, k):
        v = [m[k] for m in ms if not np.isnan(m.get(k, np.nan))]
        return float(np.mean(v)) if v else float("nan")

    final = {}
    # load the 7 sklearn models
    base_csv = RESULTS / "calibrator_comparison_per_sensor.csv"
    if base_csv.exists():
        base = pd.read_csv(base_csv)
        for name, g in base.groupby("model"):
            final[name] = {
                "RMSE": g["RMSE"].mean(), "MAE": g["MAE"].mean(),
                "R2": g["R2"].mean(), "d": g["d"].mean(),
                "n_good": int((g["R2"] > 0.5).sum()), "n": len(g),
            }
    for name in ["XGBoost", "CNN1D"]:
        ms = metrics_by_model[name]
        final[name] = {"RMSE": agg(ms, "RMSE"), "MAE": agg(ms, "MAE"),
                       "R2": agg(ms, "R2"), "d": agg(ms, "d"),
                       "n_good": sum(1 for m in ms if m["R2"] > 0.5), "n": len(ms)}

    order = sorted(final, key=lambda k: final[k]["RMSE"])
    L = ["=" * 66,
         "FINAL CALIBRATION MODEL BAKE-OFF (9 models)",
         "Per-sensor, temporal 60/40 split, held-out test vs reference",
         "=" * 66,
         f"\n{'Model':<14}{'RMSE':>8}{'MAE':>8}{'R2':>8}{'d':>8}{'R2>0.5':>9}",
         "-" * 55]
    for name in order:
        f = final[name]
        L.append(f"{name:<14}{f['RMSE']:>8.3f}{f['MAE']:>8.3f}{f['R2']:>8.3f}"
                 f"{f['d']:>8.3f}{f['n_good']:>6}/{f['n']}")
    L.append(f"\nRanked best->worst by mean RMSE. "
             f"Winner: {order[0]} ({final[order[0]]['RMSE']:.3f})")
    rf = final.get("RandomForest", {}).get("RMSE", float("nan"))
    if not np.isnan(rf):
        L.append(f"\nRandomForest = {rf:.3f}. Deltas vs RandomForest:")
        for name in order:
            if name == "RandomForest":
                continue
            d = (rf - final[name]["RMSE"]) / rf * 100
            L.append(f"  {name:<12}: {d:+.1f}% ({'better' if d>0 else 'worse'})")

    report = "\n".join(L)
    (RESULTS / "calibrator_final.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    print("\nSaved results/calibrator_final.txt")


if __name__ == "__main__":
    main()
