"""
diffusion_imputer.py - Fair gap-fill benchmark: diffusion (CSDI) vs Chronos vs linear.

Design for a fair test:
  * Train ONE CSDI diffusion model on windows POOLED across all sensors
    (diffusion models are data-hungry; per-sensor data is too little).
  * Test at several gap lengths (6 / 12 / 24 h). Short gaps favour linear
    interpolation; longer gaps are where learned models should win.
  * Compare on identical artificially-masked points against known truth.

Methods:
  CSDI    - conditional diffusion, uses both sides of the gap + channel
            correlations (NO2, O3, NO, temp, rh, pressure).
  Chronos - foundation forecasting model, left-context only.
  Linear  - interpolation between the two gap endpoints (both sides).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW = 48
FEATURES = ["no2_raw", "o3", "no", "temp", "rh", "pressure"]
GAP_LENGTHS = [6, 12, 24]
RANDOM_STATE = 42


def _windows_for(df: pd.DataFrame):
    cols = [f for f in FEATURES if f in df.columns]
    # pad missing optional channels so every sensor has the same feature axis
    arr = np.full((len(df), len(FEATURES)), np.nan, dtype="float32")
    for j, f in enumerate(FEATURES):
        if f in df.columns:
            arr[:, j] = df[f].values.astype("float32")
    n = (len(arr) // WINDOW) * WINDOW
    return arr[:n].reshape(-1, WINDOW, len(FEATURES))


def build_pooled(frames: dict) -> tuple[np.ndarray, dict]:
    """Return pooled training windows and per-sensor fully-observed test windows."""
    no2_idx = FEATURES.index("no2_raw")
    train_parts = []
    test_by_sensor = {}
    rng = np.random.default_rng(RANDOM_STATE)

    for sid, df in frames.items():
        w = _windows_for(df)
        no2_full = ~np.isnan(w[:, :, no2_idx]).any(axis=1)
        full = w[no2_full]
        if len(full) < 20:
            # still contribute to training pool
            train_parts.append(w)
            continue
        perm = rng.permutation(len(full))
        n_test = max(8, int(len(full) * 0.2))
        test_by_sensor[sid] = full[perm[:n_test]]
        # everything else (incl. partially-missing windows) goes to training
        train_mask = np.ones(len(w), dtype=bool)
        train_parts.append(w)  # pool all windows for training the generative model

    pooled = np.concatenate(train_parts, axis=0)
    return pooled, test_by_sensor


def _standardise_fit(arr):
    flat = arr.reshape(-1, arr.shape[-1])
    mean = np.nanmean(flat, axis=0)
    std = np.nanstd(flat, axis=0)
    std[std == 0] = 1.0
    return mean, std


def run_benchmark(frames: dict, train_epochs: int = 40) -> list[dict]:
    import torch
    from pypots.imputation import CSDI

    no2_idx = FEATURES.index("no2_raw")
    pooled, test_by_sensor = build_pooled(frames)
    mean, std = _standardise_fit(pooled)

    pooled_s = (pooled - mean) / std
    print(f"[diffusion] Training CSDI on {len(pooled_s)} pooled windows "
          f"({train_epochs} epochs)...")

    model = CSDI(n_steps=WINDOW, n_features=len(FEATURES), n_layers=4, n_heads=4,
                 n_channels=64, d_time_embedding=64, d_feature_embedding=32,
                 d_diffusion_embedding=64, n_diffusion_steps=50,
                 batch_size=64, epochs=train_epochs, patience=8,
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 verbose=False)
    model.fit({"X": pooled_s})
    print("[diffusion] CSDI trained. Loading Chronos for comparison...")

    from chronos import ChronosPipeline
    chronos = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-tiny",
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float32)

    def rmse(a, b):
        a, b = a.ravel(), b.ravel()
        m = ~(np.isnan(a) | np.isnan(b))
        return float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if m.sum() else float("nan")

    results = []
    m_no2, s_no2 = mean[no2_idx], std[no2_idx]

    for gap in GAP_LENGTHS:
        g0 = WINDOW // 2 - gap // 2
        g1 = g0 + gap
        # accumulate predictions across all sensors' test windows
        truth_all, csdi_all, lin_all, chr_all = [], [], [], []

        for sid, test_w in test_by_sensor.items():
            test_s = (test_w - mean) / std
            masked = test_s.copy()
            truth = test_w[:, g0:g1, no2_idx].copy()
            masked[:, g0:g1, no2_idx] = np.nan

            imp = np.asarray(model.predict({"X": masked})["imputation"])
            if imp.ndim == 4:
                imp = np.median(imp, axis=1)
            csdi = imp[:, g0:g1, no2_idx] * s_no2 + m_no2

            lin = np.empty_like(truth)
            for i in range(len(test_w)):
                col = test_w[i, :, no2_idx].copy()
                col[g0:g1] = np.nan
                lin[i] = pd.Series(col).interpolate(
                    method="linear", limit_direction="both").values[g0:g1]

            chr_pred = np.full_like(truth, np.nan)
            for i in range(len(test_w)):
                ctx = test_w[i, :g0, no2_idx].astype("float32")
                ctx = ctx[~np.isnan(ctx)]
                if len(ctx) < 3:
                    continue
                t = torch.tensor(ctx).unsqueeze(0)
                with torch.no_grad():
                    s = chronos.predict(inputs=t, prediction_length=gap, num_samples=20)
                chr_pred[i] = np.median(s[0].cpu().numpy(), axis=0)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            truth_all.append(truth); csdi_all.append(csdi)
            lin_all.append(lin); chr_all.append(chr_pred)

        truth = np.concatenate(truth_all); csdi = np.concatenate(csdi_all)
        lin = np.concatenate(lin_all); chrs = np.concatenate(chr_all)
        results.append({
            "gap_len": gap,
            "n_points": int(np.sum(~np.isnan(truth))),
            "csdi_rmse": rmse(truth, csdi),
            "chronos_rmse": rmse(truth, chrs),
            "linear_rmse": rmse(truth, lin),
        })
        print(f"  gap={gap:>2}h  CSDI={results[-1]['csdi_rmse']:.3f}  "
              f"Chronos={results[-1]['chronos_rmse']:.3f}  "
              f"Linear={results[-1]['linear_rmse']:.3f}")

    return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from aqcn.features import build_feature_frames
    frames = build_feature_frames(use_cache=True)
    run_benchmark(frames, train_epochs=40)
