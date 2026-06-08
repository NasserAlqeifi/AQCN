"""
imputer_benchmark.py - Fair gap-filling comparison among PROPER imputers.

Chronos is dropped: it is a forecaster (past-only), not an imputation model, so
comparing it here was apples-to-oranges. This benchmark compares methods that
are actually designed to fill gaps using BOTH sides + channel correlations:

  SAITS   - self-attention imputation (top performer on air-quality data,
            Hua et al. 2024 six-dataset benchmark)
  BRITS   - bidirectional recurrent imputation (co-top performer on AQ data)
  CSDI    - conditional diffusion imputation (general SOTA)
  Linear  - interpolation between gap endpoints (simple baseline)

Fair protocol (identical for every method):
  * train ONE model on windows POOLED across all sensors (more data),
  * test on held-out fully-observed windows from each sensor,
  * artificially blank a contiguous NO2 block of length 6 / 12 / 24 h,
  * compare each method's fill to the known truth (RMSE / MAE, ppb).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aqcn.diffusion_imputer import (
    WINDOW, FEATURES, GAP_LENGTHS, build_pooled, _standardise_fit)


def _rmse(a, b):
    a, b = a.ravel(), b.ravel()
    m = ~(np.isnan(a) | np.isnan(b))
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if m.sum() else float("nan")


def _mae(a, b):
    a, b = a.ravel(), b.ravel()
    m = ~(np.isnan(a) | np.isnan(b))
    return float(np.mean(np.abs(a[m] - b[m]))) if m.sum() else float("nan")


def _build_models(n_feat, device, epochs):
    from pypots.imputation import SAITS, BRITS, CSDI
    return {
        "SAITS": SAITS(n_steps=WINDOW, n_features=n_feat, n_layers=2, d_model=128,
                       n_heads=4, d_k=32, d_v=32, d_ffn=128, dropout=0.1,
                       batch_size=64, epochs=epochs, patience=8,
                       device=device, verbose=False),
        "BRITS": BRITS(n_steps=WINDOW, n_features=n_feat, rnn_hidden_size=128,
                       batch_size=64, epochs=epochs, patience=8,
                       device=device, verbose=False),
        "CSDI": CSDI(n_steps=WINDOW, n_features=n_feat, n_layers=4, n_heads=4,
                     n_channels=64, d_time_embedding=64, d_feature_embedding=32,
                     d_diffusion_embedding=64, n_diffusion_steps=50,
                     batch_size=64, epochs=epochs, patience=8,
                     device=device, verbose=False),
    }


def run_benchmark(frames: dict, epochs: int = 40) -> list[dict]:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    no2_idx = FEATURES.index("no2_raw")

    pooled, test_by_sensor = build_pooled(frames)
    mean, std = _standardise_fit(pooled)
    pooled_s = (pooled - mean) / std
    m_no2, s_no2 = mean[no2_idx], std[no2_idx]

    print(f"[imputer] Pooled training windows: {len(pooled_s)} | "
          f"test sensors: {len(test_by_sensor)} | device: {device}")

    models = _build_models(len(FEATURES), device, epochs)
    for name, model in models.items():
        print(f"[imputer] Training {name} ({epochs} epochs)...")
        model.fit({"X": pooled_s})
    print("[imputer] All models trained. Benchmarking gap lengths...")

    results = []
    for gap in GAP_LENGTHS:
        g0 = WINDOW // 2 - gap // 2
        g1 = g0 + gap
        truth_all = []
        pred_all = {name: [] for name in models}
        pred_all["Linear"] = []

        for sid, test_w in test_by_sensor.items():
            test_s = (test_w - mean) / std
            masked = test_s.copy()
            truth = test_w[:, g0:g1, no2_idx].copy()
            masked[:, g0:g1, no2_idx] = np.nan
            truth_all.append(truth)

            # deep imputers
            for name, model in models.items():
                imp = np.asarray(model.predict({"X": masked})["imputation"])
                if imp.ndim == 4:                      # CSDI: [n,samples,steps,feat]
                    imp = np.median(imp, axis=1)
                pred = imp[:, g0:g1, no2_idx] * s_no2 + m_no2
                pred_all[name].append(pred)

            # linear interpolation baseline (uses both endpoints)
            lin = np.empty_like(truth)
            for i in range(len(test_w)):
                col = test_w[i, :, no2_idx].copy()
                col[g0:g1] = np.nan
                lin[i] = pd.Series(col).interpolate(
                    method="linear", limit_direction="both").values[g0:g1]
            pred_all["Linear"].append(lin)

        truth = np.concatenate(truth_all)
        row = {"gap_len": gap, "n_points": int(np.sum(~np.isnan(truth)))}
        for name in list(models) + ["Linear"]:
            pred = np.concatenate(pred_all[name])
            row[f"{name}_rmse"] = _rmse(truth, pred)
            row[f"{name}_mae"] = _mae(truth, pred)
        results.append(row)
        msg = "  ".join(f"{n}={row[f'{n}_rmse']:.3f}"
                        for n in list(models) + ["Linear"])
        print(f"  gap={gap:>2}h  {msg}")

    return results


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, ".")
    from aqcn.features import build_feature_frames

    frames = build_feature_frames(use_cache=True)
    results = run_benchmark(frames, epochs=40)

    # Save report
    RES = Path("results"); RES.mkdir(exist_ok=True)
    df = pd.DataFrame(results).round(4)
    df.to_csv(RES / "imputer_comparison.csv", index=False)

    methods = ["SAITS", "BRITS", "CSDI", "Linear"]
    L = ["=" * 60,
         "GAP-FILLING COMPARISON  (proper imputers, Chronos dropped)",
         "RMSE (ppb) on artificially-masked NO2 gaps vs truth",
         "=" * 60,
         f"\n{'Gap':>5} " + "".join(f"{m:>10}" for m in methods)]
    for r in results:
        L.append(f"{r['gap_len']:>4}h " +
                 "".join(f"{r[f'{m}_rmse']:>10.3f}" for m in methods))
    # winner per gap
    L.append("\nBest method per gap length:")
    for r in results:
        best = min(methods, key=lambda m: r[f"{m}_rmse"]
                   if not np.isnan(r[f"{m}_rmse"]) else 1e9)
        L.append(f"  {r['gap_len']:>2}h -> {best} ({r[f'{best}_rmse']:.3f})")
    report = "\n".join(L)
    (RES / "imputer_comparison.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    print("\nSaved results/imputer_comparison.csv + .txt")
