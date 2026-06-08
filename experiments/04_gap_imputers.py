"""
Experiment 04 - Gap-filling imputer comparison (SAITS vs BRITS vs CSDI vs Linear).

Runs the fair imputation benchmark and writes results/imputer_comparison.{txt,csv}.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
from aqcn.features import build_feature_frames
from aqcn.imputer_benchmark import run_benchmark

RESULTS = Path("results")


def main():
    RESULTS.mkdir(exist_ok=True)
    frames = build_feature_frames(use_cache=True)
    results = run_benchmark(frames, epochs=40)

    df = pd.DataFrame(results).round(4)
    df.to_csv(RESULTS / "imputer_comparison.csv", index=False)

    methods = ["SAITS", "BRITS", "CSDI", "Linear"]
    L = ["=" * 60,
         "GAP-FILLING COMPARISON  (proper imputers, Chronos dropped)",
         "RMSE (ppb) on artificially-masked NO2 gaps vs truth",
         "=" * 60,
         f"\n{'Gap':>5} " + "".join(f"{m:>10}" for m in methods)]
    for r in results:
        L.append(f"{r['gap_len']:>4}h " +
                 "".join(f"{r[f'{m}_rmse']:>10.3f}" for m in methods))
    L.append("\nBest method per gap length:")
    for r in results:
        best = min(methods, key=lambda m: r[f"{m}_rmse"]
                   if not np.isnan(r[f"{m}_rmse"]) else 1e9)
        L.append(f"  {r['gap_len']:>2}h -> {best} ({r[f'{best}_rmse']:.3f})")
    report = "\n".join(L)
    (RESULTS / "imputer_comparison.txt").write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
