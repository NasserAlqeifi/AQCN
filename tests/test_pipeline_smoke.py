"""
test_pipeline_smoke.py - a tiny synthetic end-to-end run through the assembled
pipeline (calibration + KS drift + fault detection) on a 3-sensor "city" with a
known, learnable raw->reference relation.

It proves the pipeline wires together and that, on a relation the model *can*
learn, calibration beats the raw signal - and that the new "clean" (faulty-
excluded) metrics are produced.
"""
import numpy as np
import pandas as pd

from aqcn.ks_monitor import KsDriftMonitor
from aqcn.pipeline import AQCNPipeline


def _make_city(n_sensors=3, n=1600, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    # one shared "true" pollution signal the whole co-located city sees
    true = 25 + 12 * np.sin(2 * np.pi * np.arange(n) / 24) + rng.normal(0, 3, n)
    frames = {}
    for i in range(n_sensors):
        scale, offset = 1.4 + 0.2 * i, 6 * i           # per-unit gain + offset
        raw = scale * true + offset + rng.normal(0, 4, n)
        frames[f"TST__sensor{i}_S1"] = pd.DataFrame(
            {
                "no2_raw": raw,
                "ref_no2": true,
                "temp": rng.normal(10, 5, n),
                "rh": rng.normal(70, 10, n),
                "pressure": rng.normal(1010, 5, n),
                "hour_sin": np.sin(2 * np.pi * idx.hour / 24),
                "hour_cos": np.cos(2 * np.pi * idx.hour / 24),
                "dow_sin": np.sin(2 * np.pi * idx.dayofweek / 7),
                "dow_cos": np.cos(2 * np.pi * idx.dayofweek / 7),
                "o3": np.nan,
                "no": np.nan,
                "pm25": np.nan,
            },
            index=idx,
        )
    return frames


def test_pipeline_runs_and_calibration_beats_raw():
    pipe = AQCNPipeline(_make_city(), use_gap_fill=False,
                        drift_monitor_factory=KsDriftMonitor)
    results = pipe.run()
    ok = [r for r in results.values() if not r.get("skipped")]
    assert len(ok) == 3
    for r in ok:
        assert r["adaptive"]["RMSE"] < r["raw"]["RMSE"]   # calibration helps
        assert r["adaptive"]["R2"] > 0.5                  # learns the relation
        assert "adaptive_clean" in r                      # clean metrics exist
        assert np.isfinite(r["adaptive_clean"]["RMSE"])


def test_clean_metrics_are_consistent():
    pipe = AQCNPipeline(_make_city(seed=1), use_gap_fill=False,
                        drift_monitor_factory=KsDriftMonitor)
    results = pipe.run()
    for r in results.values():
        if r.get("skipped"):
            continue
        # clean is computed on no more hours than the full test window
        assert r["adaptive_clean"]["n"] <= r["adaptive"]["n"]
        # with no FAULTY hours flagged, clean must equal the all-hours metrics
        if r.get("faulty_h", 0) == 0:
            assert abs(r["adaptive_clean"]["RMSE"] - r["adaptive"]["RMSE"]) < 1e-9
