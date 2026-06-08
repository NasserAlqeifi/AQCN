"""
test_leakage.py - guard the two places future information could leak into the
calibrator: the temporal split and the train-only imputation medians.

These are the tests that make the "no leakage" claim verifiable instead of
asserted. If a refactor ever lets test data into training, one of these fails.
"""
import numpy as np
import pandas as pd

from aqcn.calibrator import SensorCalibrator, temporal_split


def _toy_frame(n=600, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    raw = rng.normal(40, 10, n)
    ref = 0.5 * raw + rng.normal(0, 2, n)
    return pd.DataFrame(
        {
            "no2_raw": raw,
            "ref_no2": ref,
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


def test_train_strictly_precedes_test_in_time():
    df = _toy_frame()
    tr, te = temporal_split(df, 0.6)
    assert tr.sum() > 0 and te.sum() > 0
    assert not (tr & te).any()                      # disjoint
    assert df.index[tr].max() < df.index[te].min()  # the split is a point in time


def test_split_is_a_single_contiguous_cut():
    df = _toy_frame()
    tr, te = temporal_split(df, 0.6)
    cut = tr.sum()
    assert tr[:cut].all() and te[cut:].all()        # prefix train, suffix test
    assert cut + te.sum() == len(df)                # together they cover everything


def test_impute_medians_come_from_train_only():
    df = _toy_frame().copy()
    tr, te = temporal_split(df, 0.6)
    # Poison every TEST-window covariate with an extreme value. If medians were
    # fit on the full frame, they would be dragged toward 1e6.
    df.loc[df.index[te], "temp"] = 1e6
    cal = SensorCalibrator("toy")
    cal.fit(df, tr)
    train_median = float(df["temp"][tr].median())
    assert abs(cal._impute["temp"] - train_median) < 1e-6
    assert cal._impute["temp"] < 1e5                # untouched by the test poison


def test_predict_unaffected_by_future_rows():
    """Predictions on the train window must not change when future rows change."""
    df = _toy_frame()
    tr, te = temporal_split(df, 0.6)
    cal = SensorCalibrator("toy")
    cal.fit(df, tr)
    pred_before, _ = cal.predict(df)

    poisoned = df.copy()
    poisoned.loc[poisoned.index[te], ["no2_raw", "temp", "rh"]] = 9999.0
    pred_after, _ = cal.predict(poisoned)

    # Train-window predictions must not move when only future rows change.
    # (Tolerance is machine-epsilon: sklearn's parallel RF predict sums tree
    # outputs non-deterministically at ~1e-14; anything larger means leakage.)
    np.testing.assert_allclose(pred_before[tr], pred_after[tr], rtol=1e-9, atol=1e-9)
