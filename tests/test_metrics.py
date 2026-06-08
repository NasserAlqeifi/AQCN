"""
test_metrics.py - sanity checks on the scientific metric suite. Cheap insurance
that RMSE/MAE/R²/Willmott keep their textbook meaning and that NaN pairs are
dropped pairwise (the property the whole evaluation relies on).
"""
import numpy as np

from aqcn.evaluator import all_metrics, mae, r2, rmse, willmott


def test_perfect_prediction():
    o = np.array([1.0, 2, 3, 4, 5, 6])
    p = o.copy()
    assert rmse(o, p) == 0.0
    assert mae(o, p) == 0.0
    assert abs(r2(o, p) - 1.0) < 1e-9
    assert abs(willmott(o, p) - 1.0) < 1e-9


def test_constant_error_is_recovered():
    o = np.array([0.0, 0, 0, 0, 0])
    p = np.array([3.0, 3, 3, 3, 3])
    assert abs(rmse(o, p) - 3.0) < 1e-9
    assert abs(mae(o, p) - 3.0) < 1e-9


def test_mean_predictor_gives_r2_zero():
    rng = np.random.default_rng(0)
    o = rng.normal(10, 5, 200)
    p = np.full_like(o, o.mean())
    assert abs(r2(o, p)) < 1e-9


def test_worse_than_mean_gives_negative_r2():
    o = np.array([1.0, 2, 3, 4, 5])
    p = np.array([5.0, 4, 3, 2, 1])      # anti-correlated
    assert r2(o, p) < 0


def test_nan_pairs_dropped_pairwise():
    o = np.array([1.0, 2, np.nan, 4, 5, 6])
    p = np.array([1.0, 2, 3, 4, np.nan, 6])
    # the two surviving complete pairs (and 1,2,4,6) are exact -> zero error
    assert rmse(o, p) == 0.0
    assert all_metrics(o, p)["n"] == 4
