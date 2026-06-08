"""
calibrator.py - Random Forest calibration core.

Learns f(sensor signal, co-pollutants, meteorology, time) -> reference NO2.
This single supervised step corrects environmental cross-sensitivity,
cross-gas interference, and unit-to-unit scale/offset error.

Confidence per prediction = 1 / (1 + std across the RF's trees), normalised.
A wide spread between trees => low confidence.

Strict temporal split: train on an early fraction, test on the later remainder.
No future information ever enters training (no leakage).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from aqcn.features import feature_columns

RANDOM_STATE = 42
N_ESTIMATORS = 200
MAX_DEPTH = None
MIN_SAMPLES_LEAF = 3


class SensorCalibrator:
    """Per-sensor Random Forest calibrator."""

    def __init__(self, sensor_id: str):
        self.sensor_id = sensor_id
        self.model: RandomForestRegressor | None = None
        self.features: list[str] = []
        self._conf_scale: float = 1.0   # for normalising tree-spread to [0,1]
        self._impute: dict[str, float] = {}  # train-fitted median per feature

    # ── training ──────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame, train_mask: np.ndarray) -> int:
        """
        Fit on training rows that have the core signal (no2_raw) and target.
        Missing covariates are median-imputed (medians fitted on TRAIN only,
        so no leakage). Returns number of training rows used.
        """
        self.features = feature_columns(df)

        # Fit imputation medians on training rows only
        train_df = df[train_mask]
        self._impute = {
            f: float(train_df[f].median()) if train_df[f].notna().any() else 0.0
            for f in self.features
        }

        X = self._apply_impute(df)
        y = df["ref_no2"]

        # Require only the core signal + target to be present for a training row
        core_ok = df["no2_raw"].notna().values & y.notna().values
        usable = train_mask & core_ok
        n = int(usable.sum())
        if n < 50:
            self.model = None
            return n

        self.model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        self.model.fit(X[usable], y[usable])

        spreads = self._tree_spread(X[usable])
        self._conf_scale = float(np.nanmedian(spreads)) if len(spreads) else 1.0
        if self._conf_scale <= 0:
            self._conf_scale = 1.0
        return n

    def _apply_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return feature matrix with missing covariates filled by train medians."""
        X = df[self.features].copy()
        for f in self.features:
            X[f] = X[f].fillna(self._impute.get(f, 0.0))
        return X

    # ── prediction ────────────────────────────────────────────────────
    def predict(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (calibrated_values, confidence) for every row.
        A prediction is produced whenever the core signal no2_raw is present;
        missing covariates are imputed. Rows lacking no2_raw yield NaN.
        """
        n = len(df)
        calibrated = np.full(n, np.nan)
        confidence = np.zeros(n)
        if self.model is None:
            return calibrated, confidence

        X = self._apply_impute(df)
        have_signal = df["no2_raw"].notna().values
        if have_signal.sum() == 0:
            return calibrated, confidence

        Xc = X[have_signal]
        preds = self.model.predict(Xc)
        spreads = self._tree_spread(Xc)
        conf = 1.0 / (1.0 + spreads / self._conf_scale)

        calibrated[have_signal] = preds
        confidence[have_signal] = np.clip(conf, 0.0, 1.0)
        return calibrated, confidence

    def predict_one(self, feat_row: dict) -> tuple[float, float]:
        """Single-row prediction from a feature dict (used in streaming mode).
        Missing covariates are imputed; only no2_raw is mandatory."""
        if self.model is None:
            return float("nan"), 0.0
        if feat_row.get("no2_raw") is None or (
            isinstance(feat_row.get("no2_raw"), float) and np.isnan(feat_row["no2_raw"])
        ):
            return float("nan"), 0.0
        x = np.array([[
            feat_row.get(f) if feat_row.get(f) is not None
            and not (isinstance(feat_row.get(f), float) and np.isnan(feat_row[f]))
            else self._impute.get(f, 0.0)
            for f in self.features
        ]], dtype=float)
        pred = float(self.model.predict(x)[0])
        spread = float(self._tree_spread(x)[0])
        conf = float(np.clip(1.0 / (1.0 + spread / self._conf_scale), 0.0, 1.0))
        return pred, conf

    # ── helpers ───────────────────────────────────────────────────────
    def _tree_spread(self, X) -> np.ndarray:
        """Std of predictions across individual trees (model uncertainty)."""
        if self.model is None:
            return np.zeros(len(X))
        per_tree = np.stack([t.predict(np.asarray(X)) for t in self.model.estimators_])
        return per_tree.std(axis=0)

    def feature_importance(self) -> dict[str, float]:
        if self.model is None:
            return {}
        return dict(zip(self.features, self.model.feature_importances_))


def temporal_split(df: pd.DataFrame, train_frac: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
    """
    Strict temporal split anchored on PAIRED rows (rows that have both the
    sensor signal and the reference target). The first `train_frac` of paired
    rows in time go to train; the remainder to test. This guarantees both
    windows contain evaluable ground truth while never letting future data
    into training (the split is a single point in time).

    Returns (train_mask, test_mask) aligned to df rows. Rows before the cut
    time are train-eligible; rows at/after are test-eligible.
    """
    paired = (df["no2_raw"].notna() & df["ref_no2"].notna()).values
    n = len(df)
    train_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)

    paired_positions = np.where(paired)[0]
    if len(paired_positions) < 20:
        # too little paired data - fall back to row split
        cut = int(n * train_frac)
    else:
        k = int(len(paired_positions) * train_frac)
        cut = paired_positions[min(k, len(paired_positions) - 1)]

    train_mask[:cut] = True
    test_mask[cut:] = True
    return train_mask, test_mask


if __name__ == "__main__":
    # Prove the core calibration works on held-out future data
    import sys
    sys.path.insert(0, ".")
    from aqcn.features import build_feature_frames

    frames = build_feature_frames(use_cache=True)

    def rmse(o, p):
        m = ~(np.isnan(o) | np.isnan(p))
        return float(np.sqrt(np.mean((o[m] - p[m]) ** 2))) if m.sum() else float("nan")

    def r2(o, p):
        m = ~(np.isnan(o) | np.isnan(p))
        if m.sum() < 5:
            return float("nan")
        ss = np.sum((o[m] - p[m]) ** 2)
        st = np.sum((o[m] - np.mean(o[m])) ** 2)
        return float(1 - ss / st) if st > 0 else float("nan")

    print("\n" + "=" * 78)
    print("CORE CALIBRATION PROOF - Random Forest, held-out future test window")
    print("=" * 78)
    print(f"\n{'Sensor':<26}{'RawRMSE':>9}{'CalRMSE':>9}{'RawR2':>8}{'CalR2':>8}{'Impr%':>8}")
    print("-" * 70)

    examples = ["MCH__Prax1_S1", "MCH__AQY872_S1", "MCH__AQM388_S1",
                "LON__AQY874_S1", "YRK__Ari093_S1", "MCH__Poll2_S1"]
    for sid in examples:
        if sid not in frames:
            continue
        df = frames[sid]
        tr, te = temporal_split(df, 0.6)
        cal = SensorCalibrator(sid)
        n_train = cal.fit(df, tr)
        pred, conf = cal.predict(df)

        y = df["ref_no2"].values
        raw = df["no2_raw"].values

        raw_rmse = rmse(y[te], raw[te])
        cal_rmse = rmse(y[te], pred[te])
        raw_r2 = r2(y[te], raw[te])
        cal_r2 = r2(y[te], pred[te])
        impr = (raw_rmse - cal_rmse) / raw_rmse * 100 if raw_rmse else float("nan")
        print(f"{sid:<26}{raw_rmse:>9.3f}{cal_rmse:>9.3f}{raw_r2:>8.3f}"
              f"{cal_r2:>8.3f}{impr:>7.1f}%")

    # Show feature importance for one sensor
    sid = "MCH__Prax1_S1"
    if sid in frames:
        cal = SensorCalibrator(sid)
        tr, te = temporal_split(frames[sid], 0.6)
        cal.fit(frames[sid], tr)
        imp = sorted(cal.feature_importance().items(), key=lambda x: -x[1])
        print(f"\nFeature importance ({sid}):")
        for f, v in imp:
            print(f"  {f:<12} {v:.3f}")
