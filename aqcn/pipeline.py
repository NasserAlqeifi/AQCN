"""
pipeline.py - The unified AQCN calibration pipeline.

Per sensor, chronologically and without leakage:
  1. (optional, OFF by default) GAP FILL the raw signal so more hours are
     calibratable. The benchmark winner is SAITS (experiments/04); the wired-in
     helper (gap_filler.py) currently uses a lightweight Chronos stand-in.
  2. TRAIN a Random Forest calibrator on the early (train) window.
  3. STATIC calibrate the held-out test window.
  4. ADAPTIVE calibrate the test window with drift-triggered retraining. The
     drift monitor is pluggable (default CUSUM; the final system uses the KS
     monitor). Each candidate refit is judged on a HELD-OUT TAIL of its refit
     window - never on its own training rows - so the accept/reject decision is
     unbiased.
  5. FAULT-detect using same-city peer consensus. FAULTY-flagged hours are
     excluded from the reported "clean" metrics.
Produces per-sensor metrics (all-hours AND clean) and a per-hour output table.

Evaluation always compares against the ratified reference on the test window.
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd

from aqcn.features import build_feature_frames
from aqcn.calibrator import SensorCalibrator, temporal_split
from aqcn.drift_monitor import CusumDriftMonitor
from aqcn.fault_ensemble import detect_faults
from aqcn.evaluator import all_metrics

TRAIN_FRAC = 0.6
RETRAIN_WINDOW_DAYS = 60      # rolling window used when drift triggers a refit
MAX_RETRAINS = 10             # cap accepted refits per sensor
MAX_RETRAIN_ATTEMPTS = 30     # cap total refit attempts (performance guard)
RETRAIN_COOLDOWN_H = 168      # min hours between retrain attempts (1 week)
RETRAIN_VAL_MIN_H = 168       # min held-out tail used to judge a candidate refit


class AQCNPipeline:
    def __init__(self, frames: dict[str, pd.DataFrame], use_gap_fill: bool = False,
                 drift_monitor_factory=None):
        self.frames = frames
        self.use_gap_fill = use_gap_fill
        # factory() -> a fresh drift monitor with .calibrate()/.update().
        # Defaults to CUSUM; pass AdwinDriftMonitor / BocpdDriftMonitor to swap.
        if drift_monitor_factory is None:
            drift_monitor_factory = CusumDriftMonitor
        self.drift_monitor_factory = drift_monitor_factory
        self.sensor_ids = list(frames.keys())
        self.calibrators: dict[str, SensorCalibrator] = {}
        self.results: dict[str, dict] = {}
        self.per_hour: dict[str, pd.DataFrame] = {}

    # ── stage 1: optional gap fill ────────────────────────────────────
    def _maybe_gap_fill(self, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        if not self.use_gap_fill:
            return df, 0
        from aqcn.gap_filler import fill_signal_gaps
        filled, mask, n = fill_signal_gaps(df["no2_raw"])
        df = df.copy()
        df["no2_raw"] = filled
        df["_gap_filled"] = mask
        return df, n

    # ── stage 2-3: train + static calibrate ──────────────────────────
    def _train_static(self, df: pd.DataFrame, sid: str):
        tr, te = temporal_split(df, TRAIN_FRAC)
        cal = SensorCalibrator(sid)
        n_train = cal.fit(df, tr)
        self.calibrators[sid] = cal
        pred, conf = cal.predict(df)
        return tr, te, cal, n_train, pred, conf

    # ── stage 4: adaptive (drift-triggered retrain) ──────────────────
    def _adaptive(self, df: pd.DataFrame, sid: str, tr: np.ndarray,
                  te: np.ndarray, base_cal: SensorCalibrator,
                  static_pred: np.ndarray, conf: np.ndarray) -> tuple[np.ndarray, int, list]:
        """
        Walk the test window. A pluggable drift monitor watches the appropriate
        signal (residual, raw-signal distribution, or model uncertainty). When it
        fires, refit the RF on the last RETRAIN_WINDOW_DAYS of paired data ending
        now (past only -> causal) and BATCH-repredict the remaining test rows.
        At most MAX_RETRAINS batch predictions. Returns adaptive predictions,
        n_retrains, events.
        """
        n = len(df)
        adaptive = static_pred.copy()
        y = df["ref_no2"].values
        signal = df["no2_raw"].values          # raw signal (for KS detector)
        uncertainty = 1.0 - np.asarray(conf)   # model uncertainty (for unc detector)
        times = df.index

        base_resid = static_pred[tr] - y[tr]
        monitor = self.drift_monitor_factory()
        monitor.calibrate({
            "residual": base_resid,
            "signal": signal[tr],
            "uncertainty": uncertainty[tr],
        })

        cur_cal = base_cal
        n_retrains = 0
        events = []
        test_positions = np.where(te)[0]

        def _win_rmse(pred, lo, hi):
            seg_p = pred[lo:hi]
            seg_y = y[lo:hi]
            m = ~(np.isnan(seg_p) | np.isnan(seg_y))
            return float(np.sqrt(np.mean((seg_p[m] - seg_y[m]) ** 2))) if m.sum() >= 10 else np.inf

        last_attempt = -10**9          # cooldown anchor
        attempts = 0
        for t in test_positions:
            r = adaptive[t] - y[t] if not np.isnan(y[t]) else np.nan
            fired = monitor.update({
                "residual": r,
                "signal": signal[t],
                "uncertainty": uncertainty[t],
            })
            # Cooldown: at most one retrain attempt per RETRAIN_COOLDOWN_H, and a
            # hard cap on total attempts (accepted + rejected) for performance.
            if (fired and n_retrains < MAX_RETRAINS
                    and attempts < MAX_RETRAIN_ATTEMPTS
                    and (t - last_attempt) >= RETRAIN_COOLDOWN_H):
                last_attempt = t
                attempts += 1
                win_start = max(0, t - RETRAIN_WINDOW_DAYS * 24)
                win_len = t - win_start
                # Hold out the most recent slice of the refit window for an
                # UNBIASED accept/reject decision: the candidate is trained on the
                # head and judged on a tail it never saw (the current model never
                # saw it either, having been trained earlier). This guards against
                # accepting refits that merely memorise their own training rows.
                val_len = max(RETRAIN_VAL_MIN_H, win_len // 5)
                fit_end = t - val_len
                if fit_end - win_start < win_len // 2:        # keep >=50% to fit
                    fit_end = win_start + (win_len * 3) // 4
                fit_mask = np.zeros(n, dtype=bool)
                fit_mask[win_start:fit_end] = True
                cand = SensorCalibrator(sid)
                if cand.fit(df, fit_mask) >= 50:
                    pad = np.full(fit_end, np.nan)
                    cur_val, _ = cur_cal.predict(df.iloc[fit_end:t])
                    new_val, _ = cand.predict(df.iloc[fit_end:t])
                    cur_rmse = _win_rmse(np.concatenate([pad, cur_val]), fit_end, t)
                    new_rmse = _win_rmse(np.concatenate([pad, new_val]), fit_end, t)
                    # Require a real, EVALUABLE improvement on the held-out tail.
                    if (np.isfinite(cur_rmse) and np.isfinite(new_rmse)
                            and new_rmse <= cur_rmse):
                        # Accepted: refit on the FULL window (head + tail) for the
                        # strongest deployable model, then re-predict forward.
                        refit_mask = np.zeros(n, dtype=bool)
                        refit_mask[win_start:t] = True
                        deploy = SensorCalibrator(sid)
                        if deploy.fit(df, refit_mask) >= 50:
                            new_pred, _ = deploy.predict(df)
                            adaptive[t:] = new_pred[t:]
                            cur_cal = deploy
                            n_retrains += 1
                            events.append({"pos": int(t), "time": str(times[t])[:16]})
                            monitor.calibrate({
                                "residual": adaptive[win_start:t] - y[win_start:t],
                                "signal": signal[win_start:t],
                                "uncertainty": uncertainty[win_start:t],
                            })

        return adaptive, n_retrains, events

    # ── main run ──────────────────────────────────────────────────────
    def run(self):
        t0 = time.time()
        print(f"[pipeline] Running on {len(self.sensor_ids)} sensors "
              f"(gap_fill={self.use_gap_fill})...")

        # First pass: train + static + adaptive per sensor
        static_cal_by_sensor: dict[str, np.ndarray] = {}
        masks: dict[str, tuple] = {}

        for i, sid in enumerate(self.sensor_ids):
            df = self.frames[sid]
            df, n_gap = self._maybe_gap_fill(df)

            tr, te, cal, n_train, pred, conf = self._train_static(df, sid)
            if cal.model is None:
                self.results[sid] = {"skipped": True, "reason": f"train rows {n_train}"}
                continue

            adaptive, n_retrains, drift_events = self._adaptive(
                df, sid, tr, te, cal, pred, conf)

            y = df["ref_no2"].values
            raw = df["no2_raw"].values

            m_raw = all_metrics(y[te], raw[te])
            m_static = all_metrics(y[te], pred[te])
            m_adaptive = all_metrics(y[te], adaptive[te])

            self.results[sid] = {
                "skipped": False,
                "city": sid.split("__")[0],
                "n_train": n_train,
                "n_gap_filled": n_gap,
                "n_retrains": n_retrains,
                "drift_events": drift_events,
                "raw": m_raw,
                "static": m_static,
                "adaptive": m_adaptive,
                # "clean" = test metrics with FAULTY-flagged hours removed.
                # Defaults to all-hours; overwritten once fault detection runs.
                "adaptive_clean": m_adaptive,
                "importance": cal.feature_importance(),
            }
            static_cal_by_sensor[sid] = pred
            masks[sid] = (tr, te)

            self.per_hour[sid] = pd.DataFrame({
                "raw": raw, "calibrated": adaptive,
                "reference": y, "confidence": conf,
            }, index=df.index)

            print(f"  [{i+1}/{len(self.sensor_ids)}] {sid:<24} "
                  f"raw RMSE {m_raw['RMSE']:.2f} -> cal {m_adaptive['RMSE']:.2f}  "
                  f"R2 {m_raw['R2']:.2f}->{m_adaptive['R2']:.2f}  "
                  f"retrains={n_retrains}")

        # Second pass: fault detection using same-city consensus on test window
        self._run_fault_detection(static_cal_by_sensor, masks)

        print(f"[pipeline] Done in {time.time()-t0:.1f}s")
        return self.results

    def _run_fault_detection(self, static_pred: dict, masks: dict):
        # Build same-city calibrated matrices, consensus = peer median
        by_city: dict[str, list[str]] = {}
        for sid in static_pred:
            by_city.setdefault(sid.split("__")[0], []).append(sid)

        for city, sids in by_city.items():
            if len(sids) < 2:
                continue
            for sid in sids:
                own_idx = self.per_hour[sid].index
                peers = [s for s in sids if s != sid]
                # peer calibrated series reindexed onto THIS sensor's timeline
                peer_mat = pd.DataFrame(
                    {s: self.per_hour[s]["calibrated"].reindex(own_idx) for s in peers},
                    index=own_idx,
                )
                consensus = peer_mat.median(axis=1).values
                calibrated = self.per_hour[sid]["calibrated"].values
                reference = self.per_hour[sid]["reference"].values
                status, anomaly = detect_faults(calibrated, reference, consensus)
                self.per_hour[sid]["status"] = status
                self.per_hour[sid]["anomaly"] = anomaly
                # record fault summary on test window
                tr, te = masks[sid]
                faulty_h = int(np.sum((status == "FAULTY") & te))
                suspect_h = int(np.sum((status == "SUSPECT") & te))
                self.results[sid]["faulty_h"] = faulty_h
                self.results[sid]["suspect_h"] = suspect_h
                # "Clean" output: recompute test metrics with FAULTY-flagged
                # hours excluded, matching the system's promise that faulty
                # readings are dropped from the delivered stream.
                clean = te & (status != "FAULTY")
                self.results[sid]["adaptive_clean"] = all_metrics(
                    reference[clean], calibrated[clean])


if __name__ == "__main__":
    frames = build_feature_frames(use_cache=True)
    pipe = AQCNPipeline(frames, use_gap_fill=False)
    pipe.run()
