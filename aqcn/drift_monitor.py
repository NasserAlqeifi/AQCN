"""
drift_monitor.py - CUSUM drift detection on calibration residuals.

After calibration, the residual (calibrated - reference) should hover near zero.
If the sensor's response surface shifts (ageing, fouling), the residual develops
a persistent bias. A two-sided CUSUM detects that accumulation and raises a
"retrain" signal. The pipeline responds by refitting the RF on the most recent
window of past data (causal, no future leakage).

CUSUM parameters are derived from the residual statistics of the early
(training) period, so the detector is self-tuning per sensor.
"""

from __future__ import annotations

import numpy as np


class CusumDriftMonitor:
    def __init__(self, slack_k: float = 0.5, threshold_h: float = 5.0):
        self.slack_k = slack_k       # allowance, in units of residual sigma
        self.threshold_h = threshold_h  # decision threshold, in sigma
        self.mu0 = 0.0
        self.sigma = 1.0
        self.k = 0.0
        self.h = 1.0
        self.s_pos = 0.0
        self.s_neg = 0.0
        self._calibrated = False

    def calibrate(self, baseline):
        """Set CUSUM parameters from clean baseline residuals.
        Accepts an array of residuals or a dict with key 'residual'."""
        r = baseline.get("residual") if isinstance(baseline, dict) else baseline
        r = np.asarray(r, dtype=float)
        r = r[~np.isnan(r)]
        if len(r) < 24:
            self._calibrated = False
            return
        self.mu0 = float(np.mean(r))
        self.sigma = float(np.std(r)) if np.std(r) > 0 else 1.0
        self.k = self.slack_k * self.sigma
        self.h = self.threshold_h * self.sigma
        self.s_pos = 0.0
        self.s_neg = 0.0
        self._calibrated = True

    def update(self, signal) -> bool:
        """
        Feed one residual (scalar) or a signals dict with key "residual".
        Returns True if drift is signalled at this step. Resets the accumulators
        after a signal (re-arm for the next event).
        """
        residual = signal.get("residual") if isinstance(signal, dict) else signal
        if not self._calibrated or residual is None or np.isnan(residual):
            return False
        d = residual - self.mu0
        self.s_pos = max(0.0, self.s_pos + d - self.k)
        self.s_neg = max(0.0, self.s_neg - d - self.k)
        if self.s_pos > self.h or self.s_neg > self.h:
            self.s_pos = 0.0
            self.s_neg = 0.0
            return True
        return False


if __name__ == "__main__":
    # Synthetic check: clean then drifting residuals
    rng = np.random.default_rng(42)
    clean = rng.normal(0, 1, 500)
    drift = np.linspace(0, 6, 300) + rng.normal(0, 1, 300)
    series = np.concatenate([clean, drift])

    m = CusumDriftMonitor()
    m.calibrate(clean)
    first = None
    for t, r in enumerate(series):
        if m.update(r) and first is None and t > 500:
            first = t
            break
    print(f"Drift injected at t=500, CUSUM detected at t={first} "
          f"(delay {first-500}h)" if first else "not detected")
