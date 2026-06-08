"""
uncertainty_monitor.py - Model-uncertainty drift detector.

Idea (from W-UDDR, ACM ToIT 2025, and the IoT calibration-scheduling paper,
arXiv 2506.09186): a calibration model becomes UNSURE when the incoming data no
longer resembles what it was trained on. Rising predictive uncertainty is itself
a drift signal - and it needs NO reference monitor.

Those papers use a Gaussian Process's posterior variance. Our Random Forest
already provides the same kind of epistemic uncertainty for free: the spread of
predictions across its 200 trees. We monitor that spread.

Mechanism: record the forest's tree-spread over the training window (the normal
level). For each new hour, keep a rolling mean of the tree-spread; if it climbs
above (baseline_mean + k * baseline_std), the model is significantly less sure
than it was when trained -> declare drift.

Interface: calibrate(...) + update(signals: dict) -> bool. Reads
signals["uncertainty"] (the RF tree-spread for the current row).
"""

from __future__ import annotations

import numpy as np

ROLL = 48             # rolling window (hours) for current uncertainty
K_SIGMA = 3.0         # alarm when rolling mean exceeds baseline + k*sigma
COOLDOWN = 168        # min hours between alarms


class UncertaintyDriftMonitor:
    def __init__(self, k_sigma: float = K_SIGMA, roll: int = ROLL):
        self.k_sigma = k_sigma
        self.roll = roll
        self.base_mean = 0.0
        self.base_std = 1.0
        self.threshold = np.inf
        self.buffer: list[float] = []
        self._since_alarm = 10 ** 9
        self._ready = False

    def calibrate(self, baseline):
        """Accepts a 1-D array of tree-spread, or a dict with key 'uncertainty'."""
        u = baseline.get("uncertainty") if isinstance(baseline, dict) else baseline
        u = np.asarray(u, dtype=float)
        u = u[~np.isnan(u)]
        if len(u) < 24:
            self._ready = False
            return
        self.base_mean = float(np.mean(u))
        self.base_std = float(np.std(u)) if np.std(u) > 0 else 1.0
        self.threshold = self.base_mean + self.k_sigma * self.base_std
        self.buffer = []
        self._since_alarm = 10 ** 9
        self._ready = True

    def update(self, signals: dict) -> bool:
        self._since_alarm += 1
        if not self._ready:
            return False
        u = signals.get("uncertainty", np.nan)
        if u is None or (isinstance(u, float) and np.isnan(u)):
            return False
        self.buffer.append(float(u))
        if len(self.buffer) > self.roll:
            self.buffer.pop(0)
        if len(self.buffer) < self.roll:
            return False
        if self._since_alarm < COOLDOWN:
            return False
        if float(np.mean(self.buffer)) > self.threshold:
            self._since_alarm = 0
            self.buffer = []
            return True
        return False


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    # baseline tree-spread ~ small; then it grows (model unsure)
    base = np.abs(rng.normal(1.0, 0.2, 1000))
    stable = np.abs(rng.normal(1.0, 0.2, 400))
    rising = np.abs(rng.normal(1.0, 0.2, 400)) + np.linspace(0, 3, 400)
    stream = np.concatenate([stable, rising])
    m = UncertaintyDriftMonitor()
    m.calibrate(base)
    first = None
    for t, v in enumerate(stream):
        if m.update({"uncertainty": v}) and t >= 400 and first is None:
            first = t
            break
    print(f"Uncertainty: model-confidence drops from t=400, detected at t={first}"
          + (f" (delay {first-400}h)" if first else " - not detected"))
