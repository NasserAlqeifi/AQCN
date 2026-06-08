"""
ks_monitor.py - Distribution-shift drift detector (two-sample Kolmogorov-Smirnov).

Unlike CUSUM/ADWIN/BOCPD (which watch the error against the reference), this
detector watches whether the SHAPE of the incoming sensor-signal distribution
has drifted away from the training distribution. It needs NO reference monitor,
which is its key deployment advantage.

Mechanism: keep the training values of the raw signal. For each new hour, add it
to a sliding window; once the window is full, run a two-sample KS test between
the training sample and the window. The KS statistic is the largest gap between
the two cumulative distributions (0 = identical, 1 = completely separated). When
it exceeds a threshold, declare drift.

Validated for low-cost air-quality sensors in Concept Drift Mitigation in
Low-Cost AQ Networks (PMC11086340), which uses a KS threshold of 0.3.

Interface: calibrate(...) + update(signals: dict) -> bool, matching the other
monitors. It reads signals["signal"] (the raw sensor value), not the residual.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp

WINDOW = 168          # sliding window (1 week) for the current distribution
KS_THRESHOLD = 0.30   # max CDF gap that triggers drift (per the AQ paper)
COOLDOWN = 168        # min hours between alarms


class KsDriftMonitor:
    def __init__(self, threshold: float = KS_THRESHOLD, window: int = WINDOW):
        self.threshold = threshold
        self.window = window
        self.train_sample = None
        self.buffer: list[float] = []
        self._since_alarm = 10 ** 9

    def calibrate(self, baseline):
        """Accepts a 1-D array of the raw signal, or a dict with key 'signal'."""
        b = baseline.get("signal") if isinstance(baseline, dict) else baseline
        b = np.asarray(b, dtype=float)
        b = b[~np.isnan(b)]
        # subsample for speed if very long
        if len(b) > 2000:
            idx = np.linspace(0, len(b) - 1, 2000).astype(int)
            b = b[idx]
        self.train_sample = b if len(b) >= 30 else None
        self.buffer = []
        self._since_alarm = 10 ** 9

    def update(self, signals: dict) -> bool:
        self._since_alarm += 1
        if self.train_sample is None:
            return False
        x = signals.get("signal", np.nan)
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return False
        self.buffer.append(float(x))
        if len(self.buffer) > self.window:
            self.buffer.pop(0)
        if len(self.buffer) < self.window:
            return False
        if self._since_alarm < COOLDOWN:
            return False
        stat, _ = ks_2samp(self.train_sample, np.asarray(self.buffer))
        if stat > self.threshold:
            self._since_alarm = 0
            # reset buffer so the next window is fresh after recalibration
            self.buffer = []
            return True
        return False


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    train = rng.normal(10, 2, 1000)
    # stream: stable, then distribution shifts upward
    stable = rng.normal(10, 2, 400)
    shifted = rng.normal(16, 2, 400)
    stream = np.concatenate([stable, shifted])
    m = KsDriftMonitor()
    m.calibrate(train)
    first = None
    for t, v in enumerate(stream):
        if m.update({"signal": v}) and t >= 400 and first is None:
            first = t
            break
    print(f"KS: distribution shifts at t=400, detected at t={first}"
          + (f" (delay {first-400}h)" if first else " - not detected"))
