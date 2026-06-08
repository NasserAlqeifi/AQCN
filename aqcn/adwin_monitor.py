"""
adwin_monitor.py - ADWIN drift detector (self-tuning, with guarantees).

ADWIN ("ADaptive WINdowing", Bifet & Gavalda 2007) keeps a window of recent
residuals whose size adapts automatically. It splits the window into an older
and a newer part; if their means differ by more than a Hoeffding bound allows,
it declares drift and drops the old part. Its only knob, delta, is the allowed
false-alarm probability (default 0.002) - so unlike CUSUM there is no
sensitivity threshold to hand-tune, and it carries a statistical guarantee on
the false-positive rate.

Interface matches CusumDriftMonitor: calibrate(baseline) + update(residual)->bool
so it is a drop-in replacement in the pipeline.
"""

from __future__ import annotations

import numpy as np
from river import drift


class AdwinDriftMonitor:
    def __init__(self, delta: float = 0.002):
        self.delta = delta
        self._adwin = drift.ADWIN(delta=delta)

    def calibrate(self, baseline):
        """Warm up ADWIN on the clean baseline residuals (array or dict)."""
        r = baseline.get("residual") if isinstance(baseline, dict) else baseline
        r = np.asarray(r, dtype=float)
        self._adwin = drift.ADWIN(delta=self.delta)
        for v in r:
            if not np.isnan(v):
                self._adwin.update(float(v))

    def update(self, signal) -> bool:
        residual = signal.get("residual") if isinstance(signal, dict) else signal
        if residual is None or np.isnan(residual):
            return False
        self._adwin.update(float(residual))
        return bool(self._adwin.drift_detected)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    clean = rng.normal(0, 1, 500)
    drifting = np.linspace(0, 6, 300) + rng.normal(0, 1, 300)
    series = np.concatenate([clean, drifting])
    m = AdwinDriftMonitor()
    m.calibrate(clean)
    first = None
    for t in range(500, len(series)):
        if m.update(series[t]) and first is None:
            first = t
            break
    print(f"ADWIN: drift from t=500 detected at t={first} "
          f"(delay {first-500}h)" if first else "not detected")
