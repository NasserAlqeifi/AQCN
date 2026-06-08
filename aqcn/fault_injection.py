"""
fault_injection.py - Inject synthetic sensor faults with a known ground-truth
label mask, so fault detectors can be scored with precision / recall / F1.

QUANT has no labelled faults, so to benchmark detectors fairly we corrupt a clean
signal with realistic fault episodes and record exactly when each one is active.

Fault types (the common low-cost-sensor failure modes):
  spike  - short bursts of large over-reading
  stuck  - the value freezes (electronics hang)
  drift  - a growing additive offset (ageing)
  noise  - variance explosion (failing cell)
"""

from __future__ import annotations
import numpy as np


def inject_faults(signal: np.ndarray, rng, n_events: int = 4,
                  start_frac: float = 0.5):
    """
    Inject n_events faults into the LATER part of the signal (so the early part
    stays clean for detectors that need a clean training period).

    Returns (faulty_signal, label_mask, events).
    """
    x = signal.astype(float).copy()
    n = len(x)
    labels = np.zeros(n, dtype=bool)
    events = []

    valid = np.where(~np.isnan(x))[0]
    valid = valid[valid > int(n * start_frac)]
    if len(valid) < 200:
        return x, labels, events

    mean = float(np.nanmean(signal))
    std = float(np.nanstd(signal)) or 1.0
    fault_types = ["spike", "stuck", "drift", "noise"]

    # space events across the later region
    region = valid
    slots = np.linspace(region[0] + 50, region[-1] - 80, n_events).astype(int)

    for k, start in enumerate(slots):
        ftype = fault_types[k % len(fault_types)]
        dur = int(rng.integers(8, 30))
        end = min(start + dur, n)
        seg = np.arange(start, end)
        seg = seg[~np.isnan(x[seg])]
        if len(seg) < 3:
            continue

        if ftype == "spike":
            x[seg] += rng.uniform(3, 5) * std
        elif ftype == "stuck":
            x[seg] = x[seg[0]]
        elif ftype == "drift":
            x[seg] += np.linspace(0, rng.uniform(3, 5) * std, len(seg))
        elif ftype == "noise":
            x[seg] += rng.normal(0, 3 * std, len(seg))

        labels[seg] = True
        events.append({"type": ftype, "start": int(seg[0]),
                       "end": int(seg[-1]), "dur": len(seg)})

    return x, labels, events
