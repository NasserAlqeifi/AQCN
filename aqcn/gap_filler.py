"""
gap_filler.py - Chronos-based imputation of short gaps in the raw sensor signal.

The RF calibrator can only act on hours where the raw NO2 signal exists.
When the sensor drops out for a few hours, we impute the raw signal with the
Chronos-T5-tiny foundation model (its one genuinely strong role here), so those
hours become calibratable. Long gaps (> MAX_GAP_H) are left missing.

This is the only place a foundation model is used in the pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

MAX_GAP_H = 6          # only impute gaps up to this length
CONTEXT_H = 168        # history window fed to Chronos
NUM_SAMPLES = 10
MAX_CONTEXT = 512

torch.manual_seed(42)
np.random.seed(42)

_pipe = None


def _get_pipe():
    global _pipe
    if _pipe is None:
        from chronos import ChronosPipeline
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[gap_filler] Loading Chronos-T5-tiny on {device.upper()}...")
        _pipe = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-tiny", device_map=device, torch_dtype=torch.float32
        )
    return _pipe


def _forecast_one(context: np.ndarray) -> float:
    valid = context[~np.isnan(context)]
    if len(valid) < 3:
        return float("nan")
    ctx = torch.tensor(valid[-MAX_CONTEXT:].astype(np.float32)).unsqueeze(0)
    pipe = _get_pipe()
    with torch.no_grad():
        s = pipe.predict(inputs=ctx, prediction_length=1, num_samples=NUM_SAMPLES)
    out = float(torch.quantile(s[0, :, 0].float().cpu(), 0.5).item())
    del s
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def fill_signal_gaps(signal: pd.Series) -> tuple[pd.Series, np.ndarray, int]:
    """
    Impute short gaps in a raw signal series.

    Returns
    -------
    filled    : pd.Series with short gaps imputed
    was_filled: bool array, True where a value was imputed
    n_filled  : number of imputed hours
    """
    values = signal.values.astype(float).copy()
    n = len(values)
    was_filled = np.zeros(n, dtype=bool)

    i = 0
    while i < n:
        if not np.isnan(values[i]):
            i += 1
            continue
        j = i
        while j < n and np.isnan(values[j]):
            j += 1
        gap_len = j - i
        if gap_len <= MAX_GAP_H:
            for t in range(i, j):
                ctx_start = max(0, t - CONTEXT_H)
                val = _forecast_one(values[ctx_start:t])
                if not np.isnan(val):
                    values[t] = val
                    was_filled[t] = True
        i = j

    return pd.Series(values, index=signal.index), was_filled, int(was_filled.sum())


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from aqcn.features import build_feature_frames

    frames = build_feature_frames(use_cache=True)
    sid = "MCH__Prax1_S1"
    df = frames[sid]
    raw = df["no2_raw"]
    print(f"[{sid}] raw missing hours: {raw.isna().sum()}")
    filled, mask, n = fill_signal_gaps(raw)
    print(f"Imputed {n} hours (gaps <= {MAX_GAP_H}h)")
    print(f"Missing after fill: {filled.isna().sum()}")
