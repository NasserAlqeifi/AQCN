"""
fault_detector.py - Network-level fault detection.

A single sensor cannot tell whether a sustained deviation is a real pollution
event or its own fault. We resolve this with an ENSEMBLE of three weak
detectors and a persistence rule, following the majority-voting design shown
to work on air-quality networks:

  D1  Spatial consensus : |calibrated - same-city median of peers| / consensus
  D2  Self residual     : |calibrated - reference| / reference   (when ref present)
  D3  Local IQR         : value outside a rolling inter-quartile fence

A reading is "anomalous" when >= 2 of the 3 detectors fire. Persistence then
maps anomalies to status:
    HEALTHY  --(anom >= SUSPECT_H consecutive)-->  SUSPECT
    SUSPECT  --(anom >= FAULTY_H  consecutive)-->  FAULTY
    any      --(clean >= RECOVER_H consecutive)-->  HEALTHY

Spatial consensus is what disambiguates fault from real event: a genuine
pollution episode lifts the WHOLE city, so the peer median moves too and D1
stays quiet; a faulty unit diverges from its peers and D1 fires.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SPATIAL_THRESH = 0.40    # 40% deviation from peer consensus
RESIDUAL_THRESH = 0.50   # 50% deviation from reference
IQR_K = 3.0              # fence = Q1 - k*IQR .. Q3 + k*IQR
IQR_WINDOW = 168         # rolling window for IQR fence (hours)

SUSPECT_H = 3
FAULTY_H = 6
RECOVER_H = 6

HEALTHY, SUSPECT, FAULTY = "HEALTHY", "SUSPECT", "FAULTY"


def detect_faults(
    calibrated: np.ndarray,
    reference: np.ndarray,
    consensus: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    status   : array of HEALTHY/SUSPECT/FAULTY per hour
    anomaly  : bool array, True where >=2 detectors fired
    """
    n = len(calibrated)
    status = np.array([HEALTHY] * n, dtype=object)
    anomaly = np.zeros(n, dtype=bool)

    # Rolling IQR fence on the calibrated signal
    s = pd.Series(calibrated)
    q1 = s.rolling(IQR_WINDOW, min_periods=24).quantile(0.25)
    q3 = s.rolling(IQR_WINDOW, min_periods=24).quantile(0.75)
    iqr = q3 - q1
    low = (q1 - IQR_K * iqr).values
    high = (q3 + IQR_K * iqr).values

    for t in range(n):
        c = calibrated[t]
        if np.isnan(c):
            continue
        votes = 0

        # D1 spatial consensus
        if not np.isnan(consensus[t]) and consensus[t] != 0:
            if abs(c - consensus[t]) / abs(consensus[t]) > SPATIAL_THRESH:
                votes += 1

        # D2 self residual vs reference
        if not np.isnan(reference[t]) and reference[t] != 0:
            if abs(c - reference[t]) / abs(reference[t]) > RESIDUAL_THRESH:
                votes += 1

        # D3 local IQR fence
        if not np.isnan(low[t]) and not np.isnan(high[t]):
            if c < low[t] or c > high[t]:
                votes += 1

        anomaly[t] = votes >= 2

    # Persistence state machine
    cur = HEALTHY
    consec_anom = 0
    consec_clean = 0
    for t in range(n):
        if np.isnan(calibrated[t]):
            status[t] = cur
            continue
        if anomaly[t]:
            consec_anom += 1
            consec_clean = 0
        else:
            consec_clean += 1
            consec_anom = 0

        if cur == HEALTHY and consec_anom >= SUSPECT_H:
            cur = SUSPECT
        elif cur == SUSPECT and consec_anom >= FAULTY_H:
            cur = FAULTY
        elif cur in (SUSPECT, FAULTY) and consec_clean >= RECOVER_H:
            cur = HEALTHY
        status[t] = cur

    return status, anomaly


if __name__ == "__main__":
    # Synthetic: a sensor that diverges from a stable peer consensus
    rng = np.random.default_rng(42)
    n = 400
    consensus = 20 + rng.normal(0, 2, n)
    calibrated = consensus + rng.normal(0, 1, n)
    calibrated[200:] += np.linspace(0, 30, n - 200)  # growing fault
    reference = consensus.copy()
    status, anom = detect_faults(calibrated, reference, consensus)
    first_suspect = next((t for t in range(n) if status[t] == SUSPECT), None)
    first_faulty = next((t for t in range(n) if status[t] == FAULTY), None)
    print(f"Fault grows from t=200. SUSPECT at {first_suspect}, FAULTY at {first_faulty}")
    print(f"Final status: {status[-1]}")
