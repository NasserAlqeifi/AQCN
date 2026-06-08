"""
evaluator.py - Scientific metric suite for calibration evaluation.

All metrics computed on the held-out TEST window against the ratified
reference monitor.

  RMSE   Root Mean Square Error (ppb)
  MAE    Mean Absolute Error (ppb)
  MBE    Mean Bias Error, signed (ppb)
  nRMSE  RMSE as % of mean observation
  R2     Coefficient of determination
  r      Pearson correlation
  d      Willmott Index of Agreement [0,1]
"""

from __future__ import annotations
import numpy as np


def _mask(o, p):
    return ~(np.isnan(o) | np.isnan(p))

def rmse(o, p):
    m = _mask(o, p); return float(np.sqrt(np.mean((o[m]-p[m])**2))) if m.sum() >= 2 else float("nan")
def mae(o, p):
    m = _mask(o, p); return float(np.mean(np.abs(o[m]-p[m]))) if m.sum() >= 2 else float("nan")
def mbe(o, p):
    m = _mask(o, p); return float(np.mean(p[m]-o[m])) if m.sum() >= 2 else float("nan")
def nrmse(o, p):
    r = rmse(o, p); mn = np.nanmean(o[_mask(o, p)]) if _mask(o, p).sum() else np.nan
    return r/mn*100 if (not np.isnan(r) and mn not in (0, np.nan)) else float("nan")
def r2(o, p):
    m = _mask(o, p)
    if m.sum() < 5: return float("nan")
    ss = np.sum((o[m]-p[m])**2); st = np.sum((o[m]-np.mean(o[m]))**2)
    return float(1-ss/st) if st > 0 else float("nan")
def pearson(o, p):
    m = _mask(o, p); return float(np.corrcoef(o[m], p[m])[0, 1]) if m.sum() >= 5 else float("nan")
def willmott(o, p):
    m = _mask(o, p)
    if m.sum() < 5: return float("nan")
    om = np.mean(o[m]); num = np.sum((o[m]-p[m])**2)
    den = np.sum((np.abs(p[m]-om)+np.abs(o[m]-om))**2)
    return float(1-num/den) if den > 0 else float("nan")

def all_metrics(obs: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "n": int(_mask(obs, pred).sum()),
        "RMSE": rmse(obs, pred), "MAE": mae(obs, pred), "MBE": mbe(obs, pred),
        "nRMSE": nrmse(obs, pred), "R2": r2(obs, pred),
        "r": pearson(obs, pred), "d": willmott(obs, pred),
    }
