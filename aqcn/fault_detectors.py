"""
fault_detectors.py - A field of fault detectors with a common interface.

Each detector takes the (faulty) signal plus optional context and returns a
boolean mask of where it thinks a fault is active. They are scored against the
injected ground-truth labels with precision / recall / F1.

Detectors:
  zscore      - rolling z-score threshold (simple baseline)
  iqr         - rolling inter-quartile fence (our component)
  consensus   - deviation from same-city peer median (our component)
  ensemble    - consensus + residual + IQR + persistence (our full method)
  iforest     - Isolation Forest on sliding windows (unsupervised ML)
  ocsvm       - One-Class SVM on sliding windows (unsupervised ML)
  autoencoder - MLP reconstruction error on sliding windows (SOTA-style)
"""

from __future__ import annotations
import numpy as np
import pandas as pd

WIN = 24  # sliding-window length for the ML detectors


# ---------- simple statistical ----------
def detect_zscore(signal, window=168, k=3.5):
    s = pd.Series(signal)
    mu = s.rolling(window, min_periods=24).mean()
    sd = s.rolling(window, min_periods=24).std()
    z = (s - mu).abs() / sd.replace(0, np.nan)
    return (z > k).fillna(False).values


def detect_iqr(signal, window=168, k=3.0):
    s = pd.Series(signal)
    q1 = s.rolling(window, min_periods=24).quantile(0.25)
    q3 = s.rolling(window, min_periods=24).quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - k * iqr, q3 + k * iqr
    return ((s < low) | (s > high)).fillna(False).values


def detect_consensus(signal, consensus, thresh=0.4):
    s = np.asarray(signal); c = np.asarray(consensus)
    with np.errstate(invalid="ignore", divide="ignore"):
        dev = np.abs(s - c) / np.abs(c)
    out = dev > thresh
    return np.nan_to_num(out, nan=0.0).astype(bool)


def detect_ensemble(signal, reference, consensus):
    """Our full method: 2-of-3 vote + persistence state machine."""
    from aqcn.fault_ensemble import detect_faults
    status, anomaly = detect_faults(np.asarray(signal), np.asarray(reference),
                                    np.asarray(consensus))
    return np.isin(status, ["SUSPECT", "FAULTY"])


# ---------- ML detectors on sliding windows ----------
def _windows(signal, win=WIN):
    s = np.asarray(signal, dtype=float)
    n = len(s)
    feats = np.full((n, win), np.nan)
    for i in range(n):
        lo = max(0, i - win + 1)
        seg = s[lo:i + 1]
        feats[i, win - len(seg):] = seg
    # fill any internal nan with row mean
    rowmean = np.nanmean(feats, axis=1, keepdims=True)
    feats = np.where(np.isnan(feats), rowmean, feats)
    valid = ~np.isnan(feats).any(axis=1)
    return feats, valid


def _ml_detect(signal, model_kind, train_frac=0.5):
    feats, valid = _windows(signal)
    n = len(signal)
    mask = np.zeros(n, dtype=bool)
    idx = np.where(valid)[0]
    if len(idx) < 100:
        return mask
    cut = int(n * train_frac)
    train_idx = idx[idx < cut]
    if len(train_idx) < 50:
        return mask
    Xtr = feats[train_idx]

    if model_kind == "iforest":
        from sklearn.ensemble import IsolationForest
        m = IsolationForest(n_estimators=100, contamination=0.05,
                            random_state=42).fit(Xtr)
        pred = m.predict(feats[idx])           # -1 = anomaly
        mask[idx] = pred == -1
    elif model_kind == "ocsvm":
        from sklearn.svm import OneClassSVM
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(Xtr)
        m = OneClassSVM(nu=0.05, gamma="scale").fit(sc.transform(Xtr))
        pred = m.predict(sc.transform(feats[idx]))
        mask[idx] = pred == -1
    elif model_kind == "autoencoder":
        mask = _autoencoder_detect(feats, idx, train_idx)
    return mask


def _autoencoder_detect(feats, idx, train_idx):
    import torch, torch.nn as nn
    torch.manual_seed(42)
    n = feats.shape[0]
    mask = np.zeros(n, dtype=bool)
    Xtr = torch.tensor(feats[train_idx], dtype=torch.float32)
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
    Xtr = (Xtr - mu) / sd

    d = feats.shape[1]
    ae = nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, 4), nn.ReLU(),
                       nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, d))
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    for _ in range(150):
        opt.zero_grad()
        loss = lossf(ae(Xtr), Xtr)
        loss.backward(); opt.step()

    ae.eval()
    with torch.no_grad():
        train_err = ((ae(Xtr) - Xtr) ** 2).mean(1).numpy()
        thresh = train_err.mean() + 3 * train_err.std()
        Xall = (torch.tensor(feats[idx], dtype=torch.float32) - mu) / sd
        err = ((ae(Xall) - Xall) ** 2).mean(1).numpy()
    mask[idx] = err > thresh
    return mask


def detect_iforest(signal, **kw):
    return _ml_detect(signal, "iforest")


def detect_ocsvm(signal, **kw):
    return _ml_detect(signal, "ocsvm")


def detect_autoencoder(signal, **kw):
    return _ml_detect(signal, "autoencoder")


# ---------- scoring ----------
def score(pred_mask, true_mask):
    pred = np.asarray(pred_mask, dtype=bool)
    true = np.asarray(true_mask, dtype=bool)
    tp = int(np.sum(pred & true))
    fp = int(np.sum(pred & ~true))
    fn = int(np.sum(~pred & true))
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * prec * rec / (prec + rec)
          if prec and rec and not np.isnan(prec) and not np.isnan(rec) else float("nan"))
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
