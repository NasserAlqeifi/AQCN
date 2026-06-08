"""
maml_calibrator.py - Meta-learning calibrator for the cold-start problem.

Implements a first-order MAML / Reptile meta-learner (the variant used in the
air-quality few-shot calibration papers, e.g. Arora 2021). Plain idea:

  Normal training learns ONE model for ONE sensor.
  Meta-learning trains ACROSS many "donor" sensors to learn a good STARTING
  point - one that adapts to a brand-new sensor after only a few gradient steps
  on a handful of that sensor's reference readings.

Algorithm (Reptile, first-order MAML):
  init meta-weights theta
  repeat:
     pick a donor sensor (a "task")
     phi = theta
     do K SGD steps on that sensor's data        -> adapted phi
     theta <- theta + meta_lr * (phi - theta)     -> nudge start toward adapted
At test time: copy theta, take a few SGD steps on the new sensor's warmup data,
predict.

Base model: small MLP. Features standardised + median-imputed using donor-pool
statistics only (no leakage from test sensors).
"""

from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn as nn

from aqcn.features import feature_columns

torch.manual_seed(42)
np.random.seed(42)


class _MLP(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MamlCalibrator:
    def __init__(self, features: list[str], device=None):
        self.features = features
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _MLP(len(features)).to(self.device)
        self.feat_mean = None
        self.feat_std = None
        self.y_mean = 0.0
        self.y_std = 1.0

    # ---- preprocessing fitted on donor pool only ----
    def fit_scaler(self, X: np.ndarray, y: np.ndarray):
        self.feat_mean = np.nanmean(X, axis=0)
        self.feat_std = np.nanstd(X, axis=0)
        self.feat_std[self.feat_std == 0] = 1.0
        self.y_mean = float(np.nanmean(y))
        self.y_std = float(np.nanstd(y)) or 1.0

    def _prep(self, X, y=None):
        Xi = np.where(np.isnan(X), self.feat_mean, X)
        Xs = (Xi - self.feat_mean) / self.feat_std
        Xt = torch.tensor(Xs, dtype=torch.float32, device=self.device)
        if y is None:
            return Xt
        ys = (y - self.y_mean) / self.y_std
        yt = torch.tensor(ys, dtype=torch.float32, device=self.device)
        return Xt, yt

    # ---- meta-training (Reptile) ----
    def meta_train(self, tasks: list[tuple[np.ndarray, np.ndarray]],
                   iterations=2000, inner_steps=5, inner_lr=0.01, meta_lr=0.1):
        """tasks: list of (X, y) arrays, one per donor sensor."""
        loss_fn = nn.MSELoss()
        rng = np.random.default_rng(42)
        for it in range(iterations):
            X, y = tasks[rng.integers(len(tasks))]
            if len(X) < 40:
                continue
            idx = rng.choice(len(X), size=min(64, len(X)), replace=False)
            Xt, yt = self._prep(X[idx], y[idx])

            fast = copy.deepcopy(self.model)
            opt = torch.optim.SGD(fast.parameters(), lr=inner_lr)
            for _ in range(inner_steps):
                opt.zero_grad()
                loss = loss_fn(fast(Xt), yt)
                loss.backward()
                opt.step()

            # Reptile meta-update: theta += meta_lr * (phi - theta)
            with torch.no_grad():
                for p_meta, p_fast in zip(self.model.parameters(), fast.parameters()):
                    p_meta.add_(meta_lr * (p_fast.data - p_meta.data))

    # ---- adapt to a new sensor's warmup data, then predict ----
    def adapt_and_predict(self, X_support, y_support, X_query,
                          steps=20, lr=0.01) -> np.ndarray:
        loss_fn = nn.MSELoss()
        fast = copy.deepcopy(self.model)
        opt = torch.optim.SGD(fast.parameters(), lr=lr)
        Xs, ys = self._prep(X_support, y_support)
        for _ in range(steps):
            opt.zero_grad()
            loss = loss_fn(fast(Xs), ys)
            loss.backward()
            opt.step()
        fast.eval()
        with torch.no_grad():
            Xq = self._prep(X_query)
            pred_std = fast(Xq).cpu().numpy()
        return pred_std * self.y_std + self.y_mean


def matrix(df, features):
    """Return (X, y) numpy arrays for rows with signal + target present."""
    X = df[features].values.astype(float)
    y = df["ref_no2"].values.astype(float)
    ok = ~np.isnan(df["no2_raw"].values) & ~np.isnan(y)
    return X[ok], y[ok]
