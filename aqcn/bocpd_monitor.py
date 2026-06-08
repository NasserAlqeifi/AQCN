"""
bocpd_monitor.py - Bayesian Online Changepoint Detection (Adams & MacKay 2007).

Instead of a hard yes/no alarm, BOCPD maintains, at every hour, a probability
distribution over the "run length" - how many hours since the last changepoint.
When a changepoint becomes likely, probability mass collapses onto run length 0.
We expose that collapse as a changepoint PROBABILITY, and signal drift when it
exceeds a threshold.

Observation model: Gaussian with unknown mean and variance, using the conjugate
Normal-Inverse-Gamma prior, giving a Student-t predictive distribution.
Hazard: constant (geometric prior on run length) with rate 1/expected_run.

Interface matches CusumDriftMonitor (calibrate + update->bool) so it is a
drop-in replacement; the raw probability is also exposed via .last_cp_prob.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import t as student_t


class BocpdDriftMonitor:
    def __init__(self, expected_run: float = 336.0, prob_threshold: float = 0.05,
                 collapse_frac: float = 0.4, min_run_before: int = 24,
                 max_run: int = 1000):
        self.hazard = 1.0 / expected_run     # constant hazard
        self.prob_threshold = prob_threshold     # raw cp-prob spike trigger
        self.collapse_frac = collapse_frac       # run-length collapse trigger
        self.min_run_before = min_run_before     # only trust a collapse from a mature run
        self.max_run = max_run
        self.last_cp_prob = 0.0
        self.map_run_length = 0
        self._prev_map = 0
        self._reset_prior()

    def _reset_prior(self, mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0):
        # Normal-Inverse-Gamma prior parameters
        self.mu0, self.kappa0, self.alpha0, self.beta0 = mu0, kappa0, alpha0, beta0
        # Per-run-length parameter arrays; start with run length 0 only
        self.mu = np.array([mu0], dtype=float)
        self.kappa = np.array([kappa0], dtype=float)
        self.alpha = np.array([alpha0], dtype=float)
        self.beta = np.array([beta0], dtype=float)
        self.R = np.array([1.0])             # run-length distribution

    def calibrate(self, baseline):
        r = baseline.get("residual") if isinstance(baseline, dict) else baseline
        r = np.asarray(r, dtype=float)
        r = r[~np.isnan(r)]
        if len(r) >= 24:
            mu0 = float(np.mean(r))
            var0 = float(np.var(r)) if np.var(r) > 0 else 1.0
            self._reset_prior(mu0=mu0, kappa0=1.0, alpha0=2.0, beta0=var0)
            # warm up the run-length distribution on the baseline so it is
            # "mature" before the live stream begins (lets a later collapse fire)
            for v in r:
                self._update_core(float(v))
        else:
            self._reset_prior()

    def _predictive_prob(self, x: float) -> np.ndarray:
        # Student-t predictive for each current run length
        df = 2 * self.alpha
        scale = np.sqrt(self.beta * (self.kappa + 1) / (self.alpha * self.kappa))
        return student_t.pdf(x, df=df, loc=self.mu, scale=scale)

    def update(self, signal) -> bool:
        residual = signal.get("residual") if isinstance(signal, dict) else signal
        if residual is None or np.isnan(residual):
            return False
        return self._update_core(float(residual))

    def _update_core(self, x: float) -> bool:
        pred = self._predictive_prob(x)                 # length = len(R)
        # growth probabilities (run length increments)
        growth = self.R * pred * (1.0 - self.hazard)
        # changepoint probability (run length resets to 0)
        cp = float(np.sum(self.R * pred * self.hazard))

        new_R = np.empty(len(self.R) + 1)
        new_R[0] = cp
        new_R[1:] = growth
        total = new_R.sum()
        if total <= 0 or not np.isfinite(total):
            # The new observation is incompatible with EVERY current run
            # (predictive underflowed) - this is itself a strong changepoint.
            # Restart the run-length distribution and signal drift.
            self._reset_prior(mu0=x, beta0=max(self.beta0, 1.0))
            self.last_cp_prob = 1.0
            self.map_run_length = 0
            self._prev_map = 0
            return True
        new_R /= total

        # update sufficient statistics (conjugate NIG update), prepending prior
        new_mu = np.empty(len(self.mu) + 1)
        new_kappa = np.empty(len(self.kappa) + 1)
        new_alpha = np.empty(len(self.alpha) + 1)
        new_beta = np.empty(len(self.beta) + 1)

        new_mu[0], new_kappa[0] = self.mu0, self.kappa0
        new_alpha[0], new_beta[0] = self.alpha0, self.beta0

        new_kappa[1:] = self.kappa + 1.0
        new_mu[1:] = (self.kappa * self.mu + x) / (self.kappa + 1.0)
        new_alpha[1:] = self.alpha + 0.5
        new_beta[1:] = self.beta + (self.kappa * (x - self.mu) ** 2) / (2.0 * (self.kappa + 1.0))

        # truncate to max_run for efficiency
        if len(new_R) > self.max_run:
            new_R = new_R[:self.max_run]
            new_mu = new_mu[:self.max_run]
            new_kappa = new_kappa[:self.max_run]
            new_alpha = new_alpha[:self.max_run]
            new_beta = new_beta[:self.max_run]
            new_R /= new_R.sum()

        self.R, self.mu, self.kappa, self.alpha, self.beta = (
            new_R, new_mu, new_kappa, new_alpha, new_beta)

        self.last_cp_prob = float(new_R[0])
        self._prev_map = self.map_run_length
        self.map_run_length = int(np.argmax(self.R))

        # Trigger on EITHER a raw changepoint-probability spike (abrupt jump)
        # OR a collapse of the most-likely run length from a mature run
        # (the run "resets", i.e. the model now believes we are early in a new
        # regime). This makes BOCPD practical for both abrupt and slower shifts.
        spike = self.last_cp_prob > self.prob_threshold
        collapse = (self._prev_map >= self.min_run_before
                    and self.map_run_length < self.collapse_frac * self._prev_map)
        return bool(spike or collapse)


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    def run(series, change_at, label):
        # matches pipeline usage: calibrate (warms up run length on baseline),
        # then stream the live portion
        m = BocpdDriftMonitor()
        m.calibrate(series[:change_at])
        first = None
        for t in range(change_at, len(series)):
            if m.update(series[t]) and first is None:
                first = t
                break
        print(f"BOCPD {label}: change at t={change_at}, detected at t={first}"
              + (f" (delay {first-change_at}h)" if first else " - not detected"))

    # 1) abrupt step change (BOCPD's home turf)
    step = np.concatenate([rng.normal(0, 1, 500), rng.normal(5, 1, 300)])
    run(step, 500, "abrupt step")

    # 2) gradual ramp (harder for changepoint detection)
    ramp = np.concatenate([rng.normal(0, 1, 500),
                           np.linspace(0, 6, 300) + rng.normal(0, 1, 300)])
    run(ramp, 500, "gradual ramp")
