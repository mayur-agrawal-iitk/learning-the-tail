r"""Semiparametric dynamic models for (VaR, ES), estimated by minimising the FZ0 loss.

These models specify only the dynamics of the two functionals, not the full conditional
distribution, and are M-estimated: theta_hat = argmin (1/T) sum_t L_FZ0(r_t, v_t(theta), e_t(theta)).

1. CAViaR-ES (Engle & Manganelli 2004 asymmetric slope + multiplicative ES, Taylor 2019):
       v_{t+1} = omega + beta v_t - theta_p r_t^+ - theta_n r_t^-,     e_t = (1 + exp(g)) v_t.

2. FZ-GAS one-factor (Patton, Ziegel & Chen 2019):
       v_t = a exp(kappa_t),  e_t = b exp(kappa_t),  b < a < 0,
       kappa_{t+1} = beta kappa_t + gamma lambda_t,
       lambda_t = (1/alpha) 1{r_t <= v_t} r_t / e_t - 1.
   lambda_t is (minus) the score of FZ0 w.r.t. kappa:  dL/dkappa = 1 - 1{r<=v} r /(alpha e),
   so it is a martingale difference under correct specification. After a large loss
   lambda_t >> 0 and risk jumps up; on calm days lambda_t = -1 and risk decays.

3. FZ-Hybrid (Patton, Ziegel & Chen 2019): adds a GARCH-like term,
       kappa_{t+1} = beta kappa_t + gamma lambda_t + delta log|r_t|.

Estimation: grid of starting values -> Nelder-Mead from the best few, warm-started from
the previous window's optimum. Each alpha level is estimated separately.

Sign restrictions (imposed through the objective): gamma >= 0 and delta >= 0 in the GAS
models, omega < 0, 0 <= beta < 1 and theta_p, theta_n >= 0 in CAViaR. Without them, at
alpha = 1% with only ~10-20 violations per window, the optimiser occasionally prefers
gamma < 0 (risk *falls* after a violation). That fits marginally better in sample but is
self-reinforcing out of sample (each violation shrinks VaR, causing more violations) and
drove VaR to ~0 during the COVID crash in our pilot runs.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit
from scipy.optimize import minimize

from ..scoring import empirical_var_es
from .base import Estimator, Forecasts

PENALTY = 1e10
KAPPA_MAX = 10.0
ABS_FLOOR = 0.01  # floor for |r| inside log|r| (1 basis point, returns in percent)


@njit(cache=True)
def _fz0_mean(r, v, e, alpha):
    n = r.size
    tot = 0.0
    for t in range(n):
        if e[t] >= -1e-10 or v[t] >= 0.0 or e[t] > v[t]:
            return PENALTY
        hit = 1.0 if r[t] <= v[t] else 0.0
        tot += -hit * (v[t] - r[t]) / (alpha * e[t]) + v[t] / e[t] + math.log(-e[t]) - 1.0
    return tot / n


@njit(cache=True)
def caviar_path(r, omega, beta, th_p, th_n, g, v0):
    n = r.size
    v = np.empty(n + 1)
    v[0] = v0
    for t in range(n):
        rp = r[t] if r[t] > 0 else 0.0
        rn = -r[t] if r[t] < 0 else 0.0
        v[t + 1] = omega + beta * v[t] - th_p * rp - th_n * rn
    e = (1.0 + math.exp(g)) * v
    return v, e


@njit(cache=True)
def gas_path(r, beta, gamma, delta, a, b, alpha, kappa0):
    """FZ-GAS recursion; delta = 0 gives the one-factor model, delta != 0 the hybrid."""
    n = r.size
    v = np.empty(n + 1)
    e = np.empty(n + 1)
    k = kappa0
    for t in range(n + 1):
        v[t] = a * math.exp(k)
        e[t] = b * math.exp(k)
        if t == n:
            break
        hit = 1.0 if r[t] <= v[t] else 0.0
        lam = hit * r[t] / (alpha * e[t]) - 1.0
        k = beta * k + gamma * lam
        if delta != 0.0:
            k += delta * math.log(max(abs(r[t]), ABS_FLOOR))
        k = min(max(k, -KAPPA_MAX), KAPPA_MAX)
    return v, e


def _optimise(obj, starts: list[np.ndarray], prev: np.ndarray | None, scale: np.ndarray,
              rng: np.random.Generator, n_random: int = 60, n_best: int = 3) -> np.ndarray:
    """Multi-start optimisation of a non-smooth objective (FZ0 has an indicator in it).

    1. score the deterministic grid and the previous window's optimum (warm start);
    2. add ``n_random`` random perturbations around the best grid points and the warm
       start (Gaussian, per-parameter ``scale``, random magnitude in [0.5, 3]);
    3. Nelder-Mead from the ``n_best`` best candidates, each polished with Powell.
    Nelder-Mead alone gets stuck at kinks; in our diagnostics this procedure closes the
    in-sample gap to a 300-start search to < 0.005 (vs up to 0.04 for grid + NM).
    """
    cands = list(starts) + ([prev] if prev is not None else [])
    vals = np.array([obj(c) for c in cands])
    centres = [cands[i] for i in np.argsort(vals)[:2]] + ([prev] if prev is not None else [])
    for c in centres:
        for _ in range(n_random // len(centres)):
            cands.append(c + rng.normal(0.0, scale) * rng.uniform(0.5, 3.0))
    vals = np.array([obj(c) for c in cands])
    best_x, best_f = cands[int(np.argmin(vals))], float(vals.min())
    for i in np.argsort(vals)[:n_best]:
        res = minimize(obj, cands[i], method="Nelder-Mead",
                       options={"maxiter": 4000, "xatol": 1e-7, "fatol": 1e-10, "adaptive": True})
        res = minimize(obj, res.x, method="Powell",
                       options={"maxiter": 8000, "xtol": 1e-7, "ftol": 1e-10})
        if res.fun < best_f:
            best_x, best_f = res.x, float(res.fun)
    return np.asarray(best_x)


class _PerAlphaFZ(Estimator):
    """Shared driver: estimate parameters per alpha, then filter through the forecast block."""

    name = ""
    scale: np.ndarray = np.array([])   # perturbation scale per parameter (random restarts)

    def __init__(self, seed: int = 0):
        self.names = [self.name]
        self.prev: dict[float, np.ndarray] = {}
        self.rng = np.random.default_rng(seed)

    # subclasses implement these three
    def _starts(self, rw, alpha):
        raise NotImplementedError

    def _path(self, theta, r, alpha, rw):
        raise NotImplementedError

    def _valid(self, theta) -> bool:
        return True

    def fit_predict(self, r, t0, t1, window, alphas) -> Forecasts:
        n = t1 - t0
        rw = np.ascontiguousarray(r[t0 - window:t0])
        rf = np.ascontiguousarray(r[t0 - window:t1 - 1])  # window + block (up to t1-2)
        var = np.empty((n, len(alphas)))
        es = np.empty((n, len(alphas)))
        for k, a in enumerate(alphas):
            def obj(theta, a=a):
                if not self._valid(theta):
                    return PENALTY
                v, e = self._path(theta, rw, a, rw)
                return _fz0_mean(rw, v[:-1], e[:-1], a)

            theta = _optimise(obj, self._starts(rw, a), self.prev.get(a), self.scale, self.rng)
            self.prev[a] = theta
            v, e = self._path(theta, rf, a, rw)
            var[:, k] = v[window:window + n]
            es[:, k] = e[window:window + n]
        return {self.name: (var, es)}


class CaviarES(_PerAlphaFZ):
    name = "CAViaR-ES"
    scale = np.array([0.05, 0.03, 0.05, 0.05, 0.3])

    def _valid(self, th):
        # omega < 0, 0 <= beta < 1, theta_p, theta_n >= 0  =>  v_t < 0 for every t
        return th[0] < 0 and 0.0 <= th[1] < 0.9999 and th[2] >= 0 and th[3] >= 0

    def _starts(self, rw, alpha):
        v_hat, e_hat = empirical_var_es(rw, alpha)
        m_p, m_n = np.mean(np.maximum(rw, 0)), np.mean(np.maximum(-rw, 0))
        g = np.log(max(e_hat / v_hat - 1.0, 1e-3))
        starts = []
        for beta in (0.8, 0.9, 0.95):
            for th_n in (0.1, 0.3, 0.6):
                for th_p in (0.05, 0.2):
                    omega = v_hat * (1 - beta) + th_p * m_p + th_n * m_n
                    starts.append(np.array([omega, beta, th_p, th_n, g]))
        return starts

    def _path(self, th, r, alpha, rw):
        v0 = np.quantile(rw[:300], alpha)
        return caviar_path(r, th[0], th[1], th[2], th[3], th[4], v0)


class FZGas1F(_PerAlphaFZ):
    """theta = (beta, gamma, log(-a), log(b/a - 1))."""

    name = "FZ-1F"
    scale = np.array([0.03, 0.005, 0.3, 0.3])

    def _valid(self, th):
        return 0.0 <= th[0] < 0.9999 and th[1] >= 0.0

    def _starts(self, rw, alpha):
        v_hat, e_hat = empirical_var_es(rw, alpha)
        pa, pb = np.log(-v_hat), np.log(max(e_hat / v_hat - 1.0, 1e-3))
        return [np.array([b, g, pa, pb]) for b in (0.8, 0.9, 0.95, 0.99)
                for g in (0.001, 0.005, 0.02)]

    def _path(self, th, r, alpha, rw):
        a = -np.exp(th[2])
        b = a * (1.0 + np.exp(th[3]))
        return gas_path(r, th[0], th[1], 0.0, a, b, alpha, 0.0)


class FZHybrid(_PerAlphaFZ):
    """theta = (beta, gamma, delta, log(-a), log(b/a - 1)); kappa_0 = stationary mean."""

    name = "FZ-Hyb"
    scale = np.array([0.03, 0.005, 0.02, 0.3, 0.3])

    def _valid(self, th):
        return 0.0 <= th[0] < 0.9999 and th[1] >= 0.0 and th[2] >= 0.0

    @staticmethod
    def _kappa0(th, rw):
        m = np.mean(np.log(np.maximum(np.abs(rw), ABS_FLOOR)))
        return float(np.clip(th[2] * m / (1.0 - th[0]), -KAPPA_MAX, KAPPA_MAX))

    def _starts(self, rw, alpha):
        v_hat, e_hat = empirical_var_es(rw, alpha)
        pb = np.log(max(e_hat / v_hat - 1.0, 1e-3))
        starts = []
        for beta in (0.8, 0.9, 0.95):
            for gamma in (0.001, 0.005):
                for delta in (0.02, 0.05, 0.1):
                    k0 = self._kappa0(np.array([beta, gamma, delta]), rw)
                    starts.append(np.array([beta, gamma, delta, np.log(-v_hat) - k0, pb]))
        return starts

    def _path(self, th, r, alpha, rw):
        a = -np.exp(th[3])
        b = a * (1.0 + np.exp(th[4]))
        return gas_path(r, th[0], th[1], th[2], a, b, alpha, self._kappa0(th, rw))
