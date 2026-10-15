r"""GJR-GARCH-X with skewed-t innovations: the econometric benchmark with cross-market information.

    r_t = mu + sigma_t z_t,   z_t ~ Hansen skewed-t (unit variance)
    sigma_t^2 = omega + (a + g 1{eps_{t-1} < 0}) eps_{t-1}^2 + b sigma_{t-1}^2 + d1 X1_t + d2 X2_t

with X1_t = mean of x_j^2 and X2_t = mean of x_j^2 1{x_j < 0} over the foreign markets j with fresh
overnight news x_j before market i opens on day t (crossmarket.overnight_news). X_t is known
before r_t is realised, so it enters sigma_t^2 contemporaneously. d1, d2 >= 0 keep the variance
positive. Estimated by maximum likelihood on the rolling window; the parameters of GJR-skt (arch)
plus d = 0 serve as starting values, warm-started across refits.
"""

from __future__ import annotations

import warnings

import numpy as np
from arch import arch_model
from arch.univariate import SkewStudent
from numba import njit
from scipy.optimize import minimize

from .base import Estimator, Forecasts

_DIST = SkewStudent()
BOUNDS = [(-1.0, 1.0), (1e-6, 10.0), (0.0, 0.5), (0.0, 0.6), (0.0, 0.999), (0.0, 5.0), (0.0, 5.0),
          (2.1, 200.0), (-0.99, 0.99)]


@njit(cache=True)
def gjrx_variance(eps, x1, x2, omega, a, g, b, d1, d2, s2_0):
    """sigma2[t] for t = 0..n-1 given eps[0..n-2] and the exogenous X[0..n-1]."""
    n = x1.size
    s2 = np.empty(n)
    s2[0] = s2_0
    for t in range(1, n):
        e = eps[t - 1]
        s2[t] = omega + (a + (g if e < 0 else 0.0)) * e * e + b * s2[t - 1] + d1 * x1[t] + d2 * x2[t]
    return s2


def news_regressors(news: np.ndarray, avail: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(X1, X2) per day from the (days x markets) overnight-news and availability arrays."""
    k = np.maximum(avail.sum(1), 1.0)
    sq = news ** 2 * avail
    return sq.sum(1) / k, (sq * (news < 0)).sum(1) / k


class GJRXSkt(Estimator):
    name = "GJRX-skt"

    def __init__(self, x1: np.ndarray, x2: np.ndarray):
        self.x1, self.x2 = np.asarray(x1, float), np.asarray(x2, float)
        self.names = [self.name]
        self.prev: np.ndarray | None = None

    def _nll(self, p, r, x1, x2):
        mu, omega, a, g, b, d1, d2, eta, lam = p
        if a + g / 2 + b >= 0.9999:
            return 1e10
        eps = r - mu
        s2 = gjrx_variance(eps, x1, x2, omega, a, g, b, d1, d2, float(np.var(r)))
        if not np.all(np.isfinite(s2)) or np.any(s2 <= 0):
            return 1e10
        return -float(_DIST.loglikelihood(np.array([eta, lam]), eps, s2, individual=False))

    def _start(self, r):
        am = arch_model(r, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="skewt", rescale=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = am.fit(disp="off", options={"maxiter": 500}).params.to_numpy()
        mu, omega, a, g, b, eta, lam = p
        return np.clip(np.array([mu, omega, a, g, b, 0.0, 0.0, eta, lam]),
                       [lo for lo, _ in BOUNDS], [hi for _, hi in BOUNDS])

    def fit_predict(self, r, t0, t1, window, alphas) -> Forecasts:
        rw = r[t0 - window:t0]
        xw1, xw2 = self.x1[t0 - window:t0], self.x2[t0 - window:t0]
        starts = [self._start(rw)] + ([self.prev] if self.prev is not None else [])
        best = None
        for s in starts:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = minimize(self._nll, s, args=(rw, xw1, xw2), method="L-BFGS-B", bounds=BOUNDS)
            if best is None or res.fun < best.fun:
                best = res
        p = best.x
        self.prev = p
        mu, omega, a, g, b, d1, d2, eta, lam = p
        # filter through window + block with frozen parameters; X_t of block days is known pre-open
        seg = r[t0 - window:t1]
        s2 = gjrx_variance(seg - mu, self.x1[t0 - window:t1], self.x2[t0 - window:t1],
                           omega, a, g, b, d1, d2, float(np.var(rw)))
        sigma = np.sqrt(s2[window:window + (t1 - t0)])[:, None]
        alphas = np.asarray(alphas, dtype=float)
        dp = np.array([eta, lam])
        q = np.asarray(_DIST.ppf(alphas, dp), dtype=float)
        es_z = np.array([_DIST.partial_moment(1, qi, dp) for qi in q]) / alphas
        return {self.name: (mu + sigma * q, mu + sigma * es_z)}
