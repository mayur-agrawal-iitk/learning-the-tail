r"""GARCH-family models (via ``arch``), filtered historical simulation and GARCH-EVT.

Location-scale structure:  r_t = mu + sigma_t z_t,  z_t iid with unit variance, so
    VaR_t = mu + sigma_t q_alpha(z),   ES_t = mu + sigma_t ES_alpha(z).
The three models differ only in how the standardised tail (q_alpha, ES_alpha) is obtained:
  * parametric  : quantile and ES of the fitted innovation distribution (Normal, t, skew-t);
                  ES_alpha(z) = (1/alpha) int_{-inf}^{q_alpha} x f(x) dx (partial moment);
  * FHS         : empirical quantile / tail mean of in-sample standardised residuals
                  (Barone-Adesi et al. 1999);
  * GARCH-EVT   : generalised Pareto tail fitted to the largest standardised losses
                  (McNeil & Frey 2000).
"""

from __future__ import annotations

import warnings

import numpy as np
from arch import arch_model
from numba import njit
from scipy import stats

from ..scoring import empirical_var_es
from .base import Estimator, Forecasts


def gpd_var_es(losses: np.ndarray, alphas, tail_frac: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
    r"""Peaks-over-threshold estimates of upper-tail VaR/ES of ``losses`` (positive = loss).

    With threshold u (the (1 - tail_frac) empirical quantile), N_u exceedances out of n
    and fitted GPD(xi, beta) for the excesses:
        VaR_p = u + beta/xi * ((n p / N_u)^(-xi) - 1)
        ES_p  = VaR_p / (1 - xi) + (beta - xi u) / (1 - xi)        (xi < 1)
    Returns positive numbers (loss scale).
    """
    x = np.asarray(losses, dtype=float)
    n = x.size
    u = np.quantile(x, 1.0 - tail_frac)
    exc = x[x > u] - u
    n_u = exc.size
    xi, _, beta = stats.genpareto.fit(exc, floc=0.0)
    xi = min(xi, 0.9)  # guard against infinite-mean fits in tiny samples
    p = np.asarray(alphas, dtype=float)
    ratio = n * p / n_u
    if abs(xi) < 1e-6:
        var = u - beta * np.log(ratio)
        es = var + beta
    else:
        var = u + beta / xi * (ratio ** (-xi) - 1.0)
        es = var / (1.0 - xi) + (beta - xi * u) / (1.0 - xi)
    return var, es


class ArchFamily(Estimator):
    """GARCH(1,1) (o=0) or GJR-GARCH(1,1,1) (o=1) with constant mean, fitted by MLE."""

    def __init__(self, name: str, o: int = 0, dist: str = "normal",
                 report_parametric: bool = True, fhs: bool = False,
                 evt_tail_frac: float | None = None):
        self.name, self.o, self.dist = name, o, dist
        self.report_parametric = report_parametric
        self.fhs = fhs
        self.evt_tail_frac = evt_tail_frac
        self.names = ([name] if report_parametric else []) + (["FHS"] if fhs else []) \
            + (["GARCH-EVT"] if evt_tail_frac else [])
        self.prev_params = None

    def _fit(self, y: np.ndarray, last_obs: int):
        am = arch_model(y, mean="Constant", vol="GARCH", p=1, o=self.o, q=1,
                        dist=self.dist, rescale=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = am.fit(last_obs=last_obs, disp="off", starting_values=self.prev_params,
                         options={"maxiter": 500})
            if res.convergence_flag != 0 and self.prev_params is not None:
                res = am.fit(last_obs=last_obs, disp="off", options={"maxiter": 500})
        self.prev_params = res.params.to_numpy()
        return am, res

    def fit_predict(self, r, t0, t1, window, alphas) -> Forecasts:
        n = t1 - t0
        y = r[t0 - window:t1]  # block returns are appended only to run the filter forward
        am, res = self._fit(y, last_obs=window)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = res.forecast(horizon=1, start=window - 1, reindex=False)
        mu = fc.mean.to_numpy()[:n, 0][:, None]
        sigma = np.sqrt(fc.variance.to_numpy()[:n, 0])[:, None]
        alphas = np.asarray(alphas, dtype=float)
        out: Forecasts = {}

        if self.report_parametric:
            k = am.distribution.num_params
            dp = res.params.to_numpy()[len(res.params) - k:] if k else None
            q = np.asarray(am.distribution.ppf(alphas, dp), dtype=float)
            es_z = np.array([am.distribution.partial_moment(1, qi, dp) for qi in q]) / alphas
            out[self.name] = (mu + sigma * q, mu + sigma * es_z)

        if self.fhs or self.evt_tail_frac:
            z = np.asarray(res.std_resid)[:window]
            z = z[np.isfinite(z)]
            if self.fhs:
                tail = np.array([empirical_var_es(z, a) for a in alphas])
                out["FHS"] = (mu + sigma * tail[:, 0], mu + sigma * tail[:, 1])
            if self.evt_tail_frac:
                var_l, es_l = gpd_var_es(-z, alphas, self.evt_tail_frac)
                out["GARCH-EVT"] = (mu - sigma * var_l, mu - sigma * es_l)
        return out


@njit(cache=True)
def gjr_filter(r, mu, omega, a, g, b, sigma2_0):
    """Conditional variances sigma2[t] = Var(r_t | r_{<t}) for t = 0..n (n+1 values)."""
    n = r.size
    s2 = np.empty(n + 1)
    s2[0] = sigma2_0
    for t in range(n):
        eps = r[t] - mu
        s2[t + 1] = omega + (a + (g if eps < 0 else 0.0)) * eps * eps + b * s2[t]
    return s2


def fit_gjr_t(r_window: np.ndarray) -> dict:
    """Fit GJR-GARCH(1,1,1)-t on a window; returns the parameters used by the hybrid deep model."""
    am = arch_model(r_window, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t", rescale=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = am.fit(disp="off", options={"maxiter": 500})
    p = res.params
    return {"mu": float(p["mu"]), "omega": float(p["omega"]), "alpha": float(p["alpha[1]"]),
            "gamma": float(p["gamma[1]"]), "beta": float(p["beta[1]"]), "nu": float(p["nu"]),
            "var0": float(np.var(r_window))}
