r"""Absolute (calibration) backtests for VaR and ES forecasts.

VaR:  Kupiec (1995) unconditional coverage, Christoffersen (1998) independence and
      conditional coverage, Engle & Manganelli (2004) dynamic quantile (DQ).
ES:   McNeil & Frey (2000) exceedance-residual test (bootstrap, one-sided),
      Acerbi & Szekely (2014) Z2 statistic,
      Nolde & Ziegel (2017) conditional calibration test built on the
      (VaR, ES) identification function.
All inputs are aligned numpy arrays of realised returns r, VaR v and ES e.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.special import xlogy

from .scoring import identification


def _ll_bernoulli(n1: float, n0: float, p: float) -> float:
    return xlogy(n1, p) + xlogy(n0, 1.0 - p)


def kupiec(hits: np.ndarray, alpha: float) -> dict:
    """Proportion-of-failures LR test, H0: P(hit) = alpha. LR ~ chi2(1)."""
    n = hits.size
    x = float(hits.sum())
    pi = x / n
    lr = -2.0 * (_ll_bernoulli(x, n - x, alpha) - _ll_bernoulli(x, n - x, pi))
    return {"rate": pi, "lr_uc": lr, "p_uc": float(stats.chi2.sf(lr, 1))}


def christoffersen(hits: np.ndarray, alpha: float) -> dict:
    """Markov independence test (chi2(1)) and conditional coverage LR_cc = LR_uc + LR_ind (chi2(2))."""
    h = hits.astype(int)
    prev, curr = h[:-1], h[1:]
    n00 = float(np.sum((prev == 0) & (curr == 0)))
    n01 = float(np.sum((prev == 0) & (curr == 1)))
    n10 = float(np.sum((prev == 1) & (curr == 0)))
    n11 = float(np.sum((prev == 1) & (curr == 1)))
    pi01 = n01 / max(n00 + n01, 1.0)
    pi11 = n11 / max(n10 + n11, 1.0)
    pi1 = (n01 + n11) / max(n00 + n01 + n10 + n11, 1.0)
    ll_restricted = _ll_bernoulli(n01 + n11, n00 + n10, pi1)
    ll_markov = _ll_bernoulli(n01, n00, pi01) + _ll_bernoulli(n11, n10, pi11)
    lr_ind = max(-2.0 * (ll_restricted - ll_markov), 0.0)
    lr_uc = kupiec(h[1:], alpha)["lr_uc"]
    lr_cc = lr_uc + lr_ind
    return {
        "lr_ind": lr_ind, "p_ind": float(stats.chi2.sf(lr_ind, 1)),
        "lr_cc": lr_cc, "p_cc": float(stats.chi2.sf(lr_cc, 2)),
    }


def dq_test(hits: np.ndarray, var: np.ndarray, alpha: float, lags: int = 4) -> dict:
    r"""Engle-Manganelli dynamic quantile test.

    Regress Hit_t = 1{r_t <= v_t} - alpha on [1, Hit_{t-1..t-lags}, v_t]; under H0 all
    coefficients are zero and DQ = Hit' X (X'X)^{-1} X' Hit / (alpha(1-alpha)) ~ chi2(lags+2).
    """
    hit = hits.astype(float) - alpha
    n = hit.size
    cols = [np.ones(n - lags)]
    cols += [hit[lags - k:n - k] for k in range(1, lags + 1)]
    cols.append(var[lags:])
    X = np.column_stack(cols)
    y = hit[lags:]
    beta = np.linalg.pinv(X.T @ X) @ (X.T @ y)
    dq = float(beta @ (X.T @ X) @ beta) / (alpha * (1.0 - alpha))
    k = X.shape[1]
    return {"dq": dq, "p_dq": float(stats.chi2.sf(dq, k))}


def exceedance_residual_test(r, v, e, n_boot: int = 5000, seed: int = 0) -> dict:
    r"""McNeil-Frey (2000) style test that ES is not too optimistic.

    On exceedance days the residuals xi_t = (r_t - e_t)/|e_t| have zero mean if ES is
    correct (and VaR correct). H1: mean < 0 (realised tail losses exceed forecast ES).
    p-value from a bootstrap of the centred residuals (Efron & Tibshirani 1993).
    """
    hit = r <= v
    xi = (r[hit] - e[hit]) / np.abs(e[hit])
    m = xi.size
    if m < 3:
        return {"n_exc": m, "er_mean": np.nan, "p_er": np.nan}
    t_obs = xi.mean() / (xi.std(ddof=1) / np.sqrt(m))
    rng = np.random.default_rng(seed)
    centred = xi - xi.mean()
    idx = rng.integers(0, m, size=(n_boot, m))
    samples = centred[idx]
    t_boot = samples.mean(1) / (samples.std(1, ddof=1) / np.sqrt(m) + 1e-12)
    p = float(np.mean(t_boot <= t_obs))
    return {"n_exc": m, "er_mean": float(xi.mean()), "p_er": p}


def acerbi_szekely_z2(r, v, e, alpha: float) -> float:
    r"""Acerbi-Szekely Z2 = 1 - sum_t r_t 1{r_t<=v_t} / (T alpha e_t). E[Z2]=0 under H0; negative => ES too optimistic."""
    hit = (r <= v).astype(float)
    return float(1.0 - np.sum(r * hit / e) / (r.size * alpha))


def conditional_calibration_test(r, v, e, alpha: float, conditional: bool = True) -> dict:
    r"""Nolde & Ziegel (2017) Wald-type calibration test for (VaR, ES).

    Moment conditions: E[h_t (x) V_t] = 0 where V_t is the identification function with
    the ES component rescaled by |e_t| (F_{t-1}-measurable, keeps the test scale free).
    Unconditional version: h_t = 1. Conditional version: h_t = (1, |r_{t-1}|/|e_t|), which
    has power against slow reaction to shocks (a large move relative to the forecast that
    is followed by more violations than predicted).
    T = n m_bar' S^{-1} m_bar ~ chi2(dim), with S the outer-product covariance (the
    moments are a martingale difference sequence under H0, so no HAC correction).

    Note: the lagged hit is deliberately *not* used as an instrument. On non-violation days
    V1 = -alpha and V2 = e - v, so Hit_{t-1} V1 and Hit_{t-1} V2 are collinear except on
    consecutive violations (~alpha^2 n days); S becomes near-singular and the test is
    badly oversized (68% rejection at 5% nominal in our simulations). Clustering is
    covered by the Christoffersen and DQ tests instead.
    """
    V = identification(r, v, e, alpha)
    V[:, 1] /= np.abs(e)
    if conditional:
        h = np.column_stack([np.ones(r.size - 1), np.abs(r[:-1]) / np.abs(e[1:])])
        V = V[1:]
        m = np.concatenate([V * h[:, [j]] for j in range(h.shape[1])], axis=1)
    else:
        m = V
    n, k = m.shape
    mbar = m.mean(0)
    S = (m.T @ m) / n
    stat = float(n * mbar @ np.linalg.pinv(S) @ mbar)
    return {"cct": stat, "p_cct": float(stats.chi2.sf(stat, k)), "df": k}


def backtest_all(r, v, e, alpha: float, n_boot: int = 5000) -> dict:
    """Run every backtest for one (model, asset, alpha) forecast series."""
    r, v, e = (np.asarray(x, dtype=float) for x in (r, v, e))
    hits = (r <= v).astype(int)
    out = {"n": r.size}
    out.update(kupiec(hits, alpha))
    out.update(christoffersen(hits, alpha))
    out.update(dq_test(hits, v, alpha))
    out.update(exceedance_residual_test(r, v, e, n_boot=n_boot))
    out["z2"] = acerbi_szekely_z2(r, v, e, alpha)
    cc_u = conditional_calibration_test(r, v, e, alpha, conditional=False)
    cc_c = conditional_calibration_test(r, v, e, alpha, conditional=True)
    out["p_cct_u"] = cc_u["p_cct"]
    out["p_cct_c"] = cc_c["p_cct"]
    return out
