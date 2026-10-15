"""Tests for the forecasting models: closed forms, alignment with arch, and no look-ahead."""

import copy

import numpy as np
import pytest
from arch import arch_model
from arch.univariate import Normal, StudentsT
from scipy import stats

from tailrisk.models import CaviarES, FZGas1F, FZHybrid, HistoricalSimulation, ArchFamily
from tailrisk.models.garch import gjr_filter, gpd_var_es
from tailrisk.scoring import fz0_loss

ALPHAS = [0.01, 0.025, 0.05]


def simulate_gjr(n, seed=0, mu=0.05, omega=0.02, a=0.03, g=0.1, b=0.9, nu=6.0):
    rng = np.random.default_rng(seed)
    z = rng.standard_t(nu, n) / np.sqrt(nu / (nu - 2))
    r = np.empty(n)
    s2 = omega / (1 - a - g / 2 - b)
    for t in range(n):
        r[t] = mu + np.sqrt(s2) * z[t]
        eps = r[t] - mu
        s2 = omega + (a + g * (eps < 0)) * eps ** 2 + b * s2
    return r


def test_partial_moment_es_matches_closed_form():
    for q in (-2.3, -1.6):
        np.testing.assert_allclose(Normal().partial_moment(1, q, None), -stats.norm.pdf(q), rtol=1e-6)
    nu = 5.0
    a = 0.025
    q = StudentsT().ppf(a, np.array([nu]))
    s = np.sqrt((nu - 2) / nu)  # arch's t has unit variance
    x = q / s
    closed = -s * (nu + x ** 2) / (nu - 1) * stats.t.pdf(x, nu) / a
    np.testing.assert_allclose(StudentsT().partial_moment(1, q, np.array([nu])) / a, closed, rtol=1e-5)


def test_arch_fixed_window_forecast_alignment():
    """arch's forecast(start=W-1) must equal our own GJR recursion with the fitted parameters."""
    r = simulate_gjr(1600, seed=4)
    W, n = 1500, 100
    am = arch_model(r, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t", rescale=False)
    res = am.fit(last_obs=W, disp="off")
    fc = res.forecast(horizon=1, start=W - 1, reindex=False).variance.to_numpy()[:n, 0]
    p = res.params
    # arch only uses r[:W] for estimation
    res_short = arch_model(r[:W], mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t",
                           rescale=False).fit(disp="off")
    np.testing.assert_allclose(res.params, res_short.params, rtol=1e-4)
    s2_in = res.conditional_volatility[:W] ** 2
    own = gjr_filter(r[W - 1:W + n - 1], p["mu"], p["omega"], p["alpha[1]"], p["gamma[1]"],
                     p["beta[1]"], s2_in[W - 1])[1:]
    np.testing.assert_allclose(fc, own, rtol=1e-6)


def test_gpd_tail_on_exact_pareto():
    """For exact Pareto losses with tail index 1/xi the POT estimator recovers VaR."""
    rng = np.random.default_rng(5)
    xi = 0.25
    x = (rng.uniform(size=200_000) ** (-xi) - 1) / xi  # GPD(xi, 1) from threshold 0
    var, es = gpd_var_es(x, [0.01], tail_frac=0.1)
    true_var = (0.01 ** (-xi) - 1) / xi
    true_es = true_var / (1 - xi) + 1 / (1 - xi)
    np.testing.assert_allclose(var[0], true_var, rtol=0.03)
    np.testing.assert_allclose(es[0], true_es, rtol=0.03)


@pytest.mark.parametrize("est", [HistoricalSimulation(250), ArchFamily("GJR-skt", o=1, dist="skewt"),
                                 ArchFamily("GJR-QML", o=1, dist="normal", report_parametric=False,
                                            fhs=True, evt_tail_frac=0.1),
                                 CaviarES(), FZGas1F(), FZHybrid()], ids=lambda e: type(e).__name__)
def test_no_lookahead_and_ordering(est):
    """Forecast for day t must not change when returns from day t onwards are altered."""
    r = simulate_gjr(1200, seed=6)
    W, t0, t1 = 1000, 1000, 1030
    est2 = copy.deepcopy(est)          # identical state (incl. RNG) before any fitting
    out = est.fit_predict(r, t0, t1, W, ALPHAS)
    r2 = r.copy()
    r2[1015:] = r2[1015:] * 5 - 3
    out2 = est2.fit_predict(r2, t0, t1, W, ALPHAS)
    for m, (v, e) in out.items():
        v2, e2 = out2[m]
        np.testing.assert_allclose(v[:16], v2[:16], rtol=1e-8)   # t <= 1015 uses r[:1015]
        assert not np.allclose(v[16:], v2[16:])                  # later forecasts react
        assert np.all(e <= v + 1e-9)


def test_fz_gas_beats_constant_on_garch_data():
    """In-sample, the GAS model should fit GARCH data better than a constant (VaR, ES)."""
    r = simulate_gjr(3000, seed=7)
    a = 0.025
    est = FZGas1F()
    (v, e), = est.fit_predict(r, 2500, 3000, 2500, [a]).values()
    q = np.quantile(r[:2500], a)
    ec = r[:2500][r[:2500] <= q].mean()
    oos = r[2500:3000]
    assert fz0_loss(oos, v[:, 0], e[:, 0], a).mean() < fz0_loss(oos, q, ec, a).mean()
