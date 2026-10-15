"""Tests for the scoring function and backtests (consistency, size under H0, power)."""

import numpy as np
import pytest
from scipy import stats

from tailrisk.backtests import (backtest_all, christoffersen, conditional_calibration_test,
                                dq_test, exceedance_residual_test, kupiec)
from tailrisk.scoring import fz0_loss

ALPHA = 0.025


def normal_var_es(alpha, sigma=1.0):
    q = stats.norm.ppf(alpha)
    return sigma * q, -sigma * stats.norm.pdf(q) / alpha


def test_fz0_minimised_at_truth():
    """Strict consistency: E[L(Y, v, e)] is smallest at the true (VaR, ES)."""
    rng = np.random.default_rng(0)
    y = rng.standard_t(5, size=400_000)
    v0 = np.quantile(y, ALPHA)
    e0 = y[y <= v0].mean()
    base = fz0_loss(y, v0, e0, ALPHA).mean()
    for dv, de in [(0.2, 0), (-0.2, 0), (0, 0.3), (0, -0.3), (0.2, 0.3), (-0.2, -0.3)]:
        assert fz0_loss(y, v0 + dv, e0 + de, ALPHA).mean() > base


def test_fz0_scale_shift():
    """L(cy, cv, ce) = L(y, v, e) + log c, so loss differences are scale free."""
    y, v, e = np.array([-3.0, 0.5, -1.0]), -2.0, -2.6
    np.testing.assert_allclose(fz0_loss(5 * y, 5 * v, 5 * e, ALPHA),
                               fz0_loss(y, v, e, ALPHA) + np.log(5))


@pytest.mark.parametrize("test", ["uc", "cc", "dq", "cct"])
def test_size_under_null(test):
    """Correct forecasts are rejected roughly 5% of the time.

    The calibration test relies on tail moments and is mildly oversized in small samples
    (about 12% at n=2500 and 9% at n=5000 for alpha=2.5%), so its size is checked at n=20000.
    """
    rng = np.random.default_rng(1)
    v, e = normal_var_es(ALPHA)
    n = 20_000 if test == "cct" else 2500
    rej = []
    for _ in range(300):
        r = rng.standard_normal(n)
        hits = (r <= v).astype(int)
        vv = np.full(r.size, v)
        ee = np.full(r.size, e)
        p = {"uc": lambda: kupiec(hits, ALPHA)["p_uc"],
             "cc": lambda: christoffersen(hits, ALPHA)["p_cc"],
             "dq": lambda: dq_test(hits, vv + 1e-3 * rng.standard_normal(r.size), ALPHA)["p_dq"],
             "cct": lambda: conditional_calibration_test(r, vv, ee, ALPHA)["p_cct"]}[test]()
        rej.append(p < 0.05)
    assert 0.01 < np.mean(rej) < 0.11


def test_tests_detect_misspecification():
    """Forecasts from N(0,1) when data are t(3) scaled to unit variance: ES too optimistic."""
    rng = np.random.default_rng(2)
    r = rng.standard_t(3, size=5000) / np.sqrt(3.0)
    v, e = normal_var_es(ALPHA)
    out = backtest_all(r, np.full(r.size, v), np.full(r.size, e), ALPHA, n_boot=2000)
    assert out["p_er"] < 0.05          # tail losses beyond forecast ES
    assert out["z2"] < 0               # Acerbi-Szekely sign convention
    assert out["p_cct_u"] < 0.05


def test_exceedance_residual_null():
    rng = np.random.default_rng(3)
    r = rng.standard_normal(20_000)
    v, e = normal_var_es(ALPHA)
    assert exceedance_residual_test(r, np.full(r.size, v), np.full(r.size, e))["p_er"] > 0.01
