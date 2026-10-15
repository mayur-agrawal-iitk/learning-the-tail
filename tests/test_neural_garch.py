"""Tests for the neural news-impact GARCH: nesting, filter stability, causality, constraints."""

import numpy as np
import torch

from tailrisk.models.garch import gjr_filter
from tailrisk.neural_garch import NeuralGarch

ALPHAS = [0.01, 0.025, 0.05]
GJR = {"omega": 0.02, "alpha": 0.03, "gamma": 0.12, "beta": 0.88}


def returns(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_t(5, n).astype(np.float32)


def test_nests_gjr_garch():
    """With zero gates and d = 1, the variance path is exactly the GJR-GARCH(1,1) recursion."""
    model = NeuralGarch(ALPHAS, GJR, d=1)
    y = returns()
    with torch.no_grad():
        _, _, sigma = model(torch.from_numpy(y)[None])
    s2 = gjr_filter(y.astype(float), 0.0, GJR["omega"], GJR["alpha"], GJR["gamma"], GJR["beta"],
                    float(model.h0[0]))[:-1]
    np.testing.assert_allclose(sigma[0].numpy() ** 2, s2, rtol=2e-3)


def test_filter_forgets_initial_state():
    """|h_t - h'_t| <= rho^t |h_0 - h'_0| for any input path (here: random weights, gates on)."""
    torch.manual_seed(0)
    model = NeuralGarch(ALPHAS, GJR, d=8)
    with torch.no_grad():
        model.g_nic.fill_(0.7)
        for p in model.nic.parameters():
            p.add_(torch.randn_like(p))
    model = model.double()                          # float64: the bound is checked to 1e-9
    y = torch.from_numpy(returns(3000)).double()[None]  # slowest memory decays at 0.995/day
    h0a, h0b = torch.zeros(8, dtype=torch.float64), torch.full((8,), 50.0, dtype=torch.float64)
    with torch.no_grad():
        Ha, Hb = model.states(y, h0a)[0], model.states(y, h0b)[0]
    gap = (Ha - Hb).abs().max(1).values.numpy()
    a_max = float(model.decay().max().detach())
    bound = a_max ** np.arange(len(gap)) * 50.0
    assert np.all(gap <= bound * (1 + 1e-9) + 1e-12)
    assert gap[-1] < 1e-2 * gap[0]


def test_causal():
    """Forecasts for day t do not change when y_t, y_{t+1}, ... are altered."""
    torch.manual_seed(1)
    model = NeuralGarch(ALPHAS, GJR, d=8)
    with torch.no_grad():
        model.g_nic.fill_(0.5)
        model.g_read.fill_(0.5)
    y = torch.from_numpy(returns(400))[None]
    y2 = y.clone()
    y2[:, 250:] = y2[:, 250:] * 3 - 1
    with torch.no_grad():
        v1, e1, _ = model(y)
        v2, e2, _ = model(y2)
    torch.testing.assert_close(v1[:, :251], v2[:, :251])
    assert not torch.allclose(v1[:, 252:], v2[:, 252:])


def test_outputs_ordered_and_decay_bounded():
    torch.manual_seed(2)
    model = NeuralGarch(ALPHAS, GJR, d=8, rho=0.999)
    with torch.no_grad():
        model.a0_raw.add_(10.0)                     # push decays to the boundary
        model.am_raw.add_(10.0)
        v, e, _ = model(torch.from_numpy(returns(300))[None])
    assert float(model.decay().max()) < 0.999
    assert torch.all(e <= v) and torch.all(v < 0)
    assert torch.all(v[..., :-1] <= v[..., 1:])


def test_asset_specific_block_matches_each_assets_gjr():
    """v2: with zero gates and d = 1, each asset follows its own GJR-GARCH(1,1) recursion."""
    g2 = {"omega": 0.05, "alpha": 0.08, "gamma": 0.02, "beta": 0.85}
    model = NeuralGarch(ALPHAS, [GJR, g2], d=1)
    y = np.stack([returns(seed=1), returns(seed=2)])
    with torch.no_grad():
        _, _, sigma = model(torch.from_numpy(y))
    for i, g in enumerate([GJR, g2]):
        s2 = gjr_filter(y[i].astype(float), 0.0, g["omega"], g["alpha"], g["gamma"], g["beta"],
                        float(model.h0[i, 0]))[:-1]
        np.testing.assert_allclose(sigma[i].numpy() ** 2, s2, rtol=2e-3)
