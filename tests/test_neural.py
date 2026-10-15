"""Tests for the deep models: loss implementation, output constraints, causal features."""

import numpy as np
import pytest
import torch

from tailrisk.neural import MonotoneTailHead, asset_features, fz0_torch
from tailrisk.scoring import fz0_loss

ALPHAS = [0.01, 0.025, 0.05]
CFG = {"window": 600, "neural": {"lookback": 16, "scale_window": 50}}


def test_fz0_torch_matches_numpy():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(500)
    v = -np.abs(rng.normal(2, 0.3, (500, 3)))
    e = v - np.abs(rng.normal(0.5, 0.1, (500, 3)))
    want = sum(fz0_loss(y, v[:, k], e[:, k], a) for k, a in enumerate(ALPHAS)).mean()
    got = fz0_torch(torch.tensor(y), torch.tensor(v), torch.tensor(e), torch.tensor(ALPHAS)).item()
    np.testing.assert_allclose(got, want, rtol=1e-6)   # alphas are float32 in torch


def test_monotone_head_never_crosses():
    torch.manual_seed(0)
    head = MonotoneTailHead(8, 16, ALPHAS, dropout=0.0)
    with torch.no_grad():
        for p in head.parameters():
            p.add_(torch.randn_like(p) * 3)            # arbitrary weights
        v, e = head(torch.randn(10_000, 8) * 5)
    assert torch.all(v < 0)
    assert torch.all(e <= v)
    assert torch.all(v[:, :-1] <= v[:, 1:])          # VaR_1% <= VaR_2.5% <= VaR_5%
    assert torch.all(e[:, :-1] <= e[:, 1:])          # same for ES


def test_head_initialised_at_gaussian():
    head = MonotoneTailHead(8, 16, [0.025], dropout=0.0)
    head.mlp[-1].weight.data.zero_()
    v, e = head(torch.zeros(1, 8))
    np.testing.assert_allclose([v.item(), e.item()], [-1.95996, -2.33780], atol=1e-4)


def test_head_initialised_at_empirical_tail():
    """With zero last-layer weights the head outputs exactly the supplied (VaR, ES) per alpha."""
    v0 = np.array([-2.9, -2.2, -1.7])
    e0 = np.array([-4.1, -3.2, -2.6])
    head = MonotoneTailHead(8, 16, ALPHAS, dropout=0.0, init=(v0, e0))
    head.mlp[-1].weight.data.zero_()
    v, e = head(torch.zeros(1, 8))
    np.testing.assert_allclose(v.detach().numpy()[0], v0, atol=1e-4)
    np.testing.assert_allclose(e.detach().numpy()[0], e0, atol=1e-4)


@pytest.mark.parametrize("model", ["LSTM-FZ", "GJR-LSTM"])
def test_features_are_causal(model):
    """Inputs and scale for target t must not change when r[t:] is altered."""
    rng = np.random.default_rng(1)
    r = rng.standard_t(5, 1000) * 1.2
    idx0, idx_end, k = 800, 900, 850
    f1 = asset_features(model, r, idx0, idx_end, CFG)
    r2 = r.copy()
    r2[k:] = r2[k:] * 4 + 1
    f2 = asset_features(model, r2, idx0, idx_end, CFG)
    n_tr = f1["n_tr"]
    upto = n_tr + (k - idx0) + 1                      # rows with target t <= k
    for key in ("seq", "static", "scale"):
        np.testing.assert_allclose(f1[key][:upto], f2[key][:upto], rtol=1e-6)
    assert not np.allclose(f1["seq"][upto:], f2["seq"][upto:])
