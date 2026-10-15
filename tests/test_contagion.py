"""Tests for the CTG-FZ contagion network."""

import torch

from tailrisk.contagion import ContagionNet

ALPHAS = [0.01, 0.025, 0.05]
NC = {"hidden": 16, "layers": 1, "dropout": 0.0}


def _inputs(B=64, L=32, M=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    seq = torch.randn(B, L, 2, generator=g)
    static = torch.randn(B, 1, generator=g)
    news = torch.randn(B, M, generator=g) * 2
    avail = (torch.rand(B, M, generator=g) > 0.5).float()
    tgt = torch.randint(0, M, (B,), generator=g)
    return seq, static, news, avail, tgt


def test_unavailable_news_cannot_leak():
    torch.manual_seed(0)
    net = ContagionNet(5, 2, 1, ALPHAS, NC).eval()
    seq, static, news, avail, tgt = _inputs()
    with torch.no_grad():
        v1, e1 = net(seq, static, news, avail, tgt)
        news2 = news.clone()
        news2[avail == 0] = 999.0                    # garbage in masked slots
        v2, e2 = net(seq, static, news2, avail, tgt)
    torch.testing.assert_close(v1, v2)
    torch.testing.assert_close(e1, e2)


def test_no_news_means_no_contagion_signal():
    torch.manual_seed(1)
    net = ContagionNet(5, 2, 1, ALPHAS, NC).eval()
    seq, static, news, _, tgt = _inputs()
    none = torch.zeros_like(news)
    with torch.no_grad():
        va, _ = net(seq, static, news, none, tgt)
        vb, _ = net(seq, static, news * 0 + 5, none, tgt)
    torch.testing.assert_close(va, vb)


def test_attention_weights_are_a_distribution_over_available_markets():
    torch.manual_seed(2)
    net = ContagionNet(5, 2, 1, ALPHAS, NC).eval()
    seq, static, news, avail, tgt = _inputs()
    with torch.no_grad():
        v, e = net(seq, static, news, avail, tgt)
    w = net.last_weights
    has = avail.sum(1) > 0
    torch.testing.assert_close(w[has].sum(1), torch.ones(int(has.sum())))
    assert torch.all(w[avail == 0] == 0)
    assert torch.all(e <= v) and torch.all(v < 0)
