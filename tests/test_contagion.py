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


def test_scale_head_and_news_switch():
    torch.manual_seed(3)
    seq, static, news, avail, tgt = _inputs()
    net = ContagionNet(5, 2, 1, ALPHAS, NC, scale_head=True).eval()
    with torch.no_grad():
        v, e = net(seq, static, news, avail, tgt)
    assert net.last_sigma is not None and torch.all(net.last_sigma > 0)
    assert torch.all(e <= v) and torch.all(v < 0)
    off = ContagionNet(5, 2, 1, ALPHAS, NC, scale_head=True, use_news=False).eval()
    with torch.no_grad():
        a, _ = off(seq, static, news, avail, tgt)
        b, _ = off(seq, static, news * 0, avail * 0, tgt)
    torch.testing.assert_close(a, b)


def test_job_applies_the_requested_variant(tmp_path, monkeypatch):
    """End-to-end on synthetic data: forecasts carry the variant's name and its settings are used."""
    import numpy as np
    import pandas as pd
    import tailrisk.contagion as ctg
    from tailrisk.crossmarket import overnight_news

    dates = pd.bdate_range("2010-01-01", periods=1400)
    rng = np.random.default_rng(0)
    rets = {m: pd.Series(rng.standard_t(5, len(dates)), index=dates) for m in ("SPX", "N225")}
    cfg = {"name": "unit", "window": 800, "alphas": ALPHAS,
           "neural": {"lookback": 16, "scale_window": 100, "hidden": 8, "layers": 1, "dropout": 0.0,
                      "lr": 1e-3, "weight_decay": 0.0, "batch_size": 256, "max_epochs": 1,
                      "patience": 1, "val_frac": 0.2, "gpu_jobs": 0},
           "contagion": {"LSTMQ-FZ": {"scale_head": True, "aux_qlike": 1.0, "use_news": False}}}
    monkeypatch.setattr(ctg, "_gpu_lock_dir", lambda cfg: tmp_path)
    df, info = ctg._job(dates[1000], dates[1100], 0, rets, overnight_news(rets, 100), cfg, "cpu",
                        tmp_path / "w.pt", "LSTMQ-FZ")
    assert set(df["model"]) == {"LSTMQ-FZ"} and info["model"] == "LSTMQ-FZ"
    state = torch.load(tmp_path / "w.pt")["state"]
    assert any(k.startswith("var_head") for k in state)          # scale head was built
    import pytest
    with pytest.raises(KeyError):
        ctg._job(dates[1000], dates[1100], 0, rets, overnight_news(rets, 100), cfg, "cpu", None, "TYPO")
