r"""CTG-FZ: an LSTM tail-risk model with a learned cross-market contagion module.

CTG-FZ = LSTM-FZ (same inputs, same hyperparameters) + contagion attention:

    tokens   k_j = MLP( x_j, |x_j|, emb(j) )                    for each foreign market j
    query    q   = W [ h_LSTM, emb(i) ]                          target market i, current state
    weights  w_j = softmax_j( <W_k k_j, q> / sqrt(H) ) over markets with fresh overnight news
    signal   c   = sum_j w_j k_j          (0 if no foreign session closed overnight)
    (VaR, ES)    = s_t * Head( h_LSTM, c, log s_t )             monotone, non-crossing

x_j is j's overnight move in units of its own volatility (crossmarket.overnight_news). Because the
tokens are non-linear in x_j and the query depends on the target and on its own volatility state,
the model can learn that large foreign crashes matter disproportionately, that some markets lead
others, and that contagion is stronger in turbulent states. The attention weights w_j form an
interpretable, time-varying "who transmits risk to whom" matrix.

Training, validation split, refit schedule, seeds and checkpointing follow neural.py exactly.
"""

from __future__ import annotations

import copy
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from joblib import Parallel, delayed

from .config import results_dir
from .crossmarket import overnight_news
from .data import load_returns
from .engine import to_long
from .neural import (INPUT_CLIP, LSTMEncoder, MonotoneTailHead, _claim_device, _gpu_lock_dir,
                     asset_features, fz0_torch, refit_dates, resolve_device)
from .scoring import empirical_var_es

NAME = "CTG-FZ"
INPUTS = ("seq", "static", "news", "avail", "tgt")


class ContagionNet(nn.Module):
    def __init__(self, n_markets: int, n_seq: int, n_static: int, alphas: list[float], nc: dict,
                 emb_dim: int = 8, init=None, scale_head: bool = False, use_news: bool = True):
        super().__init__()
        H = nc["hidden"]
        self.H = H
        self.use_news = use_news
        # v2: location-scale output; sigma_t (in units of the slow scale) from a variance head
        self.var_head = None
        if scale_head:
            self.var_head = nn.Sequential(nn.Linear(2 * H + n_static, H), nn.Tanh(), nn.Linear(H, 1))
            with torch.no_grad():
                self.var_head[-1].weight.mul_(0.1)
                self.var_head[-1].bias.zero_()
        self.last_sigma: torch.Tensor | None = None
        self.enc = LSTMEncoder(n_seq, H, nc["layers"], nc["dropout"])
        self.emb = nn.Embedding(n_markets, emb_dim)
        self.token = nn.Sequential(nn.Linear(2 + emb_dim, H), nn.Tanh(), nn.Linear(H, H))
        self.query = nn.Linear(H + emb_dim, H)
        self.key = nn.Linear(H, H, bias=False)
        self.head = MonotoneTailHead(2 * H + n_static, H, alphas, nc["dropout"], init)
        self.last_weights: torch.Tensor | None = None

    def forward(self, seq, static, news, avail, tgt):
        h = self.enc(seq)                                           # (B, H)
        B, M = news.shape
        src = self.emb.weight[None].expand(B, -1, -1)               # (B, M, E)
        tok = self.token(torch.cat([news[..., None], news.abs()[..., None], src], -1))   # (B, M, H)
        q = self.query(torch.cat([h, self.emb(tgt)], -1))           # (B, H)
        scores = (self.key(tok) @ q[..., None]).squeeze(-1) / math.sqrt(self.H)
        scores = scores.masked_fill(avail < 0.5, -1e9)
        w = torch.softmax(scores, -1) * avail                       # exact zeros where unavailable
        w = w / w.sum(-1, keepdim=True).clamp_min(1e-9)
        any_news = (avail.sum(-1, keepdim=True) > 0).to(h.dtype)
        c = (w[..., None] * tok).sum(1) * any_news                  # (B, H)
        if not self.use_news:                                       # ablation: no foreign information
            c = torch.zeros_like(c)
        self.last_weights = w.detach()
        z = torch.cat([h, c, static], -1)
        v, e = self.head(z)
        if self.var_head is None:
            self.last_sigma = None
            return v, e
        sigma = torch.exp(0.5 * self.var_head(z).squeeze(-1))
        self.last_sigma = sigma
        return sigma[:, None] * v, sigma[:, None] * e


# ----------------------------------------------------------------------------- data
def target_features(r: np.ndarray, news: np.ndarray, avail: np.ndarray, idx0: int, idx_end: int,
                    cfg: dict, target: int) -> dict:
    f = asset_features("LSTM-FZ", r, idx0, idx_end, cfg)     # identical own-history inputs
    first = idx0 - f["n_tr"]
    all_t = np.concatenate([np.arange(first, idx0), f["te_idx"]])
    f["news"] = np.clip(np.nan_to_num(news[all_t]), -INPUT_CLIP, INPUT_CLIP).astype(np.float32)
    f["avail"] = avail[all_t].astype(np.float32)
    f["tgt"] = np.full(all_t.size, target, dtype=np.int64)
    return f


# ----------------------------------------------------------------------------- training
def _batches(n, bs, rng):
    idx = rng.permutation(n) if rng is not None else np.arange(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


def train_ctg(train: dict, val: dict, n_markets: int, alphas: list[float], nc: dict, seed: int,
              device: str = "cpu", variant: dict | None = None):
    variant = variant or {}
    lam = float(variant.get("aux_qlike", 0.0))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device(device)
    tails = np.array([empirical_var_es(train["y"].astype(float), a) for a in alphas])
    net = ContagionNet(n_markets, train["seq"].shape[2], train["static"].shape[1], alphas, nc,
                       init=(tails[:, 0], tails[:, 1]), scale_head=bool(variant.get("scale_head", False)),
                       use_news=bool(variant.get("use_news", True))).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=nc["lr"], weight_decay=nc["weight_decay"])
    a = torch.tensor(alphas, dtype=torch.float32, device=dev)
    T = {k: torch.from_numpy(v).to(dev) for k, v in train.items()}
    V = {k: torch.from_numpy(v).to(dev) for k, v in val.items()}

    def val_loss() -> float:
        net.eval()
        tot = 0.0
        with torch.no_grad():
            for b in _batches(V["y"].shape[0], 1024, None):
                b = torch.from_numpy(b).to(dev)
                v, e = net(*(V[k][b] for k in INPUTS))
                tot += fz0_torch(V["y"][b], v, e, a).item() * b.numel()
        return tot / V["y"].shape[0]

    best, best_state, wait, epoch = val_loss(), copy.deepcopy(net.state_dict()), 0, 0
    for epoch in range(1, nc["max_epochs"] + 1):
        net.train()
        for b in _batches(T["y"].shape[0], nc["batch_size"], rng):
            b = torch.from_numpy(b).to(dev)
            v, e = net(*(T[k][b] for k in INPUTS))
            loss = fz0_torch(T["y"][b], v, e, a)
            if lam > 0 and net.last_sigma is not None:
                # QLIKE for E[y^2 | info]: learns the volatility channel from every day
                s2 = net.last_sigma ** 2
                loss = loss + lam * (T["y"][b] ** 2 / s2 + torch.log(s2)).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        cur = val_loss()
        if cur < best - 1e-5:
            best, best_state, wait = cur, copy.deepcopy(net.state_dict()), 0
        else:
            wait += 1
            if wait >= nc["patience"]:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net, {"epochs": epoch, "best_val": best}


def _job(d0, d1, seed, rets, news, cfg, mode, save_path: Path | None = None, name: str = NAME):
    torch.set_num_threads(1)
    tic = time.time()
    nc, alphas = cfg["neural"], cfg["alphas"]
    device = _claim_device(mode, nc.get("gpu_jobs", 1), _gpu_lock_dir(cfg))
    markets = list(rets)
    feats = {}
    for i, (asset, rs) in enumerate(rets.items()):
        idx0 = int(np.searchsorted(rs.index, d0))
        idx_end = int(np.searchsorted(rs.index, d1)) if d1 is not None else rs.size
        if idx0 >= rs.size or idx_end <= idx0:
            continue
        nw, av = news[asset]
        feats[asset] = target_features(rs.to_numpy(dtype=float), nw[markets].to_numpy(),
                                       av[markets].to_numpy(), idx0, idx_end, cfg, i)
    parts = {"train": [], "val": []}
    for f in feats.values():
        cut = int(f["n_tr"] * (1 - nc["val_frac"]))
        for part, sl in (("train", slice(0, cut)), ("val", slice(cut, f["n_tr"]))):
            parts[part].append({k: f[k][sl] for k in (*INPUTS, "y")})
    stack = {part: {k: np.concatenate([p[k] for p in ps]) for k in (*INPUTS, "y")} for part, ps in parts.items()}
    variants = cfg.get("contagion", {})
    if variants and name not in variants:
        raise KeyError(f"model {name!r} not defined in the config's contagion section")
    variant = variants.get(name, {})
    net, info = train_ctg(stack["train"], stack["val"], len(markets), alphas, nc, seed, device, variant)
    if save_path is not None:
        torch.save({"state": net.state_dict(), "markets": markets}, save_path)
    frames = []
    with torch.no_grad():
        for asset, f in feats.items():
            n_tr = f["n_tr"]
            v, e = net(*(torch.from_numpy(f[k][n_tr:]).to(device) for k in INPUTS))
            loc, scale = f["loc"][n_tr:, None], f["scale"][n_tr:, None]
            df = to_long(name, asset, rets[asset].index[f["te_idx"]], alphas,
                         loc + scale * v.cpu().numpy().astype(float),
                         loc + scale * e.cpu().numpy().astype(float))
            df["seed"] = seed
            frames.append(df)
    info.update({"model": name, "refit": d0.date(), "seed": seed, "device": device,
                 "n_train": int(stack["train"]["y"].size), "seconds": round(time.time() - tic, 1)})
    return pd.concat(frames, ignore_index=True), info


def _job_path(cfg, d0, seed, name: str = NAME) -> Path:
    return results_dir(cfg) / "ctg_jobs" / f"{name}_{d0.date()}_s{seed}.parquet"


def _checkpointed(d0, d1, seed, rets, news, cfg, mode, name: str = NAME) -> dict:
    path = _job_path(cfg, d0, seed, name)
    df, info = _job(d0, d1, seed, rets, news, cfg, mode, path.with_suffix(".pt"), name)
    df.to_parquet(path.with_suffix(".tmp"), index=False)
    path.with_suffix(".tmp").replace(path)
    pd.DataFrame([info]).to_csv(path.with_suffix(".csv"), index=False)
    return info


def run_contagion(cfg: dict, name: str = NAME) -> None:
    nc = cfg["neural"]
    rets = load_returns(cfg)
    news = overnight_news(rets)
    dates = refit_dates(cfg)
    ends = dates[1:] + [None]
    all_jobs = [(d0, d1, s) for d0, d1 in zip(dates, ends) for s in nc["seeds"]]
    (results_dir(cfg) / "ctg_jobs").mkdir(parents=True, exist_ok=True)
    jobs = [j for j in all_jobs if not _job_path(cfg, j[0], j[2], name).exists()]
    mode = resolve_device(nc)
    lock_dir = _gpu_lock_dir(cfg)
    lock_dir.mkdir(parents=True, exist_ok=True)
    for f in lock_dir.glob("slot_*.lock"):
        f.unlink()
    print(f"[ctg] {name}: {len(all_jobs)} jobs, {len(all_jobs) - len(jobs)} already done, mode={mode}, "
          f"{nc['n_jobs']} workers", flush=True)
    if jobs:
        Parallel(n_jobs=nc["n_jobs"], verbose=10)(
            delayed(_checkpointed)(d0, d1, s, rets, news, cfg, mode, name) for d0, d1, s in jobs)
    for f in lock_dir.glob("slot_*.lock"):
        f.unlink()
    rdir = results_dir(cfg)
    paths = [_job_path(cfg, d0, s, name) for d0, _, s in all_jobs]
    log = pd.concat([pd.read_csv(p.with_suffix(".csv")) for p in paths], ignore_index=True)
    log.to_csv(rdir / "tables" / f"training_log_{name}.csv", index=False)
    preds = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    preds.to_parquet(rdir / "forecasts" / f"neural_{name}_seeds.parquet", index=False)
    ens = preds.groupby(["model", "asset", "date", "alpha"], as_index=False)[["var", "es"]].mean()
    ens.to_parquet(rdir / "forecasts" / f"neural_{name}.parquet", index=False)
    print(log.groupby("device")[["epochs", "best_val", "seconds"]].mean().round(3), flush=True)
