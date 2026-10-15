r"""Deep (VaR, ES) models trained with the FZ0 loss, pooled across assets.

Models
------
LSTM-FZ / TRF-FZ  ("pure"): inputs are the last L returns divided by a slow scale
    s_t = RMS(r_{t-250..t-1}); the network outputs standardised (VaR, ES) for every alpha,
    which are multiplied back by s_t. Volatility dynamics must be learned from the inputs.
GJR-LSTM ("hybrid"): sigma_t from a GJR-GARCH(1,1,1)-t fitted on the training window.
    The LSTM sees past standardised residuals z = (r - mu)/sigma and the volatility state,
    and outputs the conditional (VaR, ES) of r_t / sigma_t, i.e. a time-varying *tail
    shape*:  VaR_t = sigma_t q_t,  ES_t = sigma_t s_t.

Training objective: sum over alpha levels of the FZ0 loss, averaged over the pooled sample.
Because FZ0 satisfies L(cy, cv, ce) = L(y, v, e) + log c, training on standardised
targets is equivalent (same minimiser, same gradients w.r.t. network weights) to training
on raw returns with forecasts c*v, c*e, and every asset-day receives equal weight.

Output head: alphas sorted ascending (a_1 < ... < a_K). With g = softplus(u) + g_min > 0
    v_K = -g^v_K,   v_k = v_{k+1} - g^v_k,
    e_K = v_K - g^e_K,   e_k = min(v_k, e_{k+1}) - g^e_k,
which enforces e_k <= v_k < 0, v_1 <= ... <= v_K and e_1 <= ... <= e_K (no crossing).

Protocol: one network per (model, refit year, seed); refit at the start of each calendar
year on each asset's last W observations before the refit date; forecasts for the year
use parameters frozen at the refit date. Seeds are averaged (deep ensemble).
"""

from __future__ import annotations

import copy
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats

from .config import results_dir
from .data import load_returns
from .engine import to_long
from .models.garch import fit_gjr_t, gjr_filter
from .scoring import empirical_var_es

PURE = ("LSTM-FZ", "TRF-FZ")
HYBRID = ("GJR-LSTM",)
INPUT_CLIP = 10.0


# ----------------------------------------------------------------------------- loss / head
def fz0_torch(y: torch.Tensor, v: torch.Tensor, e: torch.Tensor, alphas: torch.Tensor) -> torch.Tensor:
    """Mean over the batch of sum_k FZ0(y, v_k, e_k; alpha_k). y: (B,), v, e: (B, K)."""
    e = torch.clamp(e, max=-1e-6)
    y = y[:, None]
    hit = (y <= v).float()
    loss = -hit * (v - y) / (alphas * e) + v / e + torch.log(-e) - 1.0
    return loss.sum(1).mean()


GAP_MIN = 1e-4  # floor on every gap: float32 softplus underflows to exactly 0 for u << 0


def _inv_softplus(x: np.ndarray) -> np.ndarray:
    return np.log(np.expm1(x))


class MonotoneTailHead(nn.Module):
    """MLP head producing non-crossing (VaR, ES) for K ascending alpha levels."""

    def __init__(self, n_in: int, hidden: int, alphas: list[float], dropout: float,
                 init: tuple[np.ndarray, np.ndarray] | None = None):
        super().__init__()
        self.K = len(alphas)
        self.mlp = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, 2 * self.K))
        # Bias initialisation = the model the network starts from, and the model early
        # stopping falls back to if no epoch improves validation loss. ``init`` passes the
        # empirical (VaR, ES) of the pooled standardised training targets (an unconditional
        # FHS-type tail); None gives standard-normal values (thin-tailed; used in run v1).
        a = np.asarray(alphas)
        if init is None:
            v0 = stats.norm.ppf(a)
            e0 = -stats.norm.pdf(v0) / a
        else:
            v0, e0 = init
        with torch.no_grad():
            self.mlp[-1].weight.mul_(0.1)
        self.set_init(v0, e0)

    def set_init(self, v0, e0) -> None:
        """Set the last-layer bias so that, with zero last-layer weights, the head outputs (v0, e0)."""
        v0, e0 = np.asarray(v0, dtype=float), np.asarray(e0, dtype=float)
        gv = np.empty(self.K)
        ge = np.empty(self.K)
        gv[-1] = -v0[-1]
        gv[:-1] = v0[1:] - v0[:-1]
        ge[-1] = v0[-1] - e0[-1]
        ge[:-1] = np.minimum(v0[:-1], e0[1:]) - e0[:-1]
        gv, ge = np.maximum(gv, 2 * GAP_MIN), np.maximum(ge, 2 * GAP_MIN)
        with torch.no_grad():
            self.mlp[-1].bias.copy_(torch.tensor(_inv_softplus(np.concatenate([gv, ge]) - GAP_MIN),
                                                 dtype=torch.float32))

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        g = F.softplus(self.mlp(h)) + GAP_MIN
        gv, ge = g[:, :self.K], g[:, self.K:]
        v = [None] * self.K
        e = [None] * self.K
        v[-1] = -gv[:, -1]
        for k in range(self.K - 2, -1, -1):
            v[k] = v[k + 1] - gv[:, k]
        e[-1] = v[-1] - ge[:, -1]
        for k in range(self.K - 2, -1, -1):
            e[k] = torch.minimum(v[k], e[k + 1]) - ge[:, k]
        return torch.stack(v, 1), torch.stack(e, 1)


class LSTMEncoder(nn.Module):
    def __init__(self, n_in: int, hidden: int, layers: int, dropout: float, **_):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return h[-1]


class TransformerEncoder(nn.Module):
    """Pre-norm Transformer over the input window; the last position summarises the sequence."""

    def __init__(self, n_in: int, hidden: int, layers: int, dropout: float, heads: int = 2,
                 lookback: int = 64, **_):
        super().__init__()
        self.inp = nn.Linear(n_in, hidden)
        self.pos = nn.Parameter(torch.randn(1, lookback, hidden) * 0.02)
        layer = nn.TransformerEncoderLayer(hidden, heads, dim_feedforward=2 * hidden,
                                           dropout=dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, max(layers, 1), enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):
        h = self.enc(self.inp(x) + self.pos)
        return self.norm(h[:, -1])


class TailNet(nn.Module):
    def __init__(self, encoder: str, n_seq: int, n_static: int, alphas: list[float], nc: dict,
                 init: tuple[np.ndarray, np.ndarray] | None = None):
        super().__init__()
        enc_cls = TransformerEncoder if encoder == "transformer" else LSTMEncoder
        self.enc = enc_cls(n_seq, nc["hidden"], nc["layers"], nc["dropout"],
                           heads=nc.get("heads", 2), lookback=nc["lookback"])
        self.head = MonotoneTailHead(nc["hidden"] + n_static, nc["hidden"], alphas, nc["dropout"],
                                     init)

    def forward(self, seq, static):
        return self.head(torch.cat([self.enc(seq), static], 1))


# ----------------------------------------------------------------------------- features
def _windows(x: np.ndarray, L: int, targets: np.ndarray) -> np.ndarray:
    """Rows x[t-L:t] for each target index t; x has shape (n,) or (n, C)."""
    w = sliding_window_view(x, L, axis=0)  # (n-L+1, [C,] L), row j = x[j:j+L]
    w = w[targets - L]
    return np.moveaxis(w, -1, 1) if x.ndim == 2 else w


def slow_scale(r: np.ndarray, window: int) -> np.ndarray:
    """s_t = sqrt(mean(r_{t-window..t-1}^2)), causal (uses data up to t-1)."""
    sq = pd.Series(r ** 2).rolling(window, min_periods=20).mean().shift(1)
    return np.sqrt(sq.to_numpy())


def asset_features(model: str, r: np.ndarray, idx0: int, idx_end: int, cfg: dict) -> dict:
    """Training sample (targets in [idx0-W, idx0)) and prediction sample ([idx0, idx_end))
    for one asset. Everything for target t is computed from r[:t]; the hybrid's GARCH
    parameters use only the training window r[idx0-W : idx0]."""
    nc, W = cfg["neural"], cfg["window"]
    L = nc["lookback"]
    s = slow_scale(r, nc["scale_window"])
    first = max(idx0 - W, L, 20 + 1)
    tr = np.arange(first, idx0)
    te = np.arange(idx0, idx_end)
    all_t = np.concatenate([tr, te])

    if model in PURE:
        win = _windows(r, L, all_t) / s[all_t, None]
        win = np.clip(win, -INPUT_CLIP, INPUT_CLIP)
        seq = np.stack([win, np.abs(win)], -1)
        static = np.log(s[all_t])[:, None]
        y = r[all_t] / s[all_t]
        loc, scale = np.zeros(all_t.size), s[all_t]
    else:
        p = fit_gjr_t(r[idx0 - W:idx0])
        start = max(0, idx0 - W - L - 250)       # burn-in for the variance filter
        s2 = gjr_filter(r[start:idx_end], p["mu"], p["omega"], p["alpha"], p["gamma"],
                        p["beta"], p["var0"])     # s2[j] = Var(r_{start+j} | past)
        sigma = np.full(r.size + 1, np.nan)
        sigma[start:idx_end + 1] = np.sqrt(s2)
        z = (r - p["mu"]) / sigma[:-1]
        z_seq = np.clip(_windows(np.nan_to_num(z), L, all_t), -INPUT_CLIP, INPUT_CLIP)
        seq = np.stack([z_seq, np.abs(z_seq)], -1)
        sig_t = sigma[all_t]
        static = np.column_stack([np.log(sig_t), np.log(sig_t / s[all_t])])
        # target r_t / sigma_t (not demeaned): FZ0 is scale- but not translation-equivariant,
        # so this keeps training exactly equivalent to FZ0 on raw returns; the network
        # absorbs the small mean mu / sigma_t into its outputs.
        y = r[all_t] / sig_t
        loc, scale = np.zeros(all_t.size), sig_t

    n_tr = tr.size
    return {
        "seq": seq.astype(np.float32), "static": static.astype(np.float32),
        "y": y.astype(np.float32), "loc": loc, "scale": scale,
        "n_tr": n_tr, "te_idx": te,
    }


# ----------------------------------------------------------------------------- training
def _batches(n: int, bs: int, rng: np.random.Generator | None):
    idx = rng.permutation(n) if rng is not None else np.arange(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


def train_tailnet(model: str, train: dict, val: dict, alphas: list[float], nc: dict, seed: int,
                  device: str = "cpu"):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device(device)
    encoder = "transformer" if model == "TRF-FZ" else "lstm"
    init = None
    if nc.get("head_init", "empirical") == "empirical":
        tails = np.array([empirical_var_es(train["y"].astype(float), a) for a in alphas])
        init = (tails[:, 0], tails[:, 1])
    net = TailNet(encoder, train["seq"].shape[2], train["static"].shape[1], alphas, nc, init).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=nc["lr"], weight_decay=nc["weight_decay"])
    a = torch.tensor(alphas, dtype=torch.float32, device=dev)
    # the pooled sample is small (~20 MB), so it lives on the device for the whole run
    T = {k: torch.from_numpy(v).to(dev) for k, v in train.items()}
    V = {k: torch.from_numpy(v).to(dev) for k, v in val.items()}

    def val_loss() -> float:
        net.eval()
        tot = 0.0
        with torch.no_grad():
            for b in _batches(V["y"].shape[0], 1024, None):  # small batches keep peak RAM low
                b = torch.from_numpy(b).to(dev)
                v, e = net(V["seq"][b], V["static"][b])
                tot += fz0_torch(V["y"][b], v, e, a).item() * b.numel()
        return tot / V["y"].shape[0]

    best, best_state, wait, epoch = val_loss(), copy.deepcopy(net.state_dict()), 0, 0
    for epoch in range(1, nc["max_epochs"] + 1):
        net.train()
        for b in _batches(T["y"].shape[0], nc["batch_size"], rng):
            b = torch.from_numpy(b).to(dev)
            v, e = net(T["seq"][b], T["static"][b])
            loss = fz0_torch(T["y"][b], v, e, a)
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


def resolve_device(nc: dict) -> str:
    """'auto' -> 'hybrid' (GPU slots + CPU workers) when a CUDA build of torch sees a GPU, else 'cpu'."""
    want = nc.get("device", "auto")
    if want == "auto":
        return "hybrid" if torch.cuda.is_available() else "cpu"
    return want


# Each worker process creates at most one CUDA context (>1 GB on Windows/cuDNN), so GPU use
# is limited to ``gpu_jobs`` worker processes. A worker that wins a slot keeps it for its
# lifetime; all other workers train on one CPU thread. Slots are lock files created with
# O_EXCL (atomic on Windows and POSIX) and are cleared at the start of every run.
_GPU_SLOT: int | None = None


def _gpu_lock_dir(cfg: dict) -> Path:
    return results_dir(cfg) / ".gpu_slots"


def _claim_device(mode: str, n_slots: int, lock_dir: Path) -> str:
    global _GPU_SLOT
    if mode == "cpu":
        return "cpu"
    if _GPU_SLOT is not None:
        return "cuda"
    for k in range(n_slots):
        try:
            fd = os.open(lock_dir / f"slot_{k}.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        _GPU_SLOT = k
        return "cuda"
    if mode == "cuda":
        raise RuntimeError("device=cuda but no free GPU slot; set n_jobs <= gpu_jobs")
    return "cpu"


def _refit_job(model: str, d0: pd.Timestamp, d1: pd.Timestamp | None, seed: int,
               rets: dict[str, pd.Series], cfg: dict, mode: str = "cpu") -> tuple[pd.DataFrame, dict]:
    torch.set_num_threads(1)
    tic = time.time()
    device = _claim_device(mode, cfg["neural"].get("gpu_jobs", 2), _gpu_lock_dir(cfg))
    nc, alphas = cfg["neural"], cfg["alphas"]
    feats = {}
    for asset, rs in rets.items():
        dates = rs.index
        idx0 = int(np.searchsorted(dates, d0))
        idx_end = int(np.searchsorted(dates, d1)) if d1 is not None else dates.size
        if idx0 >= dates.size or idx_end <= idx0:
            continue
        feats[asset] = asset_features(model, rs.to_numpy(dtype=float), idx0, idx_end, cfg)

    # time-ordered validation split: last val_frac of each asset's training targets
    parts = {"train": [], "val": []}
    for f in feats.values():
        n_tr = f["n_tr"]
        cut = int(n_tr * (1 - nc["val_frac"]))
        for name, sl in (("train", slice(0, cut)), ("val", slice(cut, n_tr))):
            parts[name].append({k: f[k][sl] for k in ("seq", "static", "y")})
    stack = {name: {k: np.concatenate([p[k] for p in ps]) for k in ("seq", "static", "y")}
             for name, ps in parts.items()}
    net, info = train_tailnet(model, stack["train"], stack["val"], alphas, nc, seed, device)

    frames = []
    with torch.no_grad():
        for asset, f in feats.items():
            n_tr = f["n_tr"]
            v, e = net(torch.from_numpy(f["seq"][n_tr:]).to(device),
                       torch.from_numpy(f["static"][n_tr:]).to(device))
            loc, scale = f["loc"][n_tr:, None], f["scale"][n_tr:, None]
            var = loc + scale * v.cpu().numpy().astype(float)
            es = loc + scale * e.cpu().numpy().astype(float)
            df = to_long(model, asset, rets[asset].index[f["te_idx"]], alphas, var, es)
            df["seed"] = seed
            frames.append(df)
    info.update({"model": model, "refit": d0.date(), "seed": seed, "device": device,
                 "n_train": int(stack["train"]["y"].size), "seconds": round(time.time() - tic, 1)})
    return pd.concat(frames, ignore_index=True), info


def refit_dates(cfg: dict) -> list[pd.Timestamp]:
    start = pd.Timestamp(cfg["oos_start"])
    years = pd.date_range(start, cfg["data"]["end"], freq="YS")
    dates = sorted({start, *years})
    return [d for d in dates if d >= start]


def _job_path(cfg: dict, model: str, d0: pd.Timestamp, seed: int) -> Path:
    return results_dir(cfg) / "nn_jobs" / f"{model}_{d0.date()}_s{seed}.parquet"


def _checkpointed_job(model, d0, d1, seed, rets, cfg, mode) -> dict:
    """Run one training job and write its forecasts to disk, so a crash never loses finished work."""
    df, info = _refit_job(model, d0, d1, seed, rets, cfg, mode)
    path = _job_path(cfg, model, d0, seed)
    df.to_parquet(path.with_suffix(".tmp"), index=False)
    path.with_suffix(".tmp").replace(path)                 # atomic: no half-written files
    pd.DataFrame([info]).to_csv(path.with_suffix(".csv"), index=False)
    return info


def run_neural(cfg: dict, models: list[str] | None = None) -> None:
    """Train every (model, refit year, seed) job; resumable (finished jobs are skipped)."""
    nc = cfg["neural"]
    models = models or nc["models"]
    rets = load_returns(cfg)
    dates = refit_dates(cfg)
    ends = dates[1:] + [None]
    all_jobs = [(m, d0, d1, s) for m in models for d0, d1 in zip(dates, ends) for s in nc["seeds"]]
    (results_dir(cfg) / "nn_jobs").mkdir(parents=True, exist_ok=True)
    jobs = [j for j in all_jobs if not _job_path(cfg, j[0], j[1], j[3]).exists()]
    mode = resolve_device(nc)
    n_jobs = nc.get("gpu_jobs", 2) if mode == "cuda" else nc["n_jobs"]
    lock_dir = _gpu_lock_dir(cfg)
    lock_dir.mkdir(parents=True, exist_ok=True)
    for f in lock_dir.glob("slot_*.lock"):
        f.unlink()
    gpu = f", GPU {torch.cuda.get_device_name(0)} x{nc.get('gpu_jobs', 2)} workers" if mode != "cpu" else ""
    print(f"[nn] {len(all_jobs)} training jobs ({len(models)} models x {len(dates)} refits x "
          f"{len(nc['seeds'])} seeds), {len(all_jobs) - len(jobs)} already done, mode={mode}, "
          f"{n_jobs} workers{gpu}", flush=True)
    # longest jobs first (Transformer, then hybrid) so the pool drains evenly
    order = {"TRF-FZ": 0, "GJR-LSTM": 1, "LSTM-FZ": 2}
    jobs.sort(key=lambda j: order.get(j[0], 3))
    if jobs:
        Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(_checkpointed_job)(m, d0, d1, s, rets, cfg, mode) for m, d0, d1, s in jobs)
    for f in lock_dir.glob("slot_*.lock"):
        f.unlink()

    # assemble from the per-job files
    rdir = results_dir(cfg)
    paths = [_job_path(cfg, m, d0, s) for m, d0, _, s in all_jobs]
    log = pd.concat([pd.read_csv(p.with_suffix(".csv")) for p in paths], ignore_index=True)
    log.to_csv(rdir / "tables" / "neural_training_log.csv", index=False)
    preds = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    for m in models:
        pm = preds[preds.model == m]
        pm.to_parquet(rdir / "forecasts" / f"neural_{m}_seeds.parquet", index=False)
        ens = pm.groupby(["model", "asset", "date", "alpha"], as_index=False)[["var", "es"]].mean()
        ens.to_parquet(rdir / "forecasts" / f"neural_{m}.parquet", index=False)
    print(log.groupby(["model", "device"])[["epochs", "best_val", "seconds"]].mean().round(3), flush=True)
