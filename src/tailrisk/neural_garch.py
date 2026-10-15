r"""Neural news-impact GARCH for (VaR, ES): NG-FZ.

Returns are scaled by a fixed per-asset, per-refit constant s (training-window RMS),
y_t = r_t / s, and the model is

    h_t      = A h_{t-1} + N_theta(y_{t-1}),          A = diag(a),  0 < a_k < rho < 1
    N_theta  = B x(y) + c + g * MLP(u(y))             (learned news-impact function)
    log sig2 = log softplus(h_t[0]) + g' * MLP(phi_t), phi_t = log softplus(h_t)
    (VaR_t, ES_t) = s * sig_t * Head(phi_t, log sig2_t)   (monotone, non-crossing)

with GARCH inputs x(y) = (y, y^2, y^2 1{y<0}, |y|) and bounded neural inputs
u(y) = (tanh(y/2), log(1 + y^2), 1{y<0}).

Properties (proved in the paper):
* Stability / invertibility. The state recursion is linear in h with ||A|| <= rho < 1, so
  for any input path the filter forgets its initial condition: |h_t - h'_t| <= rho^t |h_0 - h'_0|.
  For stationary ergodic returns with E log+ |N(y)| < inf there is a unique stationary
  solution h_t = sum_j A^j N(y_{t-1-j}) (Brandt 1986; Bougerol 1993).
* Nesting. With g = g' = 0 and d = 1, h_t is exactly the GJR-GARCH(1,1) variance and the
  head is a constant standardised tail, i.e. an FHS-type GJR model. The network is
  initialised there (gates at zero, GARCH part at the pooled GJR fit, tail at the empirical
  tail of the GJR residuals) and learns corrections by minimising FZ0.
* Interpretability. N_theta(y) is a learned news-impact curve (Engle & Ng 1993); the extra
  state coordinates are multi-scale volatility memories (EWMA of y^2 with decays
  0.5, ..., 0.995 at initialisation), as in component GARCH / HAR models.
* Pooling is exact by the scale property of FZ0, since s is fixed within a refit.
"""

from __future__ import annotations

import copy
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from arch import arch_model
from joblib import Parallel, delayed

from .config import results_dir
from .data import load_returns
from .engine import to_long
from .neural import MonotoneTailHead, fz0_torch, refit_dates
from .scoring import empirical_var_es

NAME = "NG-FZ"
SOFTPLUS_BETA = 20.0
MEMORY_DECAYS = (0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995)


def garch_inputs(y: torch.Tensor) -> torch.Tensor:
    neg = (y < 0).to(y.dtype)
    return torch.stack([y, y * y, y * y * neg, y.abs()], -1)


def neural_inputs(y: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.tanh(y / 2.0), torch.log1p(y * y), (y < 0).to(y.dtype)], -1)


class NeuralGarch(nn.Module):
    """``garch`` is one dict of GJR parameters (pooled GARCH block, run v1) or a list with one
    dict per asset (asset-specific GARCH block, v2). Memory coordinates, the neural news-impact
    correction, the variance readout and the tail head are always shared across assets.
    Inputs y have shape (n_assets, T) in the same asset order as ``garch``."""

    def __init__(self, alphas: list[float], garch: dict | list[dict], d: int = 8, hidden: int = 16,
                 rho: float = 0.999):
        super().__init__()
        self.rho, self.d = rho, d
        g = [garch] if isinstance(garch, dict) else list(garch)
        col = lambda k: np.array([x[k] for x in g], dtype=float)      # noqa: E731
        beta = np.clip(col("beta"), 1e-3, rho - 1e-4)
        mem = np.clip(np.array(MEMORY_DECAYS[:d - 1], dtype=float), 1e-3, rho - 1e-4)
        logit = lambda x: torch.logit(torch.tensor(x / rho, dtype=torch.float32))  # noqa: E731
        # GARCH block (coordinate 0): one row per asset (or a single pooled row)
        self.a0_raw = nn.Parameter(logit(beta))                                    # (n,)
        B0 = torch.zeros(len(g), 4)
        B0[:, 1] = torch.tensor(col("alpha"), dtype=torch.float32)
        B0[:, 2] = torch.tensor(col("gamma"), dtype=torch.float32)
        self.B0 = nn.Parameter(B0)
        self.c0 = nn.Parameter(torch.tensor(col("omega"), dtype=torch.float32))
        # shared multi-scale memories (coordinates 1..d-1): EWMA of y^2 at initialisation
        self.am_raw = nn.Parameter(logit(mem))
        Bm = torch.zeros(d - 1, 4)
        Bm[:, 1] = torch.tensor(1.0 - mem, dtype=torch.float32)
        self.Bm = nn.Parameter(Bm)
        self.cm = nn.Parameter(torch.zeros(d - 1))
        # shared neural parts, gated off at initialisation
        self.nic = nn.Sequential(nn.Linear(3, hidden), nn.Tanh(), nn.Linear(hidden, d))
        self.g_nic = nn.Parameter(torch.zeros(1))
        self.readout = nn.Sequential(nn.Linear(d, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.g_read = nn.Parameter(torch.zeros(1))
        self.head = MonotoneTailHead(d + 1, hidden, alphas, dropout=0.0)
        persistence = col("alpha") + col("gamma") / 2 + beta
        h0 = torch.ones(len(g), d)
        h0[:, 0] = torch.tensor(col("omega") / np.maximum(1.0 - persistence, 1e-2), dtype=torch.float32)
        self.register_buffer("h0", h0)

    def garch_params(self) -> list[nn.Parameter]:
        return [self.a0_raw, self.B0, self.c0, self.am_raw, self.Bm, self.cm]

    def decay(self) -> torch.Tensor:
        """(n_par, d) decays; every entry lies in (0, rho)."""
        a0 = self.rho * torch.sigmoid(self.a0_raw)[:, None]
        am = self.rho * torch.sigmoid(self.am_raw)[None].expand(a0.shape[0], -1)
        return torch.cat([a0, am], 1)

    def news_impact(self, y: torch.Tensor, asset: int | None = None) -> torch.Tensor:
        """N_theta(y): contribution of today's standardised return to tomorrow's state.
        y: (n_assets, T), or a 1-d grid together with ``asset`` (for plotting)."""
        if y.dim() == 1:
            y = y[None]
            B0, c0 = self.B0[[asset or 0]], self.c0[[asset or 0]]
        else:
            B0, c0 = self.B0, self.c0
        x = garch_inputs(y)
        lin0 = (x * B0[:, None, :]).sum(-1) + c0[:, None]
        lin_m = x @ self.Bm.T + self.cm
        return torch.cat([lin0[..., None], lin_m], -1) + self.g_nic * self.nic(neural_inputs(y))

    def states(self, y: torch.Tensor, h0: torch.Tensor | None = None) -> torch.Tensor:
        """H[:, t] = state used to forecast y[:, t] (depends on y[:, :t] only)."""
        N = self.news_impact(y)
        a = self.decay()
        h = (self.h0 if h0 is None else h0).expand(y.shape[0], -1)
        out = []
        for t in range(y.shape[1]):
            out.append(h)
            h = a * h + N[:, t]
        return torch.stack(out, 1)

    def forward(self, y: torch.Tensor, h0: torch.Tensor | None = None):
        H = self.states(y, h0)
        phi = torch.log(F.softplus(H, beta=SOFTPLUS_BETA) + 1e-6)
        logvar = phi[..., 0] + self.g_read * self.readout(phi).squeeze(-1)
        sigma = torch.exp(0.5 * logvar)
        b, T, d = phi.shape
        v, e = self.head(torch.cat([phi, logvar[..., None]], -1).reshape(b * T, d + 1))
        v = v.reshape(b, T, -1) * sigma[..., None]
        e = e.reshape(b, T, -1) * sigma[..., None]
        return v, e, sigma


# ----------------------------------------------------------------------------- data
def _fit_gjr_zero_mean(y: np.ndarray) -> dict:
    am = arch_model(y, mean="Zero", vol="GARCH", p=1, o=1, q=1, dist="normal", rescale=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = am.fit(disp="off", options={"maxiter": 500}).params
    return {"omega": float(p["omega"]), "alpha": float(p["alpha[1]"]),
            "gamma": float(p["gamma[1]"]), "beta": float(p["beta[1]"])}


def build_panel(rets: dict[str, pd.Series], d0: pd.Timestamp, d1: pd.Timestamp | None,
                window: int, burn_in: int, val_frac: float) -> dict:
    """Left-padded (asset x time) arrays of y = r / s, with train / val / test masks."""
    seqs = {}
    for asset, rs in rets.items():
        r = rs.to_numpy(dtype=float)
        idx0 = int(np.searchsorted(rs.index, d0))
        idx_end = int(np.searchsorted(rs.index, d1)) if d1 is not None else r.size
        if idx0 >= r.size or idx_end <= idx0 or idx0 < window:
            continue
        start = max(0, idx0 - window - burn_in)
        scale = float(np.sqrt(np.mean(r[idx0 - window:idx0] ** 2)))
        y = r[start:idx_end] / scale
        n_val = int(window * val_frac)
        pos = np.arange(start, idx_end)
        seqs[asset] = {
            "y": y, "scale": scale, "dates": rs.index[idx0:idx_end],
            "train": (pos >= idx0 - window) & (pos < idx0 - n_val),
            "val": (pos >= idx0 - n_val) & (pos < idx0),
            "test": pos >= idx0,
            "gjr": _fit_gjr_zero_mean(r[idx0 - window:idx0] / scale),
        }
    T = max(len(s["y"]) for s in seqs.values())
    names = list(seqs)
    out = {"assets": names, "scale": np.array([seqs[a]["scale"] for a in names]),
           "dates": [seqs[a]["dates"] for a in names]}
    for key in ("y", "train", "val", "test"):
        arr = np.zeros((len(names), T), dtype=np.float32 if key == "y" else bool)
        for i, a in enumerate(names):
            arr[i, T - len(seqs[a]["y"]):] = seqs[a][key]      # left padding
        out[key] = arr
    g = pd.DataFrame([seqs[a]["gjr"] for a in names]).median()   # pooled starting point (v1)
    out["gjr"] = {k: float(g[k]) for k in g.index}
    out["gjr_list"] = [seqs[a]["gjr"] for a in names]            # per-asset starting points (v2)
    out["n_test"] = [int(seqs[a]["test"].sum()) for a in names]
    return out


# ----------------------------------------------------------------------------- training
def _masked_loss(y, v, e, mask, alphas_t):
    m = torch.from_numpy(mask)
    return fz0_torch(y[m], v[m], e[m], alphas_t)


def train_ng(panel: dict, alphas: list[float], nc: dict, seed: int):
    torch.manual_seed(seed)
    model = NeuralGarch(alphas, garch_arg(panel, nc), d=nc["d"], hidden=nc["hidden"], rho=nc["rho"])
    y_all = torch.from_numpy(panel["y"])
    last_train = int(np.max(np.nonzero((panel["train"] | panel["val"]).any(0))[0])) + 1
    y = y_all[:, :last_train]                                  # never feed the test block in training
    tr, va = panel["train"][:, :last_train], panel["val"][:, :last_train]
    a_t = torch.tensor(alphas, dtype=torch.float32)

    # tail initialisation: empirical (VaR, ES) of residuals of the initial (pure GJR) model
    with torch.no_grad():
        _, _, sigma = model(y)
        z = (y / sigma).numpy()[tr]
    tails = np.array([empirical_var_es(z.astype(float), a) for a in alphas])
    model.head.set_init(tails[:, 0], tails[:, 1])

    garch_params = model.garch_params()
    ids = {id(p) for p in garch_params}
    other = [p for p in model.parameters() if id(p) not in ids]
    opt = torch.optim.Adam([{"params": garch_params, "lr": nc["lr_garch"]},
                            {"params": other, "lr": nc["lr"]}])

    def val_loss() -> float:
        with torch.no_grad():
            v, e, _ = model(y)
            return _masked_loss(y, v, e, va, a_t).item()

    best, best_state, wait, step = val_loss(), copy.deepcopy(model.state_dict()), 0, 0
    init_val = best
    lam = float(nc.get("aux_qlike", 0.0))
    tr_t = torch.from_numpy(tr)
    for step in range(1, nc["max_steps"] + 1):
        v, e, sigma = model(y)
        loss = _masked_loss(y, v, e, tr, a_t)
        if lam > 0:
            # QLIKE: strictly consistent for E[y^2 | past]; adds the information in every
            # day's squared return to the tail-only FZ0 signal (efficiency, not a new target)
            s2 = sigma[tr_t] ** 2
            loss = loss + lam * (y[tr_t] ** 2 / s2 + torch.log(s2)).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % nc["eval_every"] == 0:
            cur = val_loss()
            if cur < best - 1e-5:
                best, best_state, wait = cur, copy.deepcopy(model.state_dict()), 0
            else:
                wait += 1
                if wait >= nc["patience"]:
                    break
    model.load_state_dict(best_state)
    return model, {"steps": step, "best_val": best, "init_val": init_val,
                   "g_nic": float(model.g_nic.detach()), "g_read": float(model.g_read.detach())}


def garch_arg(panel: dict, nc: dict):
    return panel["gjr_list"] if nc.get("garch_block", "asset") == "asset" else panel["gjr"]


def _job(d0, d1, seed, rets, cfg, save_path: Path | None = None) -> tuple[pd.DataFrame, dict]:
    torch.set_num_threads(1)
    tic = time.time()
    nc, alphas = cfg["neural_garch"], cfg["alphas"]
    panel = build_panel(rets, d0, d1, cfg["window"], nc["burn_in"], nc["val_frac"])
    model, info = train_ng(panel, alphas, nc, seed)
    if save_path is not None:          # weights + constructor args, for the news-impact figure
        torch.save({"state": model.state_dict(), "gjr": garch_arg(panel, nc), "alphas": alphas,
                    "assets": panel["assets"],
                    "d": nc["d"], "hidden": nc["hidden"], "rho": nc["rho"]}, save_path)
    with torch.no_grad():
        v, e, _ = model(torch.from_numpy(panel["y"]))
    frames = []
    for i, asset in enumerate(panel["assets"]):
        m = panel["test"][i]
        s = panel["scale"][i]
        df = to_long(NAME, asset, panel["dates"][i], alphas,
                     s * v[i][m].numpy().astype(float), s * e[i][m].numpy().astype(float))
        df["seed"] = seed
        frames.append(df)
    info.update({"model": NAME, "refit": d0.date(), "seed": seed, "seconds": round(time.time() - tic, 1)})
    return pd.concat(frames, ignore_index=True), info


def _job_path(cfg: dict, d0: pd.Timestamp, seed: int) -> Path:
    return results_dir(cfg) / "ng_jobs" / f"{NAME}_{d0.date()}_s{seed}.parquet"


def _checkpointed(d0, d1, seed, rets, cfg) -> dict:
    df, info = _job(d0, d1, seed, rets, cfg, _job_path(cfg, d0, seed).with_suffix(".pt"))
    path = _job_path(cfg, d0, seed)
    df.to_parquet(path.with_suffix(".tmp"), index=False)
    path.with_suffix(".tmp").replace(path)
    pd.DataFrame([info]).to_csv(path.with_suffix(".csv"), index=False)
    return info


def run_neural_garch(cfg: dict) -> None:
    nc = cfg["neural_garch"]
    rets = load_returns(cfg)
    dates = refit_dates(cfg)
    ends = dates[1:] + [None]
    all_jobs = [(d0, d1, s) for d0, d1 in zip(dates, ends) for s in nc["seeds"]]
    (results_dir(cfg) / "ng_jobs").mkdir(parents=True, exist_ok=True)
    jobs = [j for j in all_jobs if not _job_path(cfg, j[0], j[2]).exists()]
    print(f"[ng] {len(all_jobs)} jobs ({len(dates)} refits x {len(nc['seeds'])} seeds), "
          f"{len(all_jobs) - len(jobs)} already done, {nc['n_jobs']} CPU workers", flush=True)
    if jobs:
        Parallel(n_jobs=nc["n_jobs"], verbose=10)(
            delayed(_checkpointed)(d0, d1, s, rets, cfg) for d0, d1, s in jobs)
    rdir = results_dir(cfg)
    paths = [_job_path(cfg, d0, s) for d0, _, s in all_jobs]
    log = pd.concat([pd.read_csv(p.with_suffix(".csv")) for p in paths], ignore_index=True)
    log.to_csv(rdir / "tables" / "ng_training_log.csv", index=False)
    preds = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    preds.to_parquet(rdir / "forecasts" / f"neural_{NAME}_seeds.parquet", index=False)
    ens = preds.groupby(["model", "asset", "date", "alpha"], as_index=False)[["var", "es"]].mean()
    ens.to_parquet(rdir / "forecasts" / f"neural_{NAME}.parquet", index=False)
    print(log[["steps", "init_val", "best_val", "g_nic", "g_read", "seconds"]].describe().round(3), flush=True)
