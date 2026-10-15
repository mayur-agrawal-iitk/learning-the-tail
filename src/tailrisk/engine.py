"""Rolling-window out-of-sample engine for the per-asset (econometric) models.

Timeline for one asset (indices into its own trading calendar):
    t_start = first trading day on/after cfg['oos_start']
    refit dates t0 = t_start, t_start + R, t_start + 2R, ...
    at each t0: estimate on r[t0-W : t0], then forecast every day t in [t0, t0+R)
    using parameters fixed at t0 and data up to t-1 (filters keep updating daily).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .config import results_dir
from .data import load_returns
from .models import econometric_estimators


def first_oos_index(dates: pd.DatetimeIndex, oos_start: str, window: int) -> int:
    t = int(np.searchsorted(dates, pd.Timestamp(oos_start)))
    if t < window:
        raise ValueError(f"only {t} observations before {oos_start}; need window={window}")
    return t


def to_long(model: str, asset: str, dates, alphas, var: np.ndarray, es: np.ndarray) -> pd.DataFrame:
    """(n, K) arrays -> long frame with one row per (date, alpha)."""
    n, k = var.shape
    return pd.DataFrame({
        "model": model, "asset": asset,
        "date": np.repeat(np.asarray(dates), k),
        "alpha": np.tile(np.asarray(alphas, dtype=float), n),
        "var": var.ravel(), "es": es.ravel(),
    })


def run_asset(asset: str, r: pd.Series, cfg: dict, models: list[str] | None = None,
              alphas: list[float] | None = None, verbose: bool = True) -> pd.DataFrame:
    """Rolling forecasts for one asset; optionally only the estimators producing ``models``
    and only the given ``alphas`` (used to parallelise the slow semiparametric models)."""
    tic = time.time()
    W, R = cfg["window"], cfg["econometric"]["refit_every"]
    alphas = alphas or cfg["alphas"]
    x = r.to_numpy(dtype=float)
    t_start = first_oos_index(r.index, cfg["oos_start"], W)
    estimators = econometric_estimators(cfg)
    if models is not None:
        estimators = [e for e in estimators if set(e.names) & set(models)]
    blocks: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for t0 in range(t_start, x.size, R):
        t1 = min(t0 + R, x.size)
        for est in estimators:
            for model, (v, e) in est.fit_predict(x, t0, t1, W, alphas).items():
                blocks.setdefault(model, []).append((v, e))
    dates = r.index[t_start:]
    frames = [to_long(m, asset, dates, alphas, np.vstack([b[0] for b in bl]),
                      np.vstack([b[1] for b in bl])) for m, bl in blocks.items()
              if models is None or m in models]
    if verbose:
        print(f"[econ] {asset:8s} {len(dates)} days, {len(frames)} models, alphas={alphas}, "
              f"{time.time() - tic:.0f}s", flush=True)
    return pd.concat(frames, ignore_index=True)


def run_econometric(cfg: dict, assets: list[str] | None = None,
                    models: list[str] | None = None) -> None:
    """All benchmark forecasts. With ``models``, only those models are re-run (one job per
    asset x model x alpha) and merged into the existing per-asset files."""
    rets = load_returns(cfg)
    assets = assets or list(rets)
    out = results_dir(cfg) / "forecasts"
    n_jobs = cfg["econometric"]["n_jobs"]
    if models is None:
        jobs = (delayed(run_asset)(a, rets[a], cfg) for a in assets)
        for a, df in zip(assets, Parallel(n_jobs=n_jobs)(jobs)):
            df.to_parquet(out / f"econ_{a}.parquet", index=False)
        return
    keys = [(a, m, al) for a in assets for m in models for al in cfg["alphas"]]
    res = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_asset)(a, rets[a], cfg, [m], [al], False) for a, m, al in keys)
    new = pd.concat(res, ignore_index=True)
    for a in assets:
        path = out / f"econ_{a}.parquet"
        old = pd.read_parquet(path)
        old = old[~old.model.isin(models)]
        pd.concat([old, new[new.asset == a]], ignore_index=True).to_parquet(path, index=False)
