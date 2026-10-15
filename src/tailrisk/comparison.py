r"""Relative (comparative) forecast evaluation with the FZ0 loss.

* Diebold-Mariano (1995) test of equal predictive accuracy with a Newey-West HAC
  variance; a cross-sectional (panel) version averages loss differentials over assets
  on each date before testing.
* Model Confidence Set (Hansen, Lunde & Nason 2011) via ``arch.bootstrap.MCS``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from arch.bootstrap import MCS
from scipy import stats


def newey_west_var(x: np.ndarray, lags: int | None = None) -> float:
    """Long-run variance of a series (Bartlett kernel)."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = x.size
    if lags is None:
        lags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = x @ x / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * (x[k:] @ x[:-k]) / n
    return float(lrv)


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray) -> dict:
    """DM test on d_t = L_a - L_b. Negative t-stat => model a is more accurate."""
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    se = np.sqrt(newey_west_var(d) / n)
    t = float(d.mean() / se) if se > 0 else 0.0
    return {"mean_diff": float(d.mean()), "dm_t": t, "p": float(2 * stats.norm.sf(abs(t)))}


def panel_diebold_mariano(diffs: pd.DataFrame) -> dict:
    """DM test on the cross-sectional average of loss differentials.

    ``diffs`` is a (date x asset) frame of L_a - L_b; dates where an asset did not trade
    are NaN and the average is taken over available assets. HAC variance absorbs both
    serial and (through the averaging) cross-sectional dependence.
    """
    dbar = diffs.mean(axis=1, skipna=True).dropna().to_numpy()
    n = dbar.size
    se = np.sqrt(newey_west_var(dbar) / n)
    t = float(dbar.mean() / se) if se > 0 else 0.0
    return {"mean_diff": float(dbar.mean()), "dm_t": t, "p": float(2 * stats.norm.sf(abs(t)))}


def model_confidence_set(losses: pd.DataFrame, size: float = 0.10, reps: int = 2000,
                         block_size: int = 20, seed: int = 0) -> pd.Series:
    """MCS p-values for each model (column). Models with p > size are in the (1-size) MCS."""
    mcs = MCS(losses.to_numpy(), size=size, reps=reps, block_size=block_size,
              method="R", bootstrap="stationary", seed=seed)
    mcs.compute()
    pv = mcs.pvalues["Pvalue"]
    pv.index = [losses.columns[int(i)] for i in pv.index]
    return pv.reindex(losses.columns)
