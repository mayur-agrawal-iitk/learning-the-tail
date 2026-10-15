r"""Scoring and identification functions for the pair (VaR, ES).

Sign convention (lower tail, returns in percent):
    VaR_t = v_t = alpha-quantile of r_t given F_{t-1}           (negative number)
    ES_t  = e_t = E[r_t | r_t <= v_t, F_{t-1}]                  (e_t <= v_t < 0)

ES on its own is not elicitable (Gneiting 2011), but the pair (VaR, ES) is jointly
elicitable (Fissler & Ziegel 2016). We use the FZ0 member of the Fissler-Ziegel class
(Patton, Ziegel & Chen 2019):

    L(y, v, e) = -1/(alpha e) * 1{y <= v} (v - y) + v/e + log(-e) - 1,

which is strictly consistent for (VaR, ES) on {e <= v < 0}. Loss *differences* are
invariant to rescaling (y, v, e) -> (c y, c v, c e): L(cy, cv, ce) = L(y, v, e) + log c,
so average losses can be compared across assets with different volatility.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-8


def fz0_loss(y, v, e, alpha: float) -> np.ndarray:
    """Pointwise FZ0 loss. Inputs broadcast; e is clipped to stay strictly negative."""
    y = np.asarray(y, dtype=float)
    v = np.asarray(v, dtype=float)
    e = np.minimum(np.asarray(e, dtype=float), -EPS)
    hit = (y <= v).astype(float)
    return -hit * (v - y) / (alpha * e) + v / e + np.log(-e) - 1.0


def quantile_loss(y, v, alpha: float) -> np.ndarray:
    """Pinball (tick) loss, strictly consistent for the alpha-quantile."""
    y = np.asarray(y, dtype=float)
    v = np.asarray(v, dtype=float)
    return ((y <= v).astype(float) - alpha) * (v - y)


def identification(y, v, e, alpha: float) -> np.ndarray:
    r"""Identification function V(y, v, e) for (VaR, ES), shape (n, 2).

    V1 = 1{y <= v} - alpha
    V2 = e - v + (1/alpha) 1{y <= v} (v - y)

    At the true (VaR, ES): E[V | F_{t-1}] = 0. Used by the conditional calibration test.
    """
    y = np.asarray(y, dtype=float)
    hit = (y <= v).astype(float)
    v1 = hit - alpha
    v2 = e - v + hit * (v - y) / alpha
    return np.column_stack([v1, v2])


def empirical_var_es(x: np.ndarray, alpha: float) -> tuple[float, float]:
    """Empirical lower-tail VaR (linear-interpolated quantile) and ES of a sample."""
    x = np.asarray(x, dtype=float)
    v = float(np.quantile(x, alpha))
    tail = x[x <= v]
    e = float(tail.mean()) if tail.size else v
    return v, e
