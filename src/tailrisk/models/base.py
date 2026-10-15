"""Common estimator interface."""

from __future__ import annotations

import numpy as np

Forecasts = dict[str, tuple[np.ndarray, np.ndarray]]


class Estimator:
    """Rolling-window estimator.

    ``fit_predict(r, t0, t1, window, alphas)`` must
      * estimate parameters using only r[t0 - window : t0], and
      * return, for every t in [t0, t1), the one-step-ahead forecasts (VaR_t, ES_t)
        computed from r[:t] only (no look-ahead).
    The return value maps model name -> (var, es), each of shape (t1 - t0, len(alphas)).
    One estimator may produce several models (e.g. FHS and EVT share one GARCH filter).
    Estimators may keep state between calls (e.g. warm-start parameters).
    """

    names: list[str] = []

    def fit_predict(self, r: np.ndarray, t0: int, t1: int, window: int,
                    alphas: list[float]) -> Forecasts:
        raise NotImplementedError
