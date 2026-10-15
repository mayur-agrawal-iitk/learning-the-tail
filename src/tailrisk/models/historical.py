"""Historical simulation."""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .base import Estimator, Forecasts


class HistoricalSimulation(Estimator):
    """VaR/ES = empirical quantile / tail mean of the last ``window`` returns."""

    def __init__(self, window: int = 250):
        self.window = window
        self.names = ["HS"]

    def fit_predict(self, r, t0, t1, window, alphas) -> Forecasts:
        hw = self.window
        # row j holds r[t0 + j - hw : t0 + j], i.e. information up to t-1 for t = t0 + j
        win = sliding_window_view(r[t0 - hw:t1 - 1], hw)
        var = np.quantile(win, alphas, axis=1).T  # (n, K)
        es = np.empty_like(var)
        for k in range(len(alphas)):
            mask = win <= var[:, [k]]
            es[:, k] = (win * mask).sum(1) / mask.sum(1)
        return {"HS": (var, es)}
