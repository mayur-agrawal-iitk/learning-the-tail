"""Forecasting models. Every estimator maps a return history to one-step-ahead (VaR, ES)."""

from .base import Estimator
from .garch import ArchFamily
from .historical import HistoricalSimulation
from .semiparametric import CaviarES, FZGas1F, FZHybrid


def econometric_estimators(cfg: dict) -> list[Estimator]:
    """The benchmark set used in the paper (fresh instances, one set per asset)."""
    ec = cfg["econometric"]
    return [
        HistoricalSimulation(window=ec["hs_window"]),
        ArchFamily("GARCH-N", o=0, dist="normal"),
        ArchFamily("GARCH-t", o=0, dist="t"),
        ArchFamily("GJR-skt", o=1, dist="skewt"),
        # GJR-GARCH by Gaussian QMLE, used only as a filter for FHS and GARCH-EVT
        ArchFamily("GJR-QML", o=1, dist="normal", report_parametric=False,
                   fhs=True, evt_tail_frac=ec["evt_tail_frac"]),
        CaviarES(),
        FZGas1F(),
        FZHybrid(),
    ]
