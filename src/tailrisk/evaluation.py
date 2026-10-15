"""Assemble all forecasts into a common panel and produce every table of the paper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from .backtests import backtest_all
from .comparison import diebold_mariano, model_confidence_set, panel_diebold_mariano
from .config import results_dir
from .data import load_returns
from .scoring import fz0_loss

ECON = ["HS", "GARCH-N", "GARCH-t", "GJR-skt", "FHS", "GARCH-EVT", "CAViaR-ES", "FZ-1F", "FZ-Hyb"]
DEEP = ["LSTM-FZ", "TRF-FZ", "GJR-LSTM"]
NEW = ["NG-FZ", "GJRX-skt", "LSTMQ-FZ", "CTG-FZ"]   # paper 2 (only included with include_new=True)
NEW_NEURAL = ["NG-FZ", "LSTMQ-FZ", "CTG-FZ"]
# Pre-specified equal-weight combinations (fixed before looking at results; docs/PROTOCOL.md)
COMBOS = {"COMB-Econ": ECON[1:], "COMB-All": ECON[1:] + DEEP}
COMBOS_NEW = {"COMB-All+NG": ECON[1:] + DEEP + ["NG-FZ"],
              "COMB-All+X": ECON[1:] + DEEP + ["GJRX-skt", "CTG-FZ"]}


@dataclass
class Panel:
    """Wide frames indexed by (asset, date, alpha) with one column per model."""
    var: pd.DataFrame
    es: pd.DataFrame
    loss: pd.DataFrame
    r: pd.Series
    models: list[str]

    def slice(self, alpha: float, asset: str | None = None):
        idx = self.var.index
        m = idx.get_level_values("alpha") == alpha
        if asset is not None:
            m &= idx.get_level_values("asset") == asset
        return self.var[m], self.es[m], self.loss[m], self.r[m]


def build_panel(cfg: dict, include_new: bool = False) -> Panel:
    """All forecasts on a common sample. ``include_new`` adds the paper-2 model(s) and their
    combination; the default reproduces the paper-1 model set exactly."""
    fdir = results_dir(cfg) / "forecasts"
    neural = DEEP + (NEW_NEURAL if include_new else [])
    combos = {**COMBOS, **(COMBOS_NEW if include_new else {})}
    order = ECON + DEEP + (NEW if include_new else []) + list(combos)
    files = sorted(fdir.glob("econ_*.parquet")) + [fdir / f"neural_{m}.parquet" for m in neural]
    if include_new:
        files += sorted(fdir.glob("xm_*.parquet"))           # GJR-X with cross-market news
    df = pd.concat([pd.read_parquet(f) for f in files if f.exists()], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    key = ["asset", "date", "alpha"]
    var = df.pivot(index=key, columns="model", values="var")
    es = df.pivot(index=key, columns="model", values="es")
    for name, members in combos.items():
        if all(m in var.columns for m in members):
            var[name] = var[members].mean(1)
            es[name] = es[members].mean(1)
    models = [m for m in order if m in var.columns]
    var, es = var[models], es[models]
    keep = var.notna().all(1) & es.notna().all(1)       # common evaluation sample
    var, es = var[keep], es[keep]

    rets = load_returns(cfg)
    r = pd.Series(np.nan, index=var.index, name="r")
    for a, s in rets.items():
        m = var.index.get_level_values("asset") == a
        r[m] = s.reindex(var.index[m].get_level_values("date")).to_numpy()
    alpha = var.index.get_level_values("alpha").to_numpy()
    loss = pd.DataFrame({m: fz0_loss(r.to_numpy(), var[m].to_numpy(), es[m].to_numpy(), alpha)
                         for m in models}, index=var.index)
    return Panel(var, es, loss, r, models)


# ----------------------------------------------------------------------------- tables
def _diff_wide(L: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """(date x asset) frame of loss differentials L_a - L_b."""
    d = (L[a] - L[b]).rename("d").reset_index()
    return d.pivot(index="date", columns="asset", values="d")


def descriptive_stats(cfg: dict) -> pd.DataFrame:
    rets = load_returns(cfg)
    rows = []
    for a, r in rets.items():
        oos = r[r.index >= cfg["oos_start"]]
        rows.append({"Asset": a, "Start": r.index[0].date(), "N": r.size, "N (OOS)": oos.size,
                     "Mean": r.mean(), "Std": r.std(), "Skew": stats.skew(r),
                     "Ex. kurt": stats.kurtosis(r), "Min": r.min(), "Max": r.max(),
                     "Min (OOS)": oos.min()})
    return pd.DataFrame(rows).set_index("Asset")


def loss_summary(p: Panel, alpha: float, benchmark: str) -> pd.DataFrame:
    """Average FZ0 loss, difference vs benchmark with panel DM test, per-asset DM win/loss counts, mean rank."""
    _, _, L, _ = p.slice(alpha)
    by_asset = L.groupby(level="asset").mean()           # assets x models
    ranks = by_asset.rank(axis=1).mean()
    rows = []
    for m in p.models:
        d = _diff_wide(L, m, benchmark)
        pdm = panel_diebold_mariano(d) if m != benchmark else {"mean_diff": 0.0, "dm_t": np.nan}
        better = worse = 0
        if m != benchmark:
            for a in d.columns:
                res = diebold_mariano(L[m].xs(a, level="asset"), L[benchmark].xs(a, level="asset"))
                better += res["dm_t"] < -1.96
                worse += res["dm_t"] > 1.96
        rows.append({"Model": m, "Avg loss": by_asset[m].mean(), "Diff vs bench": pdm["mean_diff"],
                     "Panel DM t": pdm["dm_t"], "# better": better, "# worse": worse,
                     "Mean rank": ranks[m]})
    return pd.DataFrame(rows).set_index("Model")


def combination_value(p: Panel, alphas: list[float], with_deep: str = "COMB-All",
                      without: str = "COMB-Econ") -> pd.DataFrame:
    """Do the deep models add information? Compare the two pre-specified combinations that
    differ only by the inclusion of the deep models (panel and per-asset DM tests)."""
    rows = []
    for a in alphas:
        _, _, L, _ = p.slice(a)
        d = _diff_wide(L, with_deep, without)
        pdm = panel_diebold_mariano(d)
        per = [diebold_mariano(L[with_deep].xs(x, level="asset"), L[without].xs(x, level="asset"))["dm_t"]
               for x in d.columns]
        rows.append({"alpha": a, "Loss diff": pdm["mean_diff"], "Panel DM t": pdm["dm_t"],
                     "p-value": pdm["p"], "# assets better": int(np.sum(np.array(per) < -1.96)),
                     "# assets worse": int(np.sum(np.array(per) > 1.96))})
    out = pd.DataFrame(rows)
    out["alpha"] = [f"{100 * a:g}%" for a in out["alpha"]]
    return out.set_index("alpha").rename_axis("Level")


def loss_by_asset(p: Panel, alpha: float) -> pd.DataFrame:
    _, _, L, _ = p.slice(alpha)
    return L.groupby(level="asset").mean().T


def _bt_job(m, a, alpha, r, v, e, n_boot):
    out = backtest_all(r, v, e, alpha, n_boot=n_boot)
    out.update({"model": m, "asset": a, "alpha": alpha})
    return out


def run_backtests(p: Panel, alphas: list[float], n_boot: int, n_jobs: int = -1) -> pd.DataFrame:
    jobs = []
    for alpha in alphas:
        for a in p.var.index.get_level_values("asset").unique():
            V, E, _, r = p.slice(alpha, a)
            for m in p.models:
                jobs.append(delayed(_bt_job)(m, a, alpha, r.to_numpy(), V[m].to_numpy(),
                                             E[m].to_numpy(), n_boot))
    return pd.DataFrame(Parallel(n_jobs=n_jobs)(jobs))


def backtest_summary(bt: pd.DataFrame, models: list[str], level: float = 0.05) -> pd.DataFrame:
    """Hit rate averaged over assets and number of assets rejecting each test at ``level``."""
    g = bt.groupby(["alpha", "model"])
    out = pd.DataFrame({
        "Hit %": g["rate"].mean() * 100,
        "UC": g["p_uc"].apply(lambda p: int((p < level).sum())),
        "CC": g["p_cc"].apply(lambda p: int((p < level).sum())),
        "DQ": g["p_dq"].apply(lambda p: int((p < level).sum())),
        "ER": g["p_er"].apply(lambda p: int((p < level).sum())),
        "CCT-u": g["p_cct_u"].apply(lambda p: int((p < level).sum())),
        "CCT-c": g["p_cct_c"].apply(lambda p: int((p < level).sum())),
        "Z2": g["z2"].mean(),
    })
    out = out.reindex(models, level="model")
    out.index = out.index.set_names(["alpha", "Model"])
    return out


def _mcs_job(alpha, a, losses, size, reps):
    pv = model_confidence_set(losses, size=size, reps=reps)
    return alpha, a, pv


def run_mcs(p: Panel, alphas: list[float], size: float, reps: int, n_jobs: int = -1) -> pd.DataFrame:
    """MCS p-values per (alpha, asset) plus a 'Pooled' set on the cross-asset average loss."""
    jobs = []
    for alpha in alphas:
        _, _, L, _ = p.slice(alpha)
        for a in L.index.get_level_values("asset").unique():
            jobs.append(delayed(_mcs_job)(alpha, a, L.xs(a, level="asset").droplevel("alpha"), size, reps))
        pooled = L.groupby(level="date").mean()
        jobs.append(delayed(_mcs_job)(alpha, "Pooled", pooled, size, reps))
    rows = []
    for alpha, a, pv in Parallel(n_jobs=n_jobs)(jobs):
        for m, v in pv.items():
            rows.append({"alpha": alpha, "asset": a, "model": m, "p_mcs": v})
    return pd.DataFrame(rows)


def mcs_summary(mcs: pd.DataFrame, models: list[str], size: float) -> pd.DataFrame:
    """Number of assets whose (1-size) MCS contains the model, and pooled-MCS membership."""
    ind = mcs[mcs.asset != "Pooled"]
    cnt = ind.assign(inc=ind.p_mcs > size).groupby(["model", "alpha"])["inc"].sum().unstack("alpha")
    pooled = mcs[mcs.asset == "Pooled"].pivot(index="model", columns="alpha", values="p_mcs")
    cnt.columns = [f"# assets {100 * a:g}%" for a in cnt.columns]
    pooled.columns = [f"Pooled p {100 * a:g}%" for a in pooled.columns]
    cnt.index.name = "Model"
    return cnt.join(pooled).reindex(models)


def subperiod_table(p: Panel, alpha: float, benchmark: str, periods: dict) -> pd.DataFrame:
    """Loss difference vs benchmark (panel DM t-stat) within each named subperiod."""
    _, _, L, _ = p.slice(alpha)
    dates = L.index.get_level_values("date")
    rows = {}
    for name, (s, e) in periods.items():
        Ls = L[(dates >= s) & (dates <= e)]
        col = {}
        for m in p.models:
            if m == benchmark:
                continue
            d = _diff_wide(Ls, m, benchmark)
            res = panel_diebold_mariano(d)
            col[m] = (res["mean_diff"], res["dm_t"])
        rows[name] = col
    diff = pd.DataFrame({k: {m: v[0] for m, v in c.items()} for k, c in rows.items()})
    tst = pd.DataFrame({k: {m: v[1] for m, v in c.items()} for k, c in rows.items()})
    return diff, tst


def regulatory_table(p: Panel, es_alpha: float = 0.025, var_alpha: float = 0.01,
                     window: int = 250) -> pd.DataFrame:
    """Capital proxy and Basel traffic-light outcomes.

    Capital proxy: average -ES at 97.5% (the FRTB risk measure), in % of position value, averaged
    over assets. Traffic light: for each day, count 99%-VaR violations over the trailing 250 days;
    green 0-4, yellow 5-9, red >= 10 (Basel backtesting zones). Reported as % of days.
    """
    V1, _, _, r1 = p.slice(var_alpha)
    _, E2, _, _ = p.slice(es_alpha)
    rows = []
    for m in p.models:
        cap = (-E2[m]).groupby(level="asset").mean().mean()
        yellow, red = [], []
        for a in V1.index.get_level_values("asset").unique():
            hits = (r1.xs(a, level="asset") <= V1[m].xs(a, level="asset")).astype(int)
            cnt = hits.rolling(window).sum().dropna()
            yellow.append(((cnt >= 5) & (cnt <= 9)).mean())
            red.append((cnt >= 10).mean())
        rows.append({"Model": m, "Capital (avg -ES 2.5%)": cap,
                     "Yellow zone % days": 100 * np.mean(yellow), "Red zone % days": 100 * np.mean(red)})
    return pd.DataFrame(rows).set_index("Model")


def seed_robustness(cfg: dict, p: Panel, alpha: float) -> pd.DataFrame:
    """Average loss of each single seed versus the seed ensemble (neural models in the panel)."""
    fdir = results_dir(cfg) / "forecasts"
    rows = []
    for m in [x for x in DEEP + NEW_NEURAL if x in p.models]:
        f = fdir / f"neural_{m}_seeds.parquet"
        if not f.exists():
            continue
        df = pd.read_parquet(f)
        df = df[df.alpha == alpha]
        df["date"] = pd.to_datetime(df["date"])
        idx = p.slice(alpha)[0].index
        for s, g in df.groupby("seed"):
            g = g.set_index(["asset", "date", "alpha"]).reindex(idx)
            l = fz0_loss(p.r.reindex(idx).to_numpy(), g["var"].to_numpy(), g["es"].to_numpy(), alpha)
            rows.append({"Model": m, "Seed": f"Seed {s}",
                         "Avg loss": pd.Series(l, index=idx).groupby(level="asset").mean().mean()})
        rows.append({"Model": m, "Seed": "Ensemble",
                     "Avg loss": p.slice(alpha)[2][m].groupby(level="asset").mean().mean()})
    return pd.DataFrame(rows).pivot(index="Model", columns="Seed", values="Avg loss") if rows else pd.DataFrame()


# ----------------------------------------------------------------------------- LaTeX output
def write_latex(df: pd.DataFrame, path: Path, caption: str, label: str, notes: str = "",
                fmt: dict | None = None, default_fmt: str = "{:.3f}", bold: dict | None = None,
                resize: bool = False) -> None:
    """Minimal booktabs table writer. ``bold`` maps column -> 'min'|'max' to bold the best entry;
    ``resize`` scales a wide table to the text width."""
    fmt = fmt or {}
    cols = list(df.columns)
    best = {}
    for c, how in (bold or {}).items():
        s = pd.to_numeric(df[c], errors="coerce")
        best[c] = s.idxmin() if how == "min" else s.idxmax()

    def cell(r, c, x):
        if isinstance(x, (float, np.floating)):
            if np.isnan(x):
                return ""
            s = fmt.get(c, default_fmt).format(x)
        else:
            s = str(x)
        s = s.replace("%", r"\%")
        return rf"\textbf{{{s}}}" if best.get(c) == r else s

    esc = lambda s: str(s).replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")
    lines = [r"\begin{table}[htbp]", r"\centering", r"\footnotesize",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             *([r"\resizebox{\textwidth}{!}{%"] if resize else []),
             r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
             " & ".join([esc(df.index.name or "")] + [esc(c) for c in cols]) + r" \\", r"\midrule"]
    for r, row in df.iterrows():
        lines.append(" & ".join([esc(r)] + [cell(r, c, row[c]) for c in cols]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"] + (["}"] if resize else [])
    if notes:
        lines.append(rf"\par\smallskip\parbox{{\linewidth}}{{\scriptsize {notes}}}")
    lines.append(r"\end{table}")
    path.write_text("\n".join(lines), encoding="utf-8")
