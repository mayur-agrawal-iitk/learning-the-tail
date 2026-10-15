"""Paper figures (static, light mode). Colour roles follow a validated categorical order:
slot 1 blue, slot 2 orange, slot 3 aqua (all-pairs CVD-safe for <= 3 series); a blue <-> red
diverging map with a neutral grey midpoint; muted grey for context data."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
CONTEXT = "#c3c2b7"
DIVERGING = LinearSegmentedColormap.from_list(
    "bluegrayred", ["#104281", "#3987e5", "#b7d3f6", "#f0efec", "#f3b5b4", "#e34948", "#9e2323"])


def set_style() -> None:
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
        "axes.titlesize": 10, "axes.titlecolor": INK, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "grid.linestyle": "-", "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "legend.frameon": False, "legend.fontsize": 8,
        "legend.labelcolor": INK2, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "savefig.dpi": 200, "lines.linewidth": 1.4,
    })


def _legend_top(ax, ncol: int) -> None:
    """Legend in a row between the title and the plot area, so it never covers data."""
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=ncol, borderaxespad=0.2,
              handlelength=1.6, columnspacing=1.2)


def _end_labels(ax, items: list[tuple[str, pd.Timestamp, float]], min_gap_px: float = 11) -> None:
    """Direct labels at line ends, nudged apart vertically so they never overlap."""
    if not items:
        return
    trans = ax.transData
    pts = sorted(((trans.transform((matplotlib.dates.date2num(x), y))[1], lab, x, y)
                  for lab, x, y in items))
    placed = []
    for ypix, lab, x, y in pts:
        if placed and ypix - placed[-1][0] < min_gap_px:
            ypix = placed[-1][0] + min_gap_px
        placed.append((ypix, lab, x, y))
    inv = trans.inverted()
    for ypix, lab, x, y in placed:
        xpix = trans.transform((matplotlib.dates.date2num(x), y))[0]
        ax.annotate(lab, xy=inv.transform((xpix, ypix)), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK2,
                    annotation_clip=False)


def _save(fig, path: Path) -> None:
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def fig_episodes(p, asset: str, alpha: float, models: list[str], episodes: dict, path: Path,
                 asset_label: str = "") -> None:
    """Returns (grey) with VaR forecasts of up to three models in crisis windows."""
    set_style()
    V, _, _, r = p.slice(alpha, asset)
    dates = V.index.get_level_values("date")
    fig, axes = plt.subplots(1, len(episodes), figsize=(7.2, 2.8), sharey=False)
    for ax, (name, (s, e)) in zip(np.atleast_1d(axes), episodes.items()):
        m = (dates >= s) & (dates <= e)
        d = dates[m]
        ax.bar(d, r[m].to_numpy(), width=1.0, color=CONTEXT, linewidth=0, label="Daily return")
        for k, mod in enumerate(models):
            ax.plot(d, V[mod][m].to_numpy(), color=SERIES[k], lw=1.2, label=mod)
        ax.set_title(name)
        ax.set_ylabel("Return / VaR (%)" if ax is np.atleast_1d(axes)[0] else "")
        ax.tick_params(axis="x", rotation=0)
        ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %y"))
        ax.axhline(0, color=AXIS, lw=0.6)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"{asset_label or asset}: daily returns and {alpha:.1%} VaR forecasts",
                 x=0.01, ha="left", fontsize=10, color=INK, fontweight="semibold")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, path)


def fig_cumulative_diff(p, alpha: float, benchmark: str, models: list[str], shade: dict,
                        path: Path) -> None:
    """Cumulative sum of the cross-asset average FZ0 loss differential vs the benchmark."""
    set_style()
    _, _, L, _ = p.slice(alpha)
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for name, (s, e) in shade.items():
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color="#f0efec", lw=0)
        ax.text(pd.Timestamp(s), 1.0, f" {name}", transform=ax.get_xaxis_transform(),
                va="top", ha="left", fontsize=7, color=MUTED)
    ends = []
    for k, m in enumerate(models):
        d = (L[m] - L[benchmark]).groupby(level="date").mean().cumsum()
        ax.plot(d.index, d.to_numpy(), color=SERIES[k], label=m)
        ends.append((m, d.index[-1], d.iloc[-1]))
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_ylabel(f"Cumulative loss difference vs {benchmark}")
    title = (f"Cumulative FZ0 loss difference, alpha = {alpha:.1%} (below zero: model beats {benchmark})"
             if len(models) > 1 else
             f"Cumulative FZ0 loss: {models[0]} minus {benchmark}, alpha = {alpha:.1%} (below zero: {models[0]} better)")
    ax.set_title(title, pad=20 if len(models) > 1 else 8)
    if len(models) > 1:
        _legend_top(ax, len(models))
    ax.margins(x=0.01)
    fig.tight_layout()
    _end_labels(ax, ends)
    _save(fig, path)


def fig_heatmap(rel: pd.DataFrame, title: str, path: Path) -> None:
    """Models x assets heat map of the average FZ0 loss difference vs the benchmark."""
    set_style()
    lim = np.nanmax(np.abs(rel.to_numpy()))
    fig, ax = plt.subplots(figsize=(7.2, 0.28 * len(rel) + 1.2))
    im = ax.imshow(rel.to_numpy(), cmap=DIVERGING, norm=TwoSlopeNorm(0.0, -lim, lim), aspect="auto")
    ax.set_xticks(range(rel.shape[1]), rel.columns, rotation=45, ha="right")
    ax.set_yticks(range(rel.shape[0]), rel.index)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, rel.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, rel.shape[0]), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="both", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.outline.set_visible(False)
    cb.set_label("Loss difference (blue = better)", color=INK2)
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, path)


def fig_news_impact(y: np.ndarray, curves: dict[str, np.ndarray], path: Path, title: str) -> None:
    """News-impact curves: next-day variance as a function of today's standardised return."""
    set_style()
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for k, (name, c) in enumerate(curves.items()):
        ax.plot(y, c, color=SERIES[k], label=name)
    ax.axvline(0, color=AXIS, lw=0.6)
    ax.set_xlabel("Today's standardised return y_t")
    ax.set_ylabel("Next-day variance (standardised)")
    ax.set_title(title, pad=20)
    _legend_top(ax, len(curves))
    ax.margins(x=0)
    fig.tight_layout()
    _save(fig, path)


def fig_tail_ratio(p, alpha: float, models: list[str], path: Path, smooth: int = 21) -> None:
    """Cross-asset median of ES/VaR: a model-implied measure of tail heaviness over time."""
    set_style()
    V, E, _, _ = p.slice(alpha)
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    for k, m in enumerate(models):
        ratio = (E[m] / V[m]).groupby(level="date").median().rolling(smooth, min_periods=1).mean()
        ax.plot(ratio.index, ratio.to_numpy(), color=SERIES[k], label=m)
    from scipy import stats
    q = stats.norm.ppf(alpha)
    gauss = stats.norm.pdf(q) / alpha / -q
    ax.axhline(gauss, color=MUTED, lw=0.8)
    ax.text(0.995, gauss, "Gaussian", transform=ax.get_yaxis_transform(), ha="right", va="top",
            fontsize=7, color=MUTED)
    ax.set_ylabel("ES / VaR")
    ax.set_title(f"Implied tail heaviness: ES/VaR at alpha = {alpha:.1%} (cross-asset median, {smooth}-day mean)",
                 pad=20)
    _legend_top(ax, len(models))
    ax.margins(x=0.01)
    fig.tight_layout()
    _save(fig, path)
