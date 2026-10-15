"""Step 4: evaluate all forecasts and write tables (CSV + LaTeX) and figures.

Usage: python scripts/evaluate.py [--config main] [--skip-mcs]
"""

import argparse
import time

import _bootstrap  # noqa: F401
import pandas as pd

from tailrisk import plots
from tailrisk.config import load_config, results_dir
from tailrisk.evaluation import (DEEP, backtest_summary, build_panel, combination_value,
                                 descriptive_stats,
                                 loss_by_asset, loss_summary, mcs_summary, regulatory_table, run_backtests,
                                 run_mcs, seed_robustness, subperiod_table, write_latex)

pd.set_option("display.width", 200)


def tag(a: float) -> str:
    """File-name-safe alpha tag, e.g. 0.025 -> a025."""
    return f"a{round(a * 1000):03d}"

pd.set_option("display.max_columns", 30)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="main")
    ap.add_argument("--skip-mcs", action="store_true")
    ap.add_argument("--with-new", action="store_true",
                    help="include the paper-2 model(s); outputs go to tables/paper2 and figures/paper2")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ev = cfg["evaluation"]
    bench = ev["benchmark"]
    rdir = results_dir(cfg)
    sub = "paper2" if args.with_new else ""
    tdir, fdir = rdir / "tables" / sub, rdir / "figures" / sub
    tdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)
    alphas = cfg["alphas"]
    tic = time.time()

    p = build_panel(cfg, include_new=args.with_new)
    n_assets = p.var.index.get_level_values("asset").nunique()
    print(f"panel: {len(p.var)} rows, {len(p.models)} models, {n_assets} assets")

    # Table 1: descriptive statistics
    ds = descriptive_stats(cfg)
    ds.to_csv(tdir / "descriptive.csv")
    write_latex(ds, tdir / "descriptive.tex", "Descriptive statistics of daily log returns (\\%)",
                "tab:desc", fmt={"N": "{:.0f}", "N (OOS)": "{:.0f}", "Mean": "{:.3f}", "Std": "{:.2f}",
                                 "Skew": "{:.2f}", "Ex. kurt": "{:.1f}", "Min": "{:.1f}", "Max": "{:.1f}",
                                 "Min (OOS)": "{:.1f}"},
                notes="Full sample 1996--2026; OOS = out-of-sample evaluation period from "
                      f"{cfg['oos_start']}.")
    print(ds.round(3))

    # Table 2: FZ0 loss comparison
    for a in alphas:
        ls = loss_summary(p, a, bench)
        ls.to_csv(tdir / f"loss_summary_{tag(a)}.csv")
        write_latex(ls, tdir / f"loss_summary_{tag(a)}.tex",
                    f"Average FZ0 loss, $\\alpha={a:g}$", f"tab:loss_{tag(a)}",
                    fmt={"Avg loss": "{:.4f}", "Diff vs bench": "{:.4f}", "Panel DM t": "{:.2f}",
                         "# better": "{:.0f}", "# worse": "{:.0f}", "Mean rank": "{:.1f}"},
                    bold={"Avg loss": "min", "Mean rank": "min"},
                    notes=f"Avg loss: FZ0 loss averaged over days and then over the {n_assets} assets. "
                          f"Diff vs bench: average loss minus that of {bench} (negative = better); "
                          "Panel DM t: Diebold--Mariano statistic on the cross-asset average loss "
                          "differential (Newey--West). \\# better / worse: assets where the "
                          "single-asset DM test is significant at 5\\%. Mean rank across assets (1 = best).")
        print(f"\nLoss summary alpha={a}\n", ls.round(4))
        lba = loss_by_asset(p, a)
        lba.to_csv(tdir / f"loss_by_asset_{tag(a)}.csv")
        rel = lba.sub(lba.loc[bench], axis=1).drop(index=bench)
        if a == 0.025:
            per = rel.T.copy()
            per.insert(0, f"{bench} (level)", lba.loc[bench])
            per.index.name = "Asset"
            write_latex(per, tdir / f"loss_by_asset_{tag(a)}.tex",
                        f"Average FZ0 loss by asset, $\\alpha={a:g}$: level for {bench}, "
                        "difference for every other model", "tab:perasset", default_fmt="{:.3f}",
                        resize=True, notes="Negative differences = lower loss than the benchmark.")
        plots.fig_heatmap(rel, f"Average FZ0 loss minus {bench}, alpha = {a:.1%}",
                          fdir / f"heatmap_{tag(a)}")

    # Do deep models add information? COMB-All vs COMB-Econ
    if {"COMB-All", "COMB-Econ"} <= set(p.models):
        cv = combination_value(p, alphas)
        cv.to_csv(tdir / "combination_value.csv")
        write_latex(cv, tdir / "combination_value.tex",
                    "Value of adding the deep models to an econometric forecast combination",
                    "tab:combo", fmt={"Loss diff": "{:.4f}", "Panel DM t": "{:.2f}", "p-value": "{:.3f}",
                                      "# assets better": "{:.0f}", "# assets worse": "{:.0f}"},
                    notes="Loss difference COMB-All minus COMB-Econ (negative = adding the three deep "
                          "models helps). The two combinations were specified before any results were "
                          "seen and differ only by the inclusion of the deep models.")
        print("\nCombination value (COMB-All vs COMB-Econ)\n", cv.round(4))
    if {"COMB-All+NG", "COMB-All"} <= set(p.models):
        cv2 = combination_value(p, alphas, with_deep="COMB-All+NG", without="COMB-All")
        cv2.to_csv(tdir / "combination_value_ng.csv")
        write_latex(cv2, tdir / "combination_value_ng.tex",
                    "Value of adding NG-FZ to the paper-1 combination", "tab:combo_ng",
                    fmt={"Loss diff": "{:.4f}", "Panel DM t": "{:.2f}", "p-value": "{:.3f}",
                         "# assets better": "{:.0f}", "# assets worse": "{:.0f}"},
                    notes="Loss difference COMB-All+NG minus COMB-All (negative = adding NG-FZ helps).")
        print("\nCombination value (COMB-All+NG vs COMB-All)\n", cv2.round(4))
    # Paper 2 pre-registered hypotheses (docs/PROTOCOL.md, amendment 2)
    hyps = {"H1: CTG-FZ vs GJRX-skt": ("CTG-FZ", "GJRX-skt"),
            "H2: CTG-FZ vs LSTM-FZ": ("CTG-FZ", "LSTM-FZ"),
            "H3: COMB-All+X vs COMB-All": ("COMB-All+X", "COMB-All"),
            "Ablation: CTG-FZ vs LSTMQ-FZ": ("CTG-FZ", "LSTMQ-FZ"),
            "Info: GJRX-skt vs GJR-skt": ("GJRX-skt", "GJR-skt")}
    rows = []
    for name, (m1, m2) in hyps.items():
        if {m1, m2} <= set(p.models):
            t = combination_value(p, alphas, with_deep=m1, without=m2).reset_index()
            t.insert(0, "Test", name)
            rows.append(t)
    if rows:
        ht = pd.concat(rows, ignore_index=True).set_index(["Test", "Level"])
        ht.to_csv(tdir / "hypotheses.csv")
        flat = ht.reset_index()
        flat.index = [f"{a} ({b})" for a, b in zip(flat["Test"], flat["Level"])]
        flat.index.name = "Test (level)"
        write_latex(flat.drop(columns=["Test", "Level"]), tdir / "hypotheses.tex",
                    "Pre-registered hypothesis tests (FZ0 loss of first model minus second)", "tab:hyp",
                    fmt={"Loss diff": "{:.4f}", "Panel DM t": "{:.2f}", "p-value": "{:.3f}",
                         "# assets better": "{:.0f}", "# assets worse": "{:.0f}"},
                    notes="Negative loss difference = the first model is more accurate. Panel "
                          "Diebold--Mariano test on the cross-asset average loss differential.")
        print("\nPre-registered hypotheses\n", ht.round(4))

    # Tables 3-4: backtests
    bt = run_backtests(p, alphas, ev["bootstrap_reps"])
    bt.to_csv(tdir / "backtests_detail.csv", index=False)
    bs = backtest_summary(bt, p.models)
    bs.to_csv(tdir / "backtest_summary.csv")
    for a in alphas:
        t = bs.xs(a, level="alpha")
        write_latex(t, tdir / f"backtests_{tag(a)}.tex",
                    f"Backtests, $\\alpha={a:g}$: number of assets (out of {n_assets}) rejecting at 5\\%",
                    f"tab:bt_{tag(a)}",
                    fmt={"Hit %": "{:.2f}", "Z2": "{:.3f}", **{c: "{:.0f}" for c in t.columns[1:-1]}},
                    notes=f"Hit \\%: average violation rate (nominal {100 * a:g}\\%). VaR tests: UC Kupiec, "
                          "CC Christoffersen conditional coverage, DQ Engle--Manganelli. ES tests: ER "
                          "McNeil--Frey exceedance residuals (one-sided bootstrap), CCT-u/CCT-c Nolde--Ziegel "
                          "unconditional/conditional calibration. Z2: Acerbi--Szekely statistic averaged "
                          "over assets (negative = ES too optimistic).")
    print("\nBacktest summary\n", bs.round(3))

    # Table 5: model confidence set
    if not args.skip_mcs:
        mcs = run_mcs(p, alphas, ev["mcs_size"], ev["mcs_reps"])
        mcs.to_csv(tdir / "mcs_detail.csv", index=False)
        ms = mcs_summary(mcs, p.models, ev["mcs_size"])
        ms.to_csv(tdir / "mcs_summary.csv")
        write_latex(ms, tdir / "mcs_summary.tex",
                    f"Model confidence set ({100 * (1 - ev['mcs_size']):.0f}\\%) membership",
                    "tab:mcs", default_fmt="{:.0f}", resize=True,
                    fmt={c: "{:.3f}" for c in ms.columns if c.startswith("Pooled")},
                    notes="Columns '\\# assets': number of assets whose MCS contains the model. "
                          "'Pooled p': MCS p-value on the cross-asset average loss series; a model is "
                          f"in the pooled MCS if p > {ev['mcs_size']}. Range statistic, stationary "
                          f"bootstrap, {ev['mcs_reps']} replications.")
        print("\nMCS summary\n", ms.round(3))

    # Table 6: subperiods (headline alpha = 2.5%)
    a_main = 0.025 if 0.025 in alphas else alphas[0]
    diff, tst = subperiod_table(p, a_main, bench, ev["subperiods"])
    diff.to_csv(tdir / "subperiod_diff.csv")
    tst.to_csv(tdir / "subperiod_t.csv")
    comb = diff.copy().astype(object)
    for c in diff.columns:
        comb[c] = [f"{d:.3f} ({t:.1f})" for d, t in zip(diff[c], tst[c])]
    write_latex(comb, tdir / "subperiods.tex",
                f"Loss difference vs {bench} by subperiod, $\\alpha={a_main:g}$", "tab:sub",
                notes="Average FZ0 loss difference (negative = better than the benchmark); panel "
                      "Diebold--Mariano t-statistic in parentheses.")
    print("\nSubperiods (diff)\n", diff.round(3), "\n(t)\n", tst.round(2))

    # Regulatory view: capital proxy and Basel traffic light
    if 0.01 in alphas and 0.025 in alphas:
        reg = regulatory_table(p)
        reg.to_csv(tdir / "regulatory.csv")
        write_latex(reg, tdir / "regulatory.tex", "Regulatory view: capital and Basel traffic light",
                    "tab:reg", fmt={"Capital (avg -ES 2.5%)": "{:.2f}", "Yellow zone % days": "{:.1f}",
                                    "Red zone % days": "{:.1f}"},
                    notes="Capital: average forecast of $-\\mathrm{ES}_{2.5\\%}$ (FRTB measure), in \\% of "
                          "position value, averaged over assets. Traffic light: share of days on which "
                          "the trailing 250-day count of 99\\% VaR violations is 5--9 (yellow) or "
                          "$\\geq$10 (red).")
        print("\nRegulatory view\n", reg.round(2))

    # Table 7: seed robustness of deep models
    sr = seed_robustness(cfg, p, a_main)
    if not sr.empty:
        sr.to_csv(tdir / "seed_robustness.csv")
        write_latex(sr, tdir / "seed_robustness.tex",
                    f"Deep models: single seeds versus the seed ensemble, $\\alpha={a_main:g}$",
                    "tab:seeds", default_fmt="{:.4f}")
        print("\nSeed robustness\n", sr.round(4))

    # Figures
    deep_in = [m for m in (["LSTM-FZ", "GJRX-skt", "CTG-FZ"] if args.with_new else DEEP) if m in p.models]
    assets = p.var.index.get_level_values("asset").unique()
    pref = "N225" if args.with_new else "SPX"          # paper 2: overnight news matters most for Asia
    show = pref if pref in assets else assets[0]
    eps = {k: v for k, v in {"Global financial crisis": ("2008-08-01", "2009-04-30"),
                             "COVID-19": ("2020-02-01", "2020-06-30")}.items()
           if pd.Timestamp(v[1]) >= p.var.index.get_level_values("date").min()}
    focus = [bench] + [m for m in (("GJRX-skt", "CTG-FZ") if args.with_new else ("FZ-Hyb", "GJR-LSTM"))
                       if m in p.models]
    label = {"SPX": "S&P 500", "N225": "Nikkei 225"}.get(show, show)
    plots.fig_episodes(p, show, a_main, focus, eps, fdir / "episodes", label)
    if {"CTG-FZ", "GJRX-skt"} <= set(p.models):
        plots.fig_cumulative_diff(p, a_main, "GJRX-skt", ["CTG-FZ"],
                                  {k: v for k, v in ev["subperiods"].items() if not k.startswith("Calm")},
                                  fdir / "cumulative_ctg_vs_gjrx")
    plots.fig_cumulative_diff(p, a_main, bench, deep_in,
                              {k: v for k, v in ev["subperiods"].items() if not k.startswith("Calm")},
                              fdir / "cumulative_diff")
    if {"COMB-All", "COMB-Econ"} <= set(p.models):
        plots.fig_cumulative_diff(p, a_main, "COMB-Econ", ["COMB-All"],
                                  {k: v for k, v in ev["subperiods"].items() if not k.startswith("Calm")},
                                  fdir / "cumulative_combo")
    if {"COMB-All+NG", "COMB-All"} <= set(p.models):
        plots.fig_cumulative_diff(p, a_main, "COMB-All", ["COMB-All+NG"],
                                  {k: v for k, v in ev["subperiods"].items() if not k.startswith("Calm")},
                                  fdir / "cumulative_combo_ng")
    tail_models = ("GJR-skt", "GJR-LSTM", "NG-FZ") if args.with_new else ("GARCH-t", "GJR-skt", "GJR-LSTM")
    plots.fig_tail_ratio(p, a_main, [m for m in tail_models if m in p.models], fdir / "tail_ratio")
    print(f"\nevaluation done in {(time.time() - tic) / 60:.1f} min -> {rdir}")


if __name__ == "__main__":
    main()
