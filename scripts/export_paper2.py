"""Copy development- and holdout-sample tables/figures into paper2/, with sample-specific labels.

Usage: python scripts/export_paper2.py
"""

import shutil

import _bootstrap  # noqa: F401

from tailrisk.config import ROOT

SRC = {"dev": ROOT / "results" / "main", "ho": ROOT / "results" / "holdout"}
NAME = {"dev": "Development sample", "ho": "Holdout sample"}
TABLES = {"dev": ["hypotheses", "loss_summary_a025"],
          "ho": ["hypotheses", "loss_summary_a010", "loss_summary_a025", "loss_summary_a050",
                 "mcs_summary", "backtests_a025", "regulatory", "subperiods"]}
FIGURES = {"ho": ["heatmap_a025", "cumulative_diff", "cumulative_ctg_vs_gjrx", "episodes"]}


def main() -> None:
    out_t, out_f = ROOT / "paper2" / "tables", ROOT / "paper2" / "figures"
    out_t.mkdir(parents=True, exist_ok=True)
    out_f.mkdir(parents=True, exist_ok=True)
    for s, names in TABLES.items():
        for n in names:
            tex = (SRC[s] / "tables" / "paper2" / f"{n}.tex").read_text(encoding="utf-8")
            tex = tex.replace("\\label{tab:", f"\\label{{tab:{s}_").replace("\\caption{", f"\\caption{{{NAME[s]}: ", 1)
            (out_t / f"{s}_{n}.tex").write_text(tex, encoding="utf-8")
    for s, names in FIGURES.items():
        for n in names:
            shutil.copy2(SRC[s] / "figures" / "paper2" / f"{n}.pdf", out_f / f"{s}_{n}.pdf")
    for f in (SRC["ho"] / "figures" / "paper2").glob("news_impact_*.pdf"):
        shutil.copy2(f, out_f / f"ho_{f.name}")
    print("exported to paper2/tables and paper2/figures")


if __name__ == "__main__":
    main()
