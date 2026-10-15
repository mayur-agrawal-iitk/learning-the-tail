"""Appendix table: first run (v1) vs final specification, average FZ0 loss of the revised models.

v1 = Gaussian head initialisation for the deep models and grid + Nelder-Mead estimation for the
semiparametric models (archived in results/main/v1_gaussian_init/). Final = empirical head
initialisation and multi-start + Powell estimation.
Usage: python scripts/compare_revisions.py
"""

import _bootstrap  # noqa: F401
import pandas as pd

from tailrisk.config import load_config, results_dir
from tailrisk.evaluation import write_latex

REVISED = ["CAViaR-ES", "FZ-1F", "FZ-Hyb", "LSTM-FZ", "TRF-FZ", "GJR-LSTM", "COMB-Econ", "COMB-All"]


def main() -> None:
    rdir = results_dir(load_config("main"))
    rows = {}
    for tag, a in (("a010", "1%"), ("a025", "2.5%"), ("a050", "5%")):
        v1 = pd.read_csv(rdir / "v1_gaussian_init" / "tables_full" / f"loss_summary_{tag}.csv", index_col=0)
        v2 = pd.read_csv(rdir / "tables" / f"loss_summary_{tag}.csv", index_col=0)
        rows[f"v1 {a}"] = v1.loc[REVISED, "Diff vs bench"]
        rows[f"final {a}"] = v2.loc[REVISED, "Diff vs bench"]
    t = pd.DataFrame(rows)
    t.index.name = "Model"
    t.to_csv(rdir / "tables" / "revisions.csv")
    write_latex(t, rdir / "tables" / "revisions.tex",
                "First run (v1) versus final specification: FZ0 loss minus GJR-skt", "tab:revisions",
                default_fmt="{:.4f}",
                notes="v1: Gaussian initialisation of the deep models' output head; grid + Nelder--Mead "
                      "estimation of the semiparametric models. Final: empirical initialisation; "
                      "multi-start + Powell estimation. Unchanged models are omitted. Negative = better "
                      "than GJR-skt.")
    print(t.round(4))


if __name__ == "__main__":
    main()
