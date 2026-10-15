"""Step 5: copy LaTeX tables and PDF figures of one experiment into paper/ (self-contained for Overleaf).

Usage: python scripts/export_paper.py [--config main]
"""

import argparse
import shutil

import _bootstrap  # noqa: F401

from tailrisk.config import ROOT, load_config, results_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="main")
    cfg = load_config(ap.parse_args().config)
    rdir = results_dir(cfg)
    for sub, pattern in (("tables", "*.tex"), ("figures", "*.pdf")):
        dst = ROOT / "paper" / sub
        dst.mkdir(parents=True, exist_ok=True)
        files = sorted((rdir / sub).glob(pattern))
        for f in files:
            shutil.copy2(f, dst / f.name)
        print(f"copied {len(files)} {sub} -> {dst}")


if __name__ == "__main__":
    main()
