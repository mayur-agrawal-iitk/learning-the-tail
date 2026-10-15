"""Paper 2: train the neural news-impact GARCH (NG-FZ), yearly refits, seed ensemble.

Usage: python scripts/run_neural_garch.py [--config main]
Writes results/<config>/forecasts/neural_NG-FZ.parquet (+ per-seed file, training log)
"""

import argparse
import time

import _bootstrap  # noqa: F401

from tailrisk.config import load_config
from tailrisk.neural_garch import run_neural_garch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="main")
    cfg = load_config(ap.parse_args().config)
    tic = time.time()
    run_neural_garch(cfg)
    print(f"done in {(time.time() - tic) / 60:.1f} min")


if __name__ == "__main__":
    main()
