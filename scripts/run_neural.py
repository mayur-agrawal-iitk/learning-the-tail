"""Step 3: pooled deep models trained with the FZ0 loss (yearly refits, seed ensemble).

Usage: python scripts/run_neural.py [--config main] [--models LSTM-FZ GJR-LSTM]
Writes results/<config>/forecasts/neural_<model>.parquet (+ per-seed files and a training log)
"""

import argparse
import time

import _bootstrap  # noqa: F401

from tailrisk.config import load_config
from tailrisk.neural import run_neural


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="main")
    ap.add_argument("--models", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config)
    tic = time.time()
    run_neural(cfg, args.models)
    print(f"done in {(time.time() - tic) / 60:.1f} min")


if __name__ == "__main__":
    main()
