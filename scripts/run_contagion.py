"""Paper 2: CTG-FZ, the LSTM tail-risk model with a learned cross-market contagion module.

Usage: python scripts/run_contagion.py [--config main] [--model CTG-FZ | LSTMQ-FZ]
Writes results/<config>/forecasts/neural_CTG-FZ.parquet (+ per-seed file, training log)
"""

import argparse
import time

import _bootstrap  # noqa: F401

from tailrisk.config import load_config
from tailrisk.contagion import run_contagion


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="main")
    ap.add_argument("--model", default="CTG-FZ", help="variant name in the config's contagion section")
    args = ap.parse_args()
    cfg = load_config(args.config)
    tic = time.time()
    run_contagion(cfg, args.model)
    print(f"done in {(time.time() - tic) / 60:.1f} min")


if __name__ == "__main__":
    main()
