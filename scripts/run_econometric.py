"""Step 2: rolling out-of-sample forecasts for the econometric benchmarks.

Usage: python scripts/run_econometric.py [--config main] [--assets SPX DAX] [--models FZ-1F FZ-Hyb]
Writes results/<config>/forecasts/econ_<asset>.parquet
"""

import argparse
import time

import _bootstrap  # noqa: F401

from tailrisk.config import load_config
from tailrisk.engine import run_econometric


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="main")
    ap.add_argument("--assets", nargs="*")
    ap.add_argument("--models", nargs="*", help="re-run only these models and merge into existing files")
    args = ap.parse_args()
    cfg = load_config(args.config)
    tic = time.time()
    run_econometric(cfg, args.assets, args.models)
    print(f"done in {(time.time() - tic) / 60:.1f} min")


if __name__ == "__main__":
    main()
