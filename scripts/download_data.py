"""Step 1: download raw prices and print a data-quality report.

Usage: python scripts/download_data.py [--config main]
"""

import argparse

import _bootstrap  # noqa: F401

from tailrisk.config import load_config
from tailrisk.data import download_all, load_returns


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="main")
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if not args.skip_download:
        download_all(cfg)
    print("\nData-quality report (cleaned log returns, %):")
    rets = load_returns(cfg, verbose=True)
    oos = cfg["oos_start"]
    for name, r in rets.items():
        n_pre = int((r.index < oos).sum())
        flag = "" if n_pre >= cfg["window"] else f"  <-- only {n_pre} obs before {oos}"
        print(f"{name:8s} first={r.index[0].date()} obs_before_oos={n_pre}{flag}")


if __name__ == "__main__":
    main()
