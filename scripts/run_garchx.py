"""Paper 2: GJR-GARCH-X (skewed-t) with pre-open cross-market news, rolling forecasts.

Usage: python scripts/run_garchx.py [--config main]
Writes results/<config>/forecasts/xm_<asset>.parquet
"""

import argparse
import time

import _bootstrap  # noqa: F401
from joblib import Parallel, delayed

from tailrisk.config import load_config, results_dir
from tailrisk.crossmarket import overnight_news
from tailrisk.data import load_returns
from tailrisk.engine import run_asset
from tailrisk.models.garchx import GJRXSkt, news_regressors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="main")
    cfg = load_config(ap.parse_args().config)
    tic = time.time()
    rets = load_returns(cfg)
    news = overnight_news(rets)
    jobs = []
    for a, r in rets.items():
        nw, av = news[a]
        x1, x2 = news_regressors(nw.to_numpy(), av.to_numpy())
        jobs.append(delayed(run_asset)(a, r, cfg, estimators=[GJRXSkt(x1, x2)]))
    out = results_dir(cfg) / "forecasts"
    for a, df in zip(rets, Parallel(n_jobs=cfg["econometric"]["n_jobs"], verbose=5)(jobs)):
        df.to_parquet(out / f"xm_{a}.parquet", index=False)
    print(f"done in {(time.time() - tic) / 60:.1f} min")


if __name__ == "__main__":
    main()
