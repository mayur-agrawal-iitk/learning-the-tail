"""Paper 2 figure: learned news-impact curve of NG-FZ vs the GJR-GARCH it starts from.

Trains (or loads) one NG-FZ model for a given refit year and seed, saving its weights, then plots
sigma^2_{t+1} as a function of y_t for one market, with that market's past held at its average
state (Engle & Ng 1993).
Usage: python scripts/plot_news_impact.py [--config main] [--year 2026] [--seed 0] [--asset SPX]
"""

import argparse

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from tailrisk import plots
from tailrisk.config import load_config, results_dir
from tailrisk.data import load_returns
from tailrisk.neural_garch import SOFTPLUS_BETA, NeuralGarch, _job, build_panel, refit_dates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="main")
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--asset", default="SPX")
    args = ap.parse_args()
    cfg = load_config(args.config)
    nc = cfg["neural_garch"]
    rdir = results_dir(cfg)
    (rdir / "ng_models").mkdir(exist_ok=True)
    (rdir / "figures" / "paper2").mkdir(parents=True, exist_ok=True)
    path = rdir / "ng_models" / f"NG-FZ_{args.year}_s{args.seed}.pt"
    rets = load_returns(cfg)
    dates = refit_dates(cfg)
    d0 = next(d for d in dates if d.year == args.year)
    d1 = next((d for d in dates if d > d0), None)
    if not path.exists():
        _job(d0, d1, args.seed, rets, cfg, save_path=path)
    ck = torch.load(path)
    model = NeuralGarch(ck["alphas"], ck["gjr"], d=ck["d"], hidden=ck["hidden"], rho=ck["rho"])
    model.load_state_dict(ck["state"])
    model.eval()
    i = ck["assets"].index(args.asset)
    g = ck["gjr"][i] if isinstance(ck["gjr"], list) else ck["gjr"]
    gi = i if isinstance(ck["gjr"], list) else 0

    panel = build_panel(rets, d0, d1, cfg["window"], nc["burn_in"], nc["val_frac"])
    with torch.no_grad():
        H = model.states(torch.from_numpy(panel["y"]))[i]
        hbar = H[torch.from_numpy(panel["train"][i])].mean(0)       # this market's average state
        y = torch.linspace(-6, 6, 241)
        h_next = model.decay()[gi] * hbar + model.news_impact(y, asset=gi)[0]
        phi = torch.log(F.softplus(h_next, beta=SOFTPLUS_BETA) + 1e-6)
        logvar = phi[:, 0] + model.g_read * model.readout(phi).squeeze(-1)
        ng = torch.exp(logvar).numpy()
    s2bar = float(F.softplus(hbar[0], beta=SOFTPLUS_BETA))
    yn = y.numpy()
    gjr = g["omega"] + g["beta"] * s2bar + (g["alpha"] + g["gamma"] * (yn < 0)) * yn ** 2
    plots.fig_news_impact(yn, {"GJR-GARCH (start of training)": gjr, "NG-FZ (learned)": ng},
                          rdir / "figures" / "paper2" / f"news_impact_{args.asset}_{args.year}",
                          f"News-impact curve, {args.asset}, refit {args.year} (past at its average state)")
    print(pd.DataFrame({"y": yn, "gjr": gjr, "ng": ng}).iloc[::30].round(3).to_string(index=False))
    print("decays:", np.round(model.decay()[gi].detach().numpy(), 4))


if __name__ == "__main__":
    main()
