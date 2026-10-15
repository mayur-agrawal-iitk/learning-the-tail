# Learning the Tail: Deep Joint Forecasting of VaR and Expected Shortfall

**Mayur Agrawal**, Department of Computer Science and Engineering and Department of Economic Sciences,
IIT Kanpur ([mayurag22@iitk.ac.in](mailto:mayurag22@iitk.ac.in)).
Paper: [`paper/main.pdf`](paper/main.pdf).

Research project comparing deep sequence models with state-of-the-art econometric models for
forecasting one-day **Value-at-Risk** and **Expected Shortfall** at α ∈ {1%, 2.5%, 5%} on 12 global equity
indices, 2006–2026 out of sample.

The core idea: Expected Shortfall is not elicitable on its own, but the pair (VaR, ES) is
(Fissler & Ziegel, 2016). The strictly consistent **FZ0 loss** (Patton, Ziegel & Chen, 2019) can therefore
be used as a training objective for neural networks, and as the basis for comparing forecasts.

## Models

| Family | Models |
|---|---|
| Historical | HS (250 days) |
| Location-scale (MLE) | GARCH-N, GARCH-t, GJR-GARCH skewed-t |
| Semi-parametric tails | Filtered HS, GARCH-EVT (generalised Pareto tail) |
| Dynamic semiparametric (FZ0 M-estimation) | CAViaR-ES, FZ-GAS one-factor, FZ-GAS hybrid |
| **Deep (FZ0-trained, pooled across assets)** | **LSTM-FZ, Transformer-FZ, GJR-GARCH + LSTM hybrid** |
| Combinations (pre-specified) | COMB-Econ, COMB-All |

The deep models use a monotone output head that guarantees ES ≤ VaR < 0 and no crossing across α.

## Evaluation

* VaR backtests: Kupiec, Christoffersen conditional coverage, Engle–Manganelli DQ.
* ES backtests: McNeil–Frey exceedance residuals, Acerbi–Szekely Z2, and the Nolde–Ziegel conditional calibration test.
* Comparisons: FZ0 loss, Diebold–Mariano tests (per asset and panel), Model Confidence Set, crisis subperiods.

## Layout

```
configs/          main.yaml (full study), quick.yaml (3-asset smoke test)
src/tailrisk/     data, scoring (FZ0), backtests, comparison (DM, MCS), engine,
                  models/ (HS, GARCH family + FHS + EVT, CAViaR/FZ-GAS), neural, evaluation, plots
scripts/          download_data -> run_econometric -> run_neural -> evaluate -> export_paper
tests/            unit tests (consistency, test sizes, arch alignment, no look-ahead)
results/<config>/ forecasts (parquet), tables (csv + tex), figures (pdf + png)
paper/            LaTeX paper (main.tex, references.bib)
```

## Reproduce

```bash
pip install -r requirements.txt
# GPU (CUDA 11.x drivers, e.g. GTX 1650 with driver 512):
#   python -m venv .venv && .venv\Scripts\pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
#   then install the rest of requirements.txt into .venv

python scripts/download_data.py   --config main   # ~1 min
python scripts/run_econometric.py --config main   # ~25 min on 12 cores
python scripts/run_neural.py      --config main   # GPU or CPU, see configs/main.yaml: neural.device
python scripts/evaluate.py        --config main   # tables + figures
python scripts/export_paper.py    --config main   # copy into paper/ for Overleaf
python -m pytest                                  # unit tests
```

Use `--config quick` for a smoke test of the whole pipeline in a few minutes.

## Design choices worth knowing (for interviews)

* **Same information set for every model**: daily returns only. No range or realised-volatility data,
  so any difference comes from the model, not the data.
* **No look-ahead**: forecasts for day *t* use returns up to *t−1*; unit tests perturb future data for every model.
* **Scale-standardised pooled training is exact**, not an approximation: L(cy, cv, ce) = L(y, v, e) + log c.
  The target is r/σ, deliberately *not* demeaned, because FZ0 is not translation-equivariant.
* **Sign restrictions in the GAS models** (γ, δ ≥ 0). Without them the 1% models occasionally learned
  "risk falls after a violation", which collapsed VaR to zero in the COVID crash. Documented in the paper.
* **Conditional calibration test instruments**: the lagged hit is excluded. It makes the test badly oversized
  (68% rejections at a nominal 5%); see `src/tailrisk/backtests.py`.
* **Hyperparameters were fixed before the out-of-sample evaluation**; combinations were pre-specified.
