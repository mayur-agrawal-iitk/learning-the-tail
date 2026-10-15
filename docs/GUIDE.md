# Project guide: what everything is and how to defend it

Read this alongside the code. Each section points to the file that implements it.

---

## 1. The research question in one breath

> Basel moved market-risk capital from 99% VaR to 97.5% Expected Shortfall. ES on its own cannot be
> learned by minimising a loss, but VaR and ES *together* can. So we train neural networks directly on a
> loss that is provably minimised by the true (VaR, ES). Then we ask, with proper statistical tests,
> whether they beat the best econometric models out of sample.

---

## 2. The math you must be able to explain

### 2.1 VaR and ES (lower tail, signs kept)
* `VaR_t = v_t` = the α-quantile of tomorrow's return. At α = 2.5%, 2.5% of days should be worse than v_t.
* `ES_t = e_t` = the average return on those bad days: E[r_t | r_t ≤ v_t]. Always e_t ≤ v_t < 0.

### 2.2 Elicitability (the key idea)
A functional is *elicitable* if some loss function has its expected value minimised exactly at it.
* Mean: squared error. Median: absolute error. Quantile (VaR): the pinball loss.
* ES: **no such loss exists** (Gneiting 2011). Intuition: elicitable functionals have convex
  "level sets", meaning if F and G have the same ES then so does any mixture of them. ES fails this.
* But the **pair (VaR, ES)** is elicitable (Fissler & Ziegel 2016). We use the FZ0 member:

  `L(y, v, e) = -1/(α e) · 1{y ≤ v}(v − y) + v/e + log(−e) − 1`

  Code: `src/tailrisk/scoring.py` → `fz0_loss`.

### 2.3 The three propositions in the paper (and their one-line proofs)
1. **FZ0 is minimised at the truth.** For fixed e, the v-part is a positive multiple of the pinball
   loss, so it is minimised at the quantile regardless of e. Plugging that in leaves
   `c/x + log x` (with x = −e and c = −ES), which is minimised at x = c. The minimum value is log(−ES).
2. **Scale property.** `L(cy, cv, ce) = L(y, v, e) + log c`. So training on returns divided by a
   volatility estimate is *exactly* the same optimisation as training on raw returns. This is why we can
   pool 12 markets with very different volatilities into one network. FZ0 is *not* translation
   invariant, which is why the hybrid's target is r/σ and not (r − μ)/σ.
3. **The output head can't cross.** VaR and ES are built as negative cumulative sums of softplus
   gaps, so ES ≤ VaR < 0 and the 1%, 2.5% and 5% levels are ordered for any weights.
   Code: `MonotoneTailHead` in `src/tailrisk/neural.py`.

### 2.4 FZ-GAS (Patton, Ziegel & Chen 2019)
`v_t = a·exp(κ_t)`, `e_t = b·exp(κ_t)`. κ is driven by λ_t = −∂L/∂κ = (1/α)·1{hit}·r/e − 1.
On quiet days λ = −1 and risk decays slowly. After a violation λ jumps, roughly by 1/α times the loss
relative to ES. You should be able to derive ∂L/∂κ yourself (Lemma 1 in the paper; a few lines of
calculus).

---

## 3. The models (one line each)

| Model | Idea | File |
|---|---|---|
| HS | quantile / tail mean of the last 250 returns | `models/historical.py` |
| GARCH-N / -t | volatility clustering, normal / Student-t shocks | `models/garch.py` |
| GJR-skt | adds the leverage effect (bad news raises volatility more) and a skewed-t tail | `models/garch.py` |
| FHS | GARCH filter + empirical distribution of standardised residuals | `models/garch.py` |
| GARCH-EVT | GARCH filter + generalised Pareto tail (extreme value theory) | `models/garch.py` |
| CAViaR-ES | VaR follows its own autoregression; ES is a multiple of VaR | `models/semiparametric.py` |
| FZ-1F / FZ-Hyb | score-driven VaR/ES from the FZ0 loss itself | `models/semiparametric.py` |
| LSTM-FZ / TRF-FZ | sequence model on 64 days of scaled returns, trained on FZ0 | `neural.py` |
| GJR-LSTM | GARCH sets the scale; the LSTM learns the time-varying *shape* of the tail | `neural.py` |
| COMB-* | equal-weight averages, fixed before seeing results | `evaluation.py` |

---

## 4. Evaluation (what each test asks)

| Test | Question | Null rejected means |
|---|---|---|
| Kupiec (UC) | Is the violation rate equal to α? | too many or too few violations |
| Christoffersen (CC) | Correct rate *and* violations not clustered? | violations come in bunches |
| DQ | Can past violations or the VaR level predict new violations? | forecasts react too slowly |
| ER (McNeil–Frey) | On violation days, are losses as bad as ES predicted? | ES too optimistic |
| Z2 (Acerbi–Szekely) | Summary statistic, < 0 means ES is too optimistic | — |
| CCT (Nolde–Ziegel) | Is the identification function unpredictable from past information? | miscalibrated VaR/ES |
| Diebold–Mariano | Is model A's average FZ0 loss lower than B's? | significant difference |
| Model Confidence Set | Which models can't be statistically beaten? | model eliminated |

**Calibration vs comparison.** Backtests ask "is this model *acceptable*?". FZ0, DM and MCS ask
"which model is *better*?". A regulator needs the first and a model selector needs the second
(Nolde & Ziegel 2017).

---

## 5. Likely interview questions (with short answers)

**Why not just train a network with MSE on returns?** MSE elicits the mean, not tail risk. For ES you
need a strictly consistent scoring function for (VaR, ES), and FZ0 is one.

**Why FZ0 and not another member of the FZ class?** It is the member whose loss differences are
scale-free (Proposition 2), so it allows pooling assets and comparing across assets. Rankings can
depend on the member chosen (Ziegel et al. 2020). Murphy diagrams would be a natural extension.

**Isn't the comparison unfair to the neural nets / to GARCH?** Same information (returns only), same
8-year window, no look-ahead (unit-tested), hyperparameters fixed in advance. The deep models are
re-trained yearly while GARCH is re-estimated monthly, which is conservative *against* the deep models.

**Why pool across countries?** A single index has about 2,000 days per window, so at 1% only about 20
tail events. Pooling 12 markets gives the network about 240 tail events, and Proposition 2 makes pooling exact.

**What went wrong during the project?** (Good answer, because it shows judgment.)
(1) Unconstrained FZ-GAS at 1% learned "risk falls after a violation" and collapsed VaR to zero in COVID.
It was fixed with economically motivated sign restrictions.
(2) The calibration test with the lagged hit as an instrument was oversized (68% rejection at a nominal 5%)
because two of its moments are collinear except on consecutive violations. That instrument was removed.

**What would you do next?** Add realised measures or intraday range as inputs (information versus
architecture); multi-day horizons; Murphy diagrams; the regression-based ES backtest of
Bayer & Dimitriadis (2022); portfolio-level (multivariate) ES.

---

## 6. How to run things

See `README.md`. `--config quick` runs the whole pipeline on 3 assets in a few minutes, which is good
for experimenting.
