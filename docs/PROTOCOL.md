# Pre-registered evaluation protocol: paper 2 (neural score-driven tail-risk model)

Written 15 Oct 2026, **before** any model has been run on the holdout markets.

## Why
The 12 markets in `configs/main.yaml` (2006–2026) were used in paper 1, and their results have been
seen. Any new model developed on them could be implicitly tuned to that sample. Claims about the new
model therefore rest on a **holdout set of 12 markets that no model has been run on**.

## Development vs. holdout
| Set | Markets | Use |
|---|---|---|
| Development | SPX, NASDAQ, RUT, FTSE, DAX, CAC, N225, HSI, SENSEX, ASX, KOSPI, IBOV (`configs/main.yaml`) | free to use for designing, debugging and tuning the new model |
| **Holdout** | TSX, SMI, TAIEX, SHCOMP, IPC, AEX, IBEX, KLCI, JKSE, STI, ATX, BEL20 (`configs/holdout.yaml`) | **one** final evaluation with frozen models; until then only data-quality statistics may be computed |

Holdout selection rule (applied before seeing any model output): broad index from Yahoo Finance with
≥ 2000 daily observations before 2006-01-01, no multi-month gaps, fewer than 15 stale (zero-change)
prices. Excluded by this rule: OMXS30 (data start 2008), MERVAL (131 stale prices; hyperinflation),
FTSE MIB (51 stale prices).

## Frozen before the holdout run
1. Model code, architecture and every hyperparameter of the new model(s) (git commit hash recorded).
2. The full benchmark set of paper 1, with unchanged settings.
3. Primary hypothesis: the new model's average FZ0 loss at alpha = 2.5% is lower than GJR-skt's
   (panel Diebold–Mariano test, two-sided, 5%). Secondary: alpha = 1% and 5%; the MCS; adding the
   new model to COMB-All lowers loss.
4. Same sample (OOS 2006-01-02 to 2026-09-30), window (2000), refit schedule and tests as paper 1.

## Development log (development markets only)
| Date | Variant | Change | Dev FZ0 loss vs GJR-skt (2.5%) |
|---|---|---|---|
| 15 Oct 2026 | NG-FZ v1 | pooled GARCH block (median GJR start), shared neural parts | +0.0044 (t = 0.42); 1% hit rate 1.29%; adds to COMB-All (t = −2.5) |
| 15 Oct 2026 | NG-FZ v2 | **asset-specific** GARCH block (each market's own GJR start), shared neural parts | +0.0272 (t = 2.18, significantly worse); no longer adds to COMB-All |
| 15 Oct 2026 | NG-FZ v3 | v2 + auxiliary QLIKE loss for the variance, lambda = 1 fixed a priori; early stopping still on FZ0 | +0.0146 (t = 1.65); 1% hit rate 1.35%; no gain in COMB-All |

**Selection (15 Oct 2026):** v1 has the lowest development loss and is **frozen** as the NG-FZ
model for the holdout. The holdout evaluation is deferred until all paper-2 development is
finished, so that every frozen model is evaluated on the holdout at the same time, exactly once.
No model has been run on the holdout markets.

**Selection rule (fixed before v2 results are seen):** the variant with the lower development
average FZ0 loss at alpha = 2.5% is frozen and is the only NG-FZ variant run on the holdout. No
further variants will be tried after v2 unless the holdout protocol is restarted with fresh markets.

**Amendment 1 (15 Oct 2026, written after seeing v2, before running v3).** The v2 result has a
diagnosable cause: FZ0 only receives information from tail events, so estimating market-specific
volatility dynamics by FZ0 alone is statistically inefficient and overfits (the same mechanism
plausibly explains why the FZ-GAS models of paper 1 underperform GARCH). v3 adds the QLIKE score
for the conditional second moment. The composite score sum_k FZ0_k + lambda * QLIKE is strictly
consistent for (VaR, ES, variance) jointly, so the tail forecasts remain properly elicited.
lambda = 1 is fixed without tuning. The selection rule is extended to {v1, v2, v3}: the lowest
development FZ0 loss at 2.5% goes to the holdout, and **v3 is the last development variant**. The
holdout markets have not been touched by any model; all three variants will be reported.

## Rules
* The holdout evaluation is run **once**. If a bug is found afterwards, it is fixed, the run is
  repeated, and both runs are reported.
* All development-sample results are reported as such, never as out-of-sample evidence.
