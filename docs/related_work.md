# Closest related work (novelty check, 14 Oct 2026)

Status: every entry cited in paper 1 was verified against Crossref (DOI registry) or the arXiv API on 15 Oct 2026, including the claims made about it. Entries 5, 7 and 8 are not cited and remain unverified.
Verify every **S** entry (authors, venue, year, DOI) on the publisher page before citing.

| # | Paper | Status | What they do | How we differ |
|---|---|---|---|---|
| 1 | Li, Tran, Wang, Gerlach, Gao (2020/21), *A Bayesian Long Short-Term Memory Model for Value at Risk and Expected Shortfall Joint Forecasting*, arXiv:2001.08374 | V | LSTM inside an ES-CAViaR / asymmetric-Laplace framework, Bayesian MCMC | FZ0 empirical-risk training, pooled, hybrid tail-shape model, MCS + calibration tests |
| 2 | *VaR and ES forecasting via recurrent neural network-based stateful models*, ScienceDirect PII S1057521924000346 (2024; ISSN prefix suggests *International Review of Financial Analysis*) | S | stateful RNN/FNN models, joint VaR/ES, FZ0 evaluation, 6 assets | 12 markets, 20 years OOS, stronger benchmarks (FZ-GAS, EVT), hybrid, theory |
| 3 | Zhang, Zhu, Zhu (2026), *ReSGA: A Large Tail Risk Model for Learning Value-at-Risk and Expected Shortfall*, arXiv:2606.04576 | V | large model, **monthly US stocks** with 153 firm characteristics, cross-sectional | daily index-level risk, returns-only information set, regulatory horizon |
| 4 | Amjad, Hassan, Ahmed (2026), *Neural and Econometric Forecasting of Market Risk: A Comparative VaR and ES Analysis Across Global Equity Markets*, *Risks* 14(10), 228, doi:10.3390/risks14100228 | S | **12 global indices 2000–2026**; HS vs GARCH-FHS vs LSTM; GARCH beats LSTM (QLIKE DM) in 10/12 | our LSTMs are trained on the joint (VaR, ES) FZ0 loss, not a volatility loss; 9 benchmarks; hybrid tail shape |
| 5 | *Does Machine Learning Beat the GARCH Benchmark? Historical, Filtered, and Neural Tail-Risk Measures Across Thirty-Nine Global Equity Markets*, MDPI *Economies* 14(9), 379 (2026) | S (authors unknown) | neural tail-risk models show ES bias vs FHS; suggest training under jointly elicitable scores | we do exactly that |
| 6 | *A dynamic factor semiparametric model for VaR and ES driven by realized measures*, arXiv:2601.01142 | S | notes ES/VaR is constant in parametric GARCH models | our hybrid learns a time-varying ES/VaR (tail shape) |
| 7 | *Deep neural expected shortfall regression with tail-robustness*, arXiv:2511.08772 | S | deep ES regression via orthogonal scores | regression with covariates vs dynamic forecasting |
| 8 | *Combining Value-at-Risk and Expected Shortfall forecasts via the Model Confidence Set*, arXiv:2406.06235 | S | MCS-based combination, weights by FZ loss | we use simple pre-specified combinations; could extend |
| 9 | *Combining deep learning and GARCH models for volatility*, arXiv:2310.01063 | S | GARCH–GRU hybrids improve volatility but not VaR/ES | our hybrid targets the tail shape directly with an ES-consistent loss |
| 10 | Kelly & Jiang (2014), *Tail Risk and Asset Prices*, *Review of Financial Studies* | S (well known) | time-varying tail risk is priced | basis for the planned economic extension |

**Positioning sentence for the paper:** existing neural VaR/ES work either trains on volatility losses
(and loses to GARCH, #4) or evaluates few assets against weak benchmarks (#2). We train on the
strictly consistent joint loss, prove pooling is exact, propose a hybrid that learns time-varying tail
shape (#6, #9), and evaluate against the strongest semiparametric benchmarks with MCS and calibration tests.
