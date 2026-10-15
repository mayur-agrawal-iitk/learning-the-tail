# learning-the-tail

Code for *Learning the Tail: Deep Joint Forecasting of Value-at-Risk and Expected Shortfall with
Strictly Consistent Scoring Functions* by Mayur Agrawal (IIT Kanpur).

The paper is in `paper/main.pdf`.

## Setup

    pip install -r requirements.txt

## Running

    python scripts/download_data.py
    python scripts/run_econometric.py
    python scripts/run_neural.py
    python scripts/evaluate.py
    python scripts/build_paper.py

Add `--config quick` to any step for a small three-index version that runs in a few minutes.
Tests: `python -m pytest`.

Price data are downloaded from Yahoo Finance and are not included in the repository.

Follow-up work on cross-market models uses a set of holdout markets; the protocol is in
`docs/PROTOCOL.md`.
