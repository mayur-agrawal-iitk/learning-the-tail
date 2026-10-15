#!/usr/bin/env bash
# One-time holdout evaluation of the frozen models (docs/PROTOCOL.md). Steps run sequentially
# because of the 7.5 GB RAM limit; each step is resumable / checkpointed.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=python
VENV=.venv/Scripts/python.exe          # CUDA build of torch for the GPU-capable steps
[ -x "$VENV" ] || VENV=python
step() { echo "[$(date +%H:%M:%S)] $*"; }

step "1/7 benchmarks (fast models)"
$PY scripts/run_econometric.py --config holdout --models HS GARCH-N GARCH-t GJR-skt FHS GARCH-EVT
step "2/7 benchmarks (semiparametric models)"
$PY scripts/run_econometric.py --config holdout --models CAViaR-ES FZ-1F FZ-Hyb
step "3/7 GJR-X with cross-market news"
$PY scripts/run_garchx.py --config holdout
step "4/7 paper-1 deep models"
rm -f results/holdout/.gpu_slots/*.lock
$VENV scripts/run_neural.py --config holdout
step "5/7 NG-FZ v1"
$PY scripts/run_neural_garch.py --config holdout
step "6/7 CTG-FZ v2 and LSTMQ-FZ ablation"
$VENV scripts/run_contagion.py --config holdout --model CTG-FZ
$VENV scripts/run_contagion.py --config holdout --model LSTMQ-FZ
step "7/7 evaluation"
$PY scripts/evaluate.py --config holdout
$PY scripts/evaluate.py --config holdout --with-new
step "HOLDOUT DONE"
