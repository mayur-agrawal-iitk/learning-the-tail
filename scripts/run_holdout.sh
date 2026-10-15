#!/usr/bin/env bash
# One-time holdout evaluation of the frozen models (docs/PROTOCOL.md). Steps run sequentially
# because of the 7.5 GB RAM limit. The script is resumable: each finished step leaves a marker in
# logs/holdout_steps/, and a re-run skips steps that are already done (the neural steps are also
# checkpointed job by job).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=python
VENV=.venv/Scripts/python.exe          # CUDA build of torch for the GPU-capable steps
[ -x "$VENV" ] || VENV=python
MARK=logs/holdout_steps
mkdir -p "$MARK"
say() { echo "[$(date +%H:%M:%S)] $*"; }
step() {
    local id=$1 label=$2; shift 2
    if [ -f "$MARK/$id.done" ]; then say "$id $label (already done, skipped)"; return; fi
    say "$id $label"
    "$@"
    touch "$MARK/$id.done"
}

step 1 "benchmarks (fast models)" \
    $PY scripts/run_econometric.py --config holdout --models HS GARCH-N GARCH-t GJR-skt FHS GARCH-EVT
step 2 "benchmarks (semiparametric models)" \
    $PY scripts/run_econometric.py --config holdout --models CAViaR-ES FZ-1F FZ-Hyb
step 3 "GJR-X with cross-market news" $PY scripts/run_garchx.py --config holdout
rm -f results/holdout/.gpu_slots/*.lock
step 4 "paper-1 deep models" $VENV scripts/run_neural.py --config holdout
step 5 "NG-FZ v1" $PY scripts/run_neural_garch.py --config holdout
step 6a "CTG-FZ v2" $VENV scripts/run_contagion.py --config holdout --model CTG-FZ
step 6b "LSTMQ-FZ ablation" $VENV scripts/run_contagion.py --config holdout --model LSTMQ-FZ
step 7a "evaluation (paper-1 model set)" $PY scripts/evaluate.py --config holdout
step 7b "evaluation (all models)" $PY scripts/evaluate.py --config holdout --with-new
say "HOLDOUT DONE"
