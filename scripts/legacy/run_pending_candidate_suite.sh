#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${HOME}/summer/m1A_m6A}"
NO_PAIR_PID_FILE="${NO_PAIR_PID_FILE:-rmbase12_work/logs/no_pair_edges_multiseed_20260730.pid}"
AUDIT_DIR="${AUDIT_DIR:-rmbase12_work/metrics/experiment_audit_20260730}"
POLL_SECONDS="${POLL_SECONDS:-60}"

cd "${PROJECT_ROOT}"

parent="$(cat "${NO_PAIR_PID_FILE}" 2>/dev/null || true)"
while [[ -n "${parent}" ]] && kill -0 "${parent}" 2>/dev/null; do
  sleep "${POLL_SECONDS}"
done

echo "===== starting sequence+RBP multiseed at $(date -Is) ====="
CUDA_ID=0 bash scripts/run_sequence_rbp_multiseed.sh

echo "===== auditing experiments at $(date -Is) ====="
"${HOME}/anaconda3/envs/rmbaseline/bin/python" -B scripts/audit_structdualrm_experiments.py \
  --models-root rmbase12_work/models \
  --out-dir "${AUDIT_DIR}"

echo "===== candidate suite complete at $(date -Is) ====="
