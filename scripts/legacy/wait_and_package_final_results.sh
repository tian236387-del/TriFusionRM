#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

TAG="${TAG:-20260726}"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/anaconda3/envs/rmbaseline/bin/python}"
PID_FILE="rmbase12_work/logs/final_results_evidence_${TAG}.pid"
PACKAGE_DIR="rmbase12_work/final_results_package_${TAG}"
PACKAGE_LOG="rmbase12_work/logs/final_results_package_${TAG}.log"

if [[ -s "${PID_FILE}" ]]; then
  suite_pid=$(cat "${PID_FILE}")
  while kill -0 "${suite_pid}" 2>/dev/null; do
    sleep 120
  done
fi

common_args=(
  --project-root .
  --factor-run-root "rmbase12_work/models/final_modality_factorial_ablation_seed42_${TAG}"
  --factor-metrics-dir "rmbase12_work/metrics/final_modality_factorial_ablation_seed42_${TAG}"
  --multiseed-run-root "rmbase12_work/models/final_main_multiseed_${TAG}"
  --multiseed-metrics-dir "rmbase12_work/metrics/final_main_multiseed_${TAG}"
  --full-model-dir rmbase12_work/models/multirm12_multi_length_fusion_cdhit80/multi_length_independent
  --baseline-table rmbase12_work/metrics/multirm12_multi_length_fusion_cdhit80/current_main_vs_retrained_fair_baselines_intersection.tsv
  --results-document docs/final_results_evidence_chain.md
  --log-file "rmbase12_work/logs/final_results_evidence_${TAG}.log"
  --package-dir "${PACKAGE_DIR}"
)

if ! "${PYTHON_BIN}" -B scripts/build_final_results_package.py "${common_args[@]}" --require-complete >> "${PACKAGE_LOG}" 2>&1; then
  "${PYTHON_BIN}" -B scripts/build_final_results_package.py "${common_args[@]}" >> "${PACKAGE_LOG}" 2>&1
  exit 1
fi

echo "Final results package: ${PACKAGE_DIR}" >> "${PACKAGE_LOG}"
