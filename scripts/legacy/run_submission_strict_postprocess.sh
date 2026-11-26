#!/usr/bin/env bash
set -euo pipefail

cd "${PROJECT_ROOT:-${HOME}/summer/m1A_m6A}"

PYTHON_BIN="${PYTHON_BIN:-${HOME}/anaconda3/envs/rmbaseline/bin/python}"
BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-${HOME}/anaconda3/envs/YmodPred/bin/python}"
WORK="${WORK:-rmbase12_work/submission_strict}"
STATE="${STATE:-${WORK}/state}"
RUN_ROOT="${RUN_ROOT:-rmbase12_work/submission_strict_runs_20260806}"
RESULT_ROOT="${RESULT_ROOT:-rmbase12_work/submission_strict_final_20260806}"
LOG="${RESULT_ROOT}/postprocess.log"
NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${RESULT_ROOT}/numba_cache}"

mkdir -p "${STATE}" "${RESULT_ROOT}" "${NUMBA_CACHE_DIR}"
rm -f "${STATE}/postprocess.failed"
trap 'status=$?; printf "%s postprocess_failed exit=%s line=%s\n" "$(date -Iseconds)" "$status" "$LINENO" | tee "${STATE}/postprocess.failed"; exit "$status"' ERR

while [[ ! -s "${STATE}/gpu_queue.ready" ]]; do
  if [[ -s "${STATE}/gpu_queue.failed" ]]; then
    printf '%s gpu_queue_failed=%s\n' "$(date -Iseconds)" "$(cat "${STATE}/gpu_queue.failed")" >> "${LOG}"
    exit 2
  fi
  sleep 300
done

"${PYTHON_BIN}" -B scripts/aggregate_submission_strict_results.py \
  --run-root "${RUN_ROOT}" \
  --dataset "${WORK}/multirm11_hg38_conflict_clean_strict_splits.tsv" \
  --output "${RESULT_ROOT}" \
  >> "${LOG}" 2>&1

PRED="${RESULT_ROOT}/ensemble_predictions"
NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR}" "${BOOTSTRAP_PYTHON}" -B scripts/grouped_paired_bootstrap_submission.py \
  --dataset "${WORK}/multirm11_hg38_conflict_clean_strict_splits.tsv" \
  --model "StructDualRM=${PRED}/structdualrm_test_predictions_raw.tsv" \
  --model "MultiRM=${PRED}/multirm_test_predictions_raw.tsv" \
  --model "TransRNAm=${PRED}/transrnam_test_predictions_raw.tsv" \
  --model "EvoRMD=${PRED}/evormd_test_predictions_raw.tsv" \
  --model "RNA-FM_sequence-only=${PRED}/rna_fm_sequence_only_test_predictions_raw.tsv" \
  --iterations 2000 \
  --seed 20260806 \
  --output "${RESULT_ROOT}/bootstrap" \
  >> "${LOG}" 2>&1

"${PYTHON_BIN}" -B scripts/audit_submission_strict_study.py \
  --work "${WORK}" \
  --run-root "${RUN_ROOT}" \
  --result-root "${RESULT_ROOT}" \
  --output "${RESULT_ROOT}/audit" \
  --strict \
  >> "${LOG}" 2>&1

date -Iseconds > "${STATE}/postprocess.ready"
printf '%s postprocess_complete\n' "$(date -Iseconds)" >> "${LOG}"
