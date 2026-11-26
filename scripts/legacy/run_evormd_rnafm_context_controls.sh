#!/usr/bin/env bash
set -euo pipefail

CUDA_ID="${CUDA_ID:-0}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-256}"

run_control() {
  local name="$1"
  local context_mode="$2"
  local shuffle_context="$3"
  local out_dir="rmbase12_work/official_reproduction_benchmark/evormd_rnafm_official_core_${name}_cdhit80_seed42"

  if [[ -s "${out_dir}/metrics.json" && -s "${out_dir}/test_per_class_metrics.tsv" ]]; then
    echo "Skip completed control: ${name}"
    return 0
  fi
  if [[ -e "${out_dir}/checkpoint.pt" || -e "${out_dir}/model.pt" ]]; then
    echo "Refusing to overwrite interrupted control: ${out_dir}" >&2
    exit 2
  fi

  echo "===== EvoRMD RNA-FM control: ${name} ====="
  CUDA_ID="${CUDA_ID}" \
  EPOCHS="${EPOCHS}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  CONTEXT_MODE="${context_mode}" \
  SHUFFLE_CONTEXT="${shuffle_context}" \
  OUT_DIR="${out_dir}" \
  bash scripts/run_evormd_rnafm_strict.sh
}

run_control safe_context safe 0
run_control sequence_only sequence_only 0
run_control full_metadata_shuffled full 1

echo "All EvoRMD RNA-FM context controls completed."
