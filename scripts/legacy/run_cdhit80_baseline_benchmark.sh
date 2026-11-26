#!/usr/bin/env bash
set -euo pipefail

# Train the three architecture-adapted baselines on the unified RMBase11
# CD-HIT80 strict split, then summarize them against the current StructDualRM
# main model.

DATASET="${DATASET:-rmbase11_full_work/datasets/rmbase11_hg38_41nt_all_sites_cdhit80_strict_splits.tsv}"
SPLIT="${SPLIT:-split_cdhit80_cluster}"
OUT_ROOT="${OUT_ROOT:-rmbase11_full_work/official_reproduction_benchmark/cdhit80_strict_split}"
PROPOSED_DIR="${PROPOSED_DIR:-rmbase11_full_work/models/motif_struct_fusion_v4_cdhit80/no_position}"
SUMMARY_DIR="${SUMMARY_DIR:-rmbase11_full_work/metrics/cdhit80_baseline_vs_structdualrm}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
CUDA_ID="${CUDA_ID:-0}"
DROPOUT="${DROPOUT:-0.2}"
LR="${LR:-3e-4}"
LR_DECAY="${LR_DECAY:-0.98}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
CLASS_WEIGHT_POWER="${CLASS_WEIGHT_POWER:-0.5}"
SEED="${SEED:-42}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${OUT_ROOT}" "${SUMMARY_DIR}"

run_baseline() {
  local model="$1"
  echo "===== CD-HIT80 baseline: ${model} ====="
  CUDA_VISIBLE_DEVICES="${CUDA_ID}" "${PYTHON_BIN}" -B scripts/train_rmbase11_official_reproduction_benchmark.py \
    --model "${model}" \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --out-dir "${OUT_ROOT}/${model}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --dropout "${DROPOUT}" \
    --lr "${LR}" \
    --lr-decay "${LR_DECAY}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --class-weight-power "${CLASS_WEIGHT_POWER}" \
    --selection-metric macro_AUPR_ovr \
    --seed "${SEED}" \
    --resume
}

run_baseline multirm_official
run_baseline transrnam_official
run_baseline evormd_adapted

"${PYTHON_BIN}" -B scripts/summarize_cdhit80_baseline_vs_structdualrm.py \
  --baseline-root "${OUT_ROOT}" \
  --proposed-dir "${PROPOSED_DIR}" \
  --out-dir "${SUMMARY_DIR}"

echo "Done. Summary directory: ${SUMMARY_DIR}"
