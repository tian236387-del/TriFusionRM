#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-rmbase12_work/datasets/rmbase_multirm12_hg38_41nt_all_sites_cdhit80_strict_splits.tsv}"
SPLIT="${SPLIT:-split_cdhit80_cluster}"
OUT_ROOT="${OUT_ROOT:-rmbase12_work/official_reproduction_benchmark/evormd_context_retraining_strict_seed42}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
CUDA_ID="${CUDA_ID:-0}"
SEED="${SEED:-42}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${OUT_ROOT}"

for context_mode in full safe sequence_only; do
  out_dir="${OUT_ROOT}/${context_mode}"
  if [[ -e "${out_dir}/checkpoint.pt" || -e "${out_dir}/model.pt" ]]; then
    echo "Refusing to reuse an existing run: ${out_dir}" >&2
    exit 2
  fi
  mkdir -p "${out_dir}"
  echo "===== EvoRMD-adapted context_mode=${context_mode} ====="
  CUDA_VISIBLE_DEVICES="${CUDA_ID}" "${PYTHON_BIN}" -B scripts/train_rmbase11_official_reproduction_benchmark.py \
    --model evormd_adapted \
    --context-mode "${context_mode}" \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --out-dir "${out_dir}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --dropout 0.2 \
    --lr 3e-4 \
    --lr-decay 0.98 \
    --weight-decay 1e-4 \
    --class-weight-power 0.5 \
    --selection-metric macro_AUPR_ovr \
    --seed "${SEED}"
done

"${PYTHON_BIN}" -B scripts/summarize_evormd_context_retraining.py \
  --run-root "${OUT_ROOT}" \
  --out-dir "${OUT_ROOT}/summary"

echo "Done: ${OUT_ROOT}/summary"
