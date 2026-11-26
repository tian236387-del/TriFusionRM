#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-${HOME}/summer/m1A_m6A}"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/anaconda3/envs/rmbaseline/bin/python}"
CUDA_ID="${CUDA_ID:-0}"
OUT="rmbase12_work/detection_strict_20260814"
DATA="${OUT}/hg38_modified_vs_unannotated_strict_splits.tsv"
RNAFOLD="${OUT}/features/rnafold_41nt_modified_vs_unannotated.tsv"
RBP_MEMMAP="${OUT}/features/modified_vs_unannotated_top30_rbp_positional.memmap"
SPLIT="split_detection_cluster201"
SEED=42
GROUP_MAX_WEIGHT=100

cd "${ROOT}"
mkdir -p "${OUT}/logs" "${OUT}/state" "${OUT}/models"

while [[ ! -s "${OUT}/state/seed42_modalities.complete" ]]; do
  printf '%s waiting_for_seed42_modalities\n' "$(date -Iseconds)"
  sleep 60
done
GROUP_MAX_WEIGHT="$(tr -d '[:space:]' < "${OUT}/state/selected_group_max_weight.txt")"
if [[ ! "${GROUP_MAX_WEIGHT}" =~ ^(100|300|1000)$ ]]; then
  printf 'Invalid selected group max weight: %s\n' "${GROUP_MAX_WEIGHT}" >&2
  exit 1
fi

run_control() {
  local name="$1"
  shift
  local model_out="${OUT}/models/${name}_seed${SEED}"
  local log="${OUT}/logs/${name}_seed${SEED}.log"
  if [[ -s "${model_out}/audited/detection_metrics.json" ]]; then
    return
  fi
  mkdir -p "${model_out}"
  CUDA_VISIBLE_DEVICES="${CUDA_ID}" PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -B \
    scripts/train_motif_struct_fusion_v4.py \
    --dataset "${DATA}" \
    --class-names unmodified,modified \
    --rbp-features "" \
    --split "${SPLIT}" \
    --out-dir "${model_out}" \
    --input-length 41 \
    --sequence-column sequence_41nt \
    --epochs 30 \
    --batch-size 256 \
    --grad-accum-steps 4 \
    --d-model 128 \
    --gat-layers 2 \
    --dropout 0.4 \
    --seq-encoder multi_length \
    --multi-length-columns sequence_41nt,sequence_101nt,sequence_201nt \
    --multi-length-lengths 41,101,201 \
    --multi-length-base-encoder motif \
    --afamily-classes "" \
    --lr 3e-4 \
    --weight-decay 1e-4 \
    --class-weight-power 0.0 \
    --loss ce \
    --aux-seq-weight 0.1 \
    --length-aux-weight 0.05 \
    --sampling-group-column original_mod_type \
    --sampling-group-power 1.0 \
    --sampling-group-max-weight "${GROUP_MAX_WEIGHT}" \
    --evaluation-group-column original_mod_type \
    --group-positive-class modified \
    --group-min-positive 20 \
    --selection-metric macro_group_AUPR \
    --selection-focus-weight 0.0 \
    --early-stopping-patience 5 \
    --early-stopping-min-delta 1e-4 \
    --disable-position \
    --seed "${SEED}" \
    "$@" 2>&1 | tee "${log}"
  "${PYTHON_BIN}" -B scripts/evaluate_hg38_detection_predictions.py \
    --dataset "${DATA}" \
    --valid-predictions "${model_out}/valid_predictions.tsv" \
    --test-predictions "${model_out}/test_predictions.tsv" \
    --output-dir "${model_out}/audited" \
    --split-column "${SPLIT}" \
    --minimum-positive-rows 20 \
    2>&1 | tee "${OUT}/logs/${name}_audit_seed${SEED}.log"
}

run_control sequence_structure_cnn_control \
  --rnafold "${RNAFOLD}" \
  --rbp-positional-features "" \
  --structure-encoder cnn_control \
  --aux-struct-weight 0.05 \
  --structure-scale 0.5 \
  --gate-l1-weight 0.001

run_control sequence_structure_shuffled \
  --rnafold "${RNAFOLD}" \
  --rbp-positional-features "" \
  --shuffle-structure \
  --aux-struct-weight 0.05 \
  --structure-scale 0.5 \
  --gate-l1-weight 0.001

run_control sequence_rbp_mlp_control \
  --rnafold "" \
  --disable-structure \
  --rbp-positional-features "${RBP_MEMMAP}" \
  --use-rbp-positional \
  --rbp-pos-encoder mlp_control \
  --aux-struct-weight 0.0 \
  --gate-l1-weight 0.0 \
  --rbp-pos-scale 0.1 \
  --rbp-pos-gate-init-logit -1.5 \
  --rbp-pos-gate-l1-weight 0.0

run_control sequence_rbp_shuffled_bins \
  --rnafold "" \
  --disable-structure \
  --rbp-positional-features "${RBP_MEMMAP}" \
  --use-rbp-positional \
  --shuffle-rbp-bins \
  --aux-struct-weight 0.0 \
  --gate-l1-weight 0.0 \
  --rbp-pos-scale 0.1 \
  --rbp-pos-gate-init-logit -1.5 \
  --rbp-pos-gate-l1-weight 0.0

date -Iseconds > "${OUT}/state/seed42_controls.complete"
printf '%s detection_seed42_controls_complete\n' "$(date -Iseconds)"
