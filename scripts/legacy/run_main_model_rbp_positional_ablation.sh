#!/usr/bin/env bash
set -euo pipefail

# RBP positional ablation for the current main StructDualRM setting.
#
# Variants:
#   no_position
#       Main sequence + RNAfold MFE structure model without positional encodings.
#   no_position_rbp_pos
#       Main model plus positional RBP/site features.
#   no_position_shuffled_rbp_pos
#       Same as above, but RBP positional features are mismatched by shuffling site ids.
#
# Example:
#   env -u LD_PRELOAD CUDA_ID=0 bash scripts/run_main_model_rbp_positional_ablation.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_ID="${CUDA_ID:-0}"
DATASET="${DATASET:-rmbase11_full_work/datasets/rmbase11_hg38_41nt_all_sites_cdhit80_strict_splits.tsv}"
SPLIT="${SPLIT:-split_cdhit80_cluster}"
RNAFOLD="${RNAFOLD:-rmbase11_full_work/structure/rnafold_41nt_all_sites.tsv}"
RBP_POSITIONAL="${RBP_POSITIONAL:-rmbase11_full_work/rbp/rmbase11_rbp_positional_features.tsv}"
OUT_ROOT="${OUT_ROOT:-rmbase11_full_work/models/rbp_positional_ablation_cdhit80}"
METRICS_OUT="${METRICS_OUT:-rmbase11_full_work/metrics/rbp_positional_ablation_cdhit80}"

EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
D_MODEL="${D_MODEL:-128}"
DROPOUT="${DROPOUT:-0.3}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
CLASS_WEIGHT_POWER="${CLASS_WEIGHT_POWER:-0.5}"
LOSS="${LOSS:-ce}"
FOCAL_GAMMA="${FOCAL_GAMMA:-2.0}"
AUX_SEQ_WEIGHT="${AUX_SEQ_WEIGHT:-0.1}"
AUX_STRUCT_WEIGHT="${AUX_STRUCT_WEIGHT:-0.05}"
GATE_L1="${GATE_L1:-0.001}"
STRUCTURE_SCALE="${STRUCTURE_SCALE:-0.5}"
GATE_INIT_LOGIT="${GATE_INIT_LOGIT:--2.0}"
RBP_POS_SCALE="${RBP_POS_SCALE:-0.05}"
RBP_POS_GATE_L1="${RBP_POS_GATE_L1:-0.001}"
RBP_POS_GATE_INIT_LOGIT="${RBP_POS_GATE_INIT_LOGIT:--2.0}"
MODALITY_DROPOUT_STRUCT="${MODALITY_DROPOUT_STRUCT:-0.0}"
MODALITY_DROPOUT_RBP="${MODALITY_DROPOUT_RBP:-0.0}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-8}"
EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-0.0001}"
SEED="${SEED:-42}"
TRAIN_MAX_ROWS="${TRAIN_MAX_ROWS:-0}"

for required in "${DATASET}" "${RNAFOLD}" "${RBP_POSITIONAL}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing required file: ${required}" >&2
    exit 1
  fi
done

mkdir -p "${OUT_ROOT}" "${METRICS_OUT}"

MAX_ROW_ARGS=()
if [[ "${TRAIN_MAX_ROWS}" != "0" ]]; then
  MAX_ROW_ARGS=(--max-rows "${TRAIN_MAX_ROWS}")
fi

run_model() {
  local name="$1"
  shift
  echo "===== Running RBP positional ablation: ${name} ====="
  CUDA_VISIBLE_DEVICES="${CUDA_ID}" "${PYTHON_BIN}" -B scripts/train_motif_struct_fusion_v4.py \
    --dataset "${DATASET}" \
    --rnafold "${RNAFOLD}" \
    --rbp-positional-features "${RBP_POSITIONAL}" \
    --split "${SPLIT}" \
    --out-dir "${OUT_ROOT}/${name}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --d-model "${D_MODEL}" \
    --dropout "${DROPOUT}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --class-weight-power "${CLASS_WEIGHT_POWER}" \
    --loss "${LOSS}" \
    --focal-gamma "${FOCAL_GAMMA}" \
    --gat-layers 2 \
    --selection-metric macro_AUPR_ovr \
    --disable-position \
    --aux-seq-weight "${AUX_SEQ_WEIGHT}" \
    --aux-struct-weight "${AUX_STRUCT_WEIGHT}" \
    --gate-l1-weight "${GATE_L1}" \
    --structure-scale "${STRUCTURE_SCALE}" \
    --gate-init-logit "${GATE_INIT_LOGIT}" \
    --rbp-pos-scale "${RBP_POS_SCALE}" \
    --rbp-pos-gate-l1-weight "${RBP_POS_GATE_L1}" \
    --rbp-pos-gate-init-logit "${RBP_POS_GATE_INIT_LOGIT}" \
    --modality-dropout-struct "${MODALITY_DROPOUT_STRUCT}" \
    --modality-dropout-rbp "${MODALITY_DROPOUT_RBP}" \
    --early-stopping-patience "${EARLY_STOPPING_PATIENCE}" \
    --early-stopping-min-delta "${EARLY_STOPPING_MIN_DELTA}" \
    --seed "${SEED}" \
    --resume \
    "${MAX_ROW_ARGS[@]}" \
    "$@"
}

run_model no_position
run_model no_position_rbp_pos --use-rbp-positional
run_model no_position_shuffled_rbp_pos --use-rbp-positional --shuffle-rbp

"${PYTHON_BIN}" -B scripts/summarize_rbp_positional_ablation.py \
  --root "${OUT_ROOT}" \
  --out-dir "${METRICS_OUT}" \
  --baseline-variant no_position \
  --rbp-variant no_position_rbp_pos \
  --shuffled-rbp-variant no_position_shuffled_rbp_pos

echo "Done. Summary files:"
echo "  ${METRICS_OUT}/rbp_positional_ablation_macro.tsv"
echo "  ${METRICS_OUT}/rbp_positional_ablation_per_class.tsv"
echo "  ${METRICS_OUT}/rbp_positional_ablation_report.md"
