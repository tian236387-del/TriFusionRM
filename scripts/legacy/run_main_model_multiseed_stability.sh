#!/usr/bin/env bash
set -euo pipefail

# Multi-seed stability runs for the current main StructDualRM-RBP model.
#
# This script fixes the final main-model setting and changes only the random
# seed.  It is intended for reporting mean +/- SD under the strict CD-HIT split.
#
# Example:
#   env -u LD_PRELOAD CUDA_ID=0 nohup bash scripts/run_main_model_multiseed_stability.sh > main_model_multiseed.log 2>&1 &

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_ID="${CUDA_ID:-0}"
DATASET="${DATASET:-rmbase11_full_work/datasets/rmbase11_hg38_41nt_all_sites_cdhit80_strict_splits.tsv}"
SPLIT="${SPLIT:-split_cdhit80_cluster}"
RNAFOLD="${RNAFOLD:-rmbase11_full_work/structure/rnafold_41nt_all_sites.tsv}"
OUT_ROOT="${OUT_ROOT:-rmbase11_full_work/models/main_model_multiseed_cdhit80}"
METRICS_OUT="${METRICS_OUT:-rmbase11_full_work/metrics/main_model_multiseed_cdhit80}"
SEEDS="${SEEDS:-42 13 17 23 37}"

if [[ -z "${RBP_POSITIONAL:-}" ]]; then
  if [[ -s "rmbase11_full_work/rbp_encode_eclip_all/rmbase11_encode_eclip_top30_rbp_positional_features.tsv" ]]; then
    RBP_POSITIONAL="rmbase11_full_work/rbp_encode_eclip_all/rmbase11_encode_eclip_top30_rbp_positional_features.tsv"
  else
    RBP_POSITIONAL="rmbase11_full_work/rbp/rmbase11_rbp_positional_features.tsv"
  fi
fi

EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
D_MODEL="${D_MODEL:-128}"
DROPOUT="${DROPOUT:-0.4}"
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
RBP_POS_SCALE="${RBP_POS_SCALE:-0.10}"
RBP_POS_GATE_L1="${RBP_POS_GATE_L1:-0.0}"
RBP_POS_GATE_INIT_LOGIT="${RBP_POS_GATE_INIT_LOGIT:--1.5}"
MODALITY_DROPOUT_STRUCT="${MODALITY_DROPOUT_STRUCT:-0.0}"
MODALITY_DROPOUT_RBP="${MODALITY_DROPOUT_RBP:-0.0}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-8}"
EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-0.0001}"
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

echo -e "seed\tout_dir\tdataset\tsplit\trbp_positional\tdropout\tlr\trbp_pos_scale\trbp_pos_gate_l1\trbp_pos_gate_init_logit" > "${METRICS_OUT}/main_model_multiseed_configs.tsv"

for seed in ${SEEDS}; do
  out_dir="${OUT_ROOT}/seed_${seed}"
  mkdir -p "${out_dir}"
  echo -e "${seed}\t${out_dir}\t${DATASET}\t${SPLIT}\t${RBP_POSITIONAL}\t${DROPOUT}\t${LR}\t${RBP_POS_SCALE}\t${RBP_POS_GATE_L1}\t${RBP_POS_GATE_INIT_LOGIT}" >> "${METRICS_OUT}/main_model_multiseed_configs.tsv"
  echo "===== Running main model stability seed=${seed} ====="
  CUDA_VISIBLE_DEVICES="${CUDA_ID}" "${PYTHON_BIN}" -B scripts/train_motif_struct_fusion_v4.py \
    --dataset "${DATASET}" \
    --rnafold "${RNAFOLD}" \
    --rbp-positional-features "${RBP_POSITIONAL}" \
    --split "${SPLIT}" \
    --out-dir "${out_dir}" \
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
    --use-rbp-positional \
    --seq-encoder motif \
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
    --seed "${seed}" \
    --resume \
    "${MAX_ROW_ARGS[@]}"
done

"${PYTHON_BIN}" -B scripts/summarize_main_model_multiseed.py \
  --root "${OUT_ROOT}" \
  --out-dir "${METRICS_OUT}" \
  --seeds "${SEEDS}"

echo "Done. Summary files:"
echo "  ${METRICS_OUT}/main_model_multiseed_by_seed.tsv"
echo "  ${METRICS_OUT}/main_model_multiseed_macro_mean_std.tsv"
echo "  ${METRICS_OUT}/main_model_multiseed_per_class_mean_std.tsv"
echo "  ${METRICS_OUT}/main_model_multiseed_focus_classes.tsv"
echo "  ${METRICS_OUT}/main_model_multiseed_report.md"
