#!/usr/bin/env bash
set -euo pipefail

# Priority sequence-encoder experiments for the current StructDualRM-RBP setting.
#
# This keeps the strict CD-HIT split, RNAfold structure branch, ENCODE/RBP
# positional branch, and the best practical RBP gate hyperparameters fixed.
# Only the sequence encoder is changed.
#
# Example:
#   env -u LD_PRELOAD CUDA_ID=0 nohup bash scripts/run_sequence_encoder_priority_experiments.sh > sequence_encoder_priority.log 2>&1 &

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_ID="${CUDA_ID:-0}"
DATASET="${DATASET:-rmbase11_full_work/datasets/rmbase11_hg38_41nt_all_sites_cdhit80_strict_splits.tsv}"
SPLIT="${SPLIT:-split_cdhit80_cluster}"
RNAFOLD="${RNAFOLD:-rmbase11_full_work/structure/rnafold_41nt_all_sites.tsv}"
OUT_ROOT="${OUT_ROOT:-rmbase11_full_work/models/sequence_encoder_priority_cdhit80}"
METRICS_OUT="${METRICS_OUT:-rmbase11_full_work/metrics/sequence_encoder_priority_cdhit80}"

if [[ -z "${RBP_POSITIONAL:-}" ]]; then
  if [[ -s "rmbase11_full_work/rbp_encode_eclip_all/rmbase11_encode_eclip_top30_rbp_positional_features.tsv" ]]; then
    RBP_POSITIONAL="rmbase11_full_work/rbp_encode_eclip_all/rmbase11_encode_eclip_top30_rbp_positional_features.tsv"
  else
    RBP_POSITIONAL="rmbase11_full_work/rbp/rmbase11_rbp_positional_features.tsv"
  fi
fi

if [[ -z "${REFERENCE_METRICS:-}" ]]; then
  for candidate in \
    "rmbase11_full_work/models/encode_top30_rbp_hparam_sweep/rbp_s010_l0000_initm15_drop04_lr3e4/no_position_rbp_pos/metrics.json" \
    "rmbase11_full_work/models/rbp_hparam_sweep_cdhit80/rbp_s010_l0000_initm15_drop04_lr3e4/no_position_rbp_pos/metrics.json" \
    "rmbase11_full_work/models/rbp_positional_hparam_sweep_cdhit80/rbp_s010_l0000_initm15_drop04_lr3e4/no_position_rbp_pos/metrics.json" \
    "rmbase11_full_work/models/rbp_positional_ablation_cdhit80/no_position_rbp_pos/metrics.json"; do
    if [[ -s "${candidate}" ]]; then
      REFERENCE_METRICS="${candidate}"
      break
    fi
  done
fi
REFERENCE_METRICS="${REFERENCE_METRICS:-}"
REFERENCE_NAME="${REFERENCE_NAME:-current_main_rbp}"

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
SEED="${SEED:-42}"
TRAIN_MAX_ROWS="${TRAIN_MAX_ROWS:-0}"
RUN_REFERENCE="${RUN_REFERENCE:-0}"
RUN_SHUFFLED_CONTROL="${RUN_SHUFFLED_CONTROL:-0}"

TRANSFORMER_LAYERS="${TRANSFORMER_LAYERS:-2}"
TRANSFORMER_HEADS="${TRANSFORMER_HEADS:-4}"
TRANSFORMER_FF_MULT="${TRANSFORMER_FF_MULT:-4}"

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

COMMON_ARGS=(
  --dataset "${DATASET}"
  --rnafold "${RNAFOLD}"
  --rbp-positional-features "${RBP_POSITIONAL}"
  --split "${SPLIT}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --d-model "${D_MODEL}"
  --dropout "${DROPOUT}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --class-weight-power "${CLASS_WEIGHT_POWER}"
  --loss "${LOSS}"
  --focal-gamma "${FOCAL_GAMMA}"
  --gat-layers 2
  --selection-metric macro_AUPR_ovr
  --disable-position
  --use-rbp-positional
  --aux-seq-weight "${AUX_SEQ_WEIGHT}"
  --aux-struct-weight "${AUX_STRUCT_WEIGHT}"
  --gate-l1-weight "${GATE_L1}"
  --structure-scale "${STRUCTURE_SCALE}"
  --gate-init-logit "${GATE_INIT_LOGIT}"
  --rbp-pos-scale "${RBP_POS_SCALE}"
  --rbp-pos-gate-l1-weight "${RBP_POS_GATE_L1}"
  --rbp-pos-gate-init-logit "${RBP_POS_GATE_INIT_LOGIT}"
  --modality-dropout-struct "${MODALITY_DROPOUT_STRUCT}"
  --modality-dropout-rbp "${MODALITY_DROPOUT_RBP}"
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE}"
  --early-stopping-min-delta "${EARLY_STOPPING_MIN_DELTA}"
  --seed "${SEED}"
  --resume
  "${MAX_ROW_ARGS[@]}"
)

run_model() {
  local name="$1"
  local seq_encoder="$2"
  shift 2
  echo "===== Running sequence encoder priority: ${name} (${seq_encoder}) ====="
  CUDA_VISIBLE_DEVICES="${CUDA_ID}" "${PYTHON_BIN}" -B scripts/train_motif_struct_fusion_v4.py \
    "${COMMON_ARGS[@]}" \
    --out-dir "${OUT_ROOT}/${name}" \
    --seq-encoder "${seq_encoder}" \
    "$@"
}

VARIANTS=()

if [[ "${RUN_REFERENCE}" == "1" ]]; then
  run_model motif_reference_rbp motif
  VARIANTS+=("motif_reference_rbp")
fi

run_model kmer_extended_rbp kmer_extended
VARIANTS+=("kmer_extended_rbp")

run_model dilated_rbp dilated
VARIANTS+=("dilated_rbp")

run_model hybrid_transformer_rbp hybrid \
  --transformer-layers "${TRANSFORMER_LAYERS}" \
  --transformer-heads "${TRANSFORMER_HEADS}" \
  --transformer-ff-mult "${TRANSFORMER_FF_MULT}"
VARIANTS+=("hybrid_transformer_rbp")

if [[ "${RUN_SHUFFLED_CONTROL}" == "1" ]]; then
  run_model dilated_rbp_shuffled_rbp dilated --shuffle-rbp
  VARIANTS+=("dilated_rbp_shuffled_rbp")
fi

VARIANT_LIST="$(IFS=,; echo "${VARIANTS[*]}")"

SUMMARY_ARGS=(
  --root "${OUT_ROOT}"
  --out-dir "${METRICS_OUT}"
  --variants "${VARIANT_LIST}"
  --reference-name "${REFERENCE_NAME}"
)
if [[ -n "${REFERENCE_METRICS}" ]]; then
  SUMMARY_ARGS+=(--reference-metrics "${REFERENCE_METRICS}")
fi

"${PYTHON_BIN}" -B scripts/summarize_sequence_encoder_priority.py "${SUMMARY_ARGS[@]}"

echo "Done. Summary files:"
echo "  ${METRICS_OUT}/sequence_encoder_priority_macro.tsv"
echo "  ${METRICS_OUT}/sequence_encoder_priority_per_class.tsv"
echo "  ${METRICS_OUT}/sequence_encoder_priority_focus_classes.tsv"
echo "  ${METRICS_OUT}/sequence_encoder_priority_report.md"
if [[ -n "${REFERENCE_METRICS}" ]]; then
  echo "Reference metrics: ${REFERENCE_METRICS}"
else
  echo "Reference metrics not found; summary contains candidate models only."
fi
