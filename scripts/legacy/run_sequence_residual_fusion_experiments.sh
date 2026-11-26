#!/usr/bin/env bash
set -euo pipefail

# Class-specific sequence residual fusion experiments.
#
# Main encoder remains the current motif CNN+BiGRU backbone.  A secondary
# sequence encoder is added only as a class-gated logit residual, restricted by
# default to the weak classes observed in the sequence encoder ablation.
#
# Example:
#   env -u LD_PRELOAD CUDA_ID=0 nohup bash scripts/run_sequence_residual_fusion_experiments.sh > sequence_residual_fusion.log 2>&1 &

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_ID="${CUDA_ID:-0}"
DATASET="${DATASET:-rmbase11_full_work/datasets/rmbase11_hg38_41nt_all_sites_cdhit80_strict_splits.tsv}"
SPLIT="${SPLIT:-split_cdhit80_cluster}"
RNAFOLD="${RNAFOLD:-rmbase11_full_work/structure/rnafold_41nt_all_sites.tsv}"
OUT_ROOT="${OUT_ROOT:-rmbase11_full_work/models/sequence_residual_fusion_cdhit80}"
METRICS_OUT="${METRICS_OUT:-rmbase11_full_work/metrics/sequence_residual_fusion_cdhit80}"

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

SEQ_RESIDUAL_ENCODER="${SEQ_RESIDUAL_ENCODER:-hybrid}"
SEQ_RESIDUAL_FOCUS_CLASSES="${SEQ_RESIDUAL_FOCUS_CLASSES:-Gm,m1A,m7G}"
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
  --out-dir "__OUT_DIR__"
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
  --seq-encoder motif
  --seq-residual-encoder "${SEQ_RESIDUAL_ENCODER}"
  --seq-residual-focus-classes "${SEQ_RESIDUAL_FOCUS_CLASSES}"
  --transformer-layers "${TRANSFORMER_LAYERS}"
  --transformer-heads "${TRANSFORMER_HEADS}"
  --transformer-ff-mult "${TRANSFORMER_FF_MULT}"
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

if [[ -n "${SWEEP_CONFIGS:-}" ]]; then
  mapfile -t CONFIGS <<< "${SWEEP_CONFIGS}"
else
  CONFIGS=(
    "seqres_hybrid_focus_s005_initm2_l0000|0.05|-2.0|0.0"
    "seqres_hybrid_focus_s010_initm2_l0000|0.10|-2.0|0.0"
    "seqres_hybrid_focus_s020_initm2_l0100|0.20|-2.0|0.0001"
  )
fi

echo -e "variant\tseq_residual_encoder\tfocus_classes\tseq_residual_scale\tseq_residual_gate_init_logit\tseq_residual_gate_l1\tout_dir" > "${METRICS_OUT}/sequence_residual_fusion_configs.tsv"

VARIANTS=()
for config in "${CONFIGS[@]}"; do
  [[ -z "${config// }" ]] && continue
  IFS="|" read -r name scale init_logit gate_l1 <<< "${config}"
  if [[ -z "${name:-}" || -z "${scale:-}" || -z "${init_logit:-}" || -z "${gate_l1:-}" ]]; then
    echo "Invalid config: ${config}" >&2
    exit 1
  fi

  out_dir="${OUT_ROOT}/${name}"
  mkdir -p "${out_dir}"
  echo -e "${name}\t${SEQ_RESIDUAL_ENCODER}\t${SEQ_RESIDUAL_FOCUS_CLASSES}\t${scale}\t${init_logit}\t${gate_l1}\t${out_dir}" >> "${METRICS_OUT}/sequence_residual_fusion_configs.tsv"
  echo "===== Running sequence residual fusion: ${name} ====="

  args=("${COMMON_ARGS[@]}")
  for i in "${!args[@]}"; do
    if [[ "${args[$i]}" == "__OUT_DIR__" ]]; then
      args[$i]="${out_dir}"
    fi
  done

  CUDA_VISIBLE_DEVICES="${CUDA_ID}" "${PYTHON_BIN}" -B scripts/train_motif_struct_fusion_v4.py \
    "${args[@]}" \
    --seq-residual-scale "${scale}" \
    --seq-residual-gate-init-logit "${init_logit}" \
    --seq-residual-gate-l1-weight "${gate_l1}"

  VARIANTS+=("${name}")
done

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
echo "  ${METRICS_OUT}/sequence_residual_fusion_configs.tsv"
if [[ -n "${REFERENCE_METRICS}" ]]; then
  echo "Reference metrics: ${REFERENCE_METRICS}"
else
  echo "Reference metrics not found; summary contains candidate models only."
fi
