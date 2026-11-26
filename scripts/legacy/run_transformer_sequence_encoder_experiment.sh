#!/usr/bin/env bash
set -euo pipefail

# Compare the original motif CNN+BiGRU sequence encoder with a hybrid
# motif-CNN + Transformer encoder under the same CD-HIT80 strict split.
#
# Example:
#   env -u LD_PRELOAD bash scripts/run_transformer_sequence_encoder_experiment.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-rmbase11_full_work/datasets/rmbase11_hg38_41nt_all_sites_cdhit80_strict_splits.tsv}"
SPLIT="${SPLIT:-split_cdhit80_cluster}"
RNAFOLD="${RNAFOLD:-rmbase11_full_work/structure/rnafold_41nt_all_sites.tsv}"
REFERENCE_ROOT="${REFERENCE_ROOT:-rmbase11_full_work/models/structure_ablation_cdhit80}"
OUT_ROOT="${OUT_ROOT:-rmbase11_full_work/models/sequence_encoder_ablation_cdhit80}"
METRICS_OUT="${METRICS_OUT:-rmbase11_full_work/metrics/sequence_encoder_ablation_cdhit80}"

EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
D_MODEL="${D_MODEL:-128}"
GATE_L1="${GATE_L1:-0.001}"
STRUCTURE_SCALE="${STRUCTURE_SCALE:-0.5}"
GATE_INIT_LOGIT="${GATE_INIT_LOGIT:--2.0}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-8}"
EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-0.0001}"
SEED="${SEED:-42}"
TRAIN_MAX_ROWS="${TRAIN_MAX_ROWS:-0}"
DISABLE_POSITION="${DISABLE_POSITION:-1}"
SEQ_ENCODER="${SEQ_ENCODER:-hybrid}"
TRANSFORMER_LAYERS="${TRANSFORMER_LAYERS:-2}"
TRANSFORMER_HEADS="${TRANSFORMER_HEADS:-4}"
TRANSFORMER_FF_MULT="${TRANSFORMER_FF_MULT:-4}"

mkdir -p "${OUT_ROOT}" "${METRICS_OUT}"

if [[ ! -f "${DATASET}" ]]; then
  echo "ERROR: dataset not found: ${DATASET}" >&2
  exit 1
fi

if [[ ! -f "${RNAFOLD}" ]]; then
  echo "ERROR: RNAfold structure file not found: ${RNAFOLD}" >&2
  exit 1
fi

POSITION_ARGS=()
if [[ "${DISABLE_POSITION}" == "1" ]]; then
  POSITION_ARGS=(--disable-position)
fi

MAX_ROW_ARGS=()
if [[ "${TRAIN_MAX_ROWS}" != "0" ]]; then
  MAX_ROW_ARGS=(--max-rows "${TRAIN_MAX_ROWS}")
fi

run_model() {
  local name="$1"
  shift
  echo "===== Running sequence encoder ablation: ${name} ====="
  "${PYTHON_BIN}" -B scripts/train_motif_struct_fusion_v4.py \
    --dataset "${DATASET}" \
    --rnafold "${RNAFOLD}" \
    --split "${SPLIT}" \
    --out-dir "${OUT_ROOT}/${name}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --d-model "${D_MODEL}" \
    --gat-layers 2 \
    --selection-metric macro_AUPR_ovr \
    --seq-encoder "${SEQ_ENCODER}" \
    --transformer-layers "${TRANSFORMER_LAYERS}" \
    --transformer-heads "${TRANSFORMER_HEADS}" \
    --transformer-ff-mult "${TRANSFORMER_FF_MULT}" \
    --gate-l1-weight "${GATE_L1}" \
    --structure-scale "${STRUCTURE_SCALE}" \
    --gate-init-logit "${GATE_INIT_LOGIT}" \
    --early-stopping-patience "${EARLY_STOPPING_PATIENCE}" \
    --early-stopping-min-delta "${EARLY_STOPPING_MIN_DELTA}" \
    --seed "${SEED}" \
    --resume \
    "${POSITION_ARGS[@]}" \
    "${MAX_ROW_ARGS[@]}" \
    "$@"
}

run_model transformer_hybrid_sequence_only --disable-structure --aux-struct-weight 0 --structure-scale 0 --gate-l1-weight 0
run_model transformer_hybrid_rnafold_mfe
run_model transformer_hybrid_rnafold_mfe_shuffled --shuffle-structure

"${PYTHON_BIN}" -B scripts/summarize_sequence_encoder_ablation.py \
  --reference-root "${REFERENCE_ROOT}" \
  --root "${OUT_ROOT}" \
  --out-dir "${METRICS_OUT}"

echo "Done."
echo "Macro table: ${METRICS_OUT}/sequence_encoder_ablation_macro_metrics.tsv"
echo "Per-class matrix: ${METRICS_OUT}/sequence_encoder_ablation_per_class_matrix.tsv"
