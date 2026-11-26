#!/usr/bin/env bash
set -euo pipefail

# V4: strong motif/k-mer sequence backbone + sequence-free class-specific RNAfold structure gates.
# The model is resume-safe: rerunning this script continues from each variant checkpoint.

DATASET="${DATASET:-rmbase11_full_work/datasets/rmbase11_hg38_41nt_all_sites_mmseqs80_strict_splits.tsv}"
RNAFOLD="${RNAFOLD:-rmbase11_full_work/structure/rnafold_41nt_all_sites.tsv}"
RBP="${RBP:-rmbase11_full_work/rbp/rmbase11_rbp_features.tsv}"
RBP_POSITIONAL="${RBP_POSITIONAL:-rmbase11_full_work/rbp/rmbase11_rbp_positional_features.tsv}"
OUT_ROOT="${OUT_ROOT:-rmbase11_full_work/models/motif_struct_fusion_v4_class_modality_gate_v3}"
SPLIT="${SPLIT:-split_mmseqs80_cluster}"
BASELINE_ROOT="${BASELINE_ROOT:-rmbase11_full_work/official_reproduction_benchmark/strict_mmseqs80_split}"
METRICS_OUT="${METRICS_OUT:-rmbase11_full_work/metrics/motif_struct_fusion_v4_class_modality_gate_v3_strict}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
D_MODEL="${D_MODEL:-128}"
GATE_L1="${GATE_L1:-0.001}"
STRUCTURE_SCALE="${STRUCTURE_SCALE:-0.5}"
GATE_INIT_LOGIT="${GATE_INIT_LOGIT:--2.0}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-8}"
EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-0.0001}"
RBP_POS_SCALE="${RBP_POS_SCALE:-0.05}"
RBP_POS_GATE_L1="${RBP_POS_GATE_L1:-0.001}"
RBP_POS_GATE_INIT_LOGIT="${RBP_POS_GATE_INIT_LOGIT:--2.0}"
MODALITY_DROPOUT_STRUCT="${MODALITY_DROPOUT_STRUCT:-0.0}"
MODALITY_DROPOUT_RBP="${MODALITY_DROPOUT_RBP:-0.0}"

mkdir -p "${OUT_ROOT}"

run_model() {
  local name="$1"
  shift
  echo "===== Running V4 ${name} ====="
  python -B scripts/train_motif_struct_fusion_v4.py \
    --dataset "${DATASET}" \
    --rnafold "${RNAFOLD}" \
    --rbp-features "${RBP}" \
    --rbp-positional-features "${RBP_POSITIONAL}" \
    --split "${SPLIT}" \
    --out-dir "${OUT_ROOT}/${name}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --d-model "${D_MODEL}" \
    --gat-layers 2 \
    --selection-metric macro_AUPR_ovr \
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
    --resume \
    "$@"
}

run_model full
run_model sequence_only --disable-structure --aux-struct-weight 0 --structure-scale 0 --gate-l1-weight 0
run_model shuffled_structure --shuffle-structure
run_model no_position --disable-position
run_model with_rbp --use-rbp
if [[ -f "${RBP_POSITIONAL}" ]]; then
  run_model with_rbp_positional --use-rbp-positional
else
  echo "skip with_rbp_positional: missing ${RBP_POSITIONAL}"
fi

python -B scripts/summarize_paired_gat_rbp_fusion_experiments.py \
  --proposed-root "${OUT_ROOT}" \
  --baseline-root "${BASELINE_ROOT}" \
  --out-dir "${METRICS_OUT}"

python -B scripts/compare_motif_struct_fusion_v4_per_class.py \
  --root "${OUT_ROOT}" \
  --out-dir "${METRICS_OUT}"

echo "Done. Summary: ${METRICS_OUT}"
