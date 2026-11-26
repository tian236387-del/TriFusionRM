#!/usr/bin/env bash
set -euo pipefail

# Run the RNAfold-loop structure ablation variants and merge them into the
# existing structure-source ablation summary.
#
# Example:
#   env -u LD_PRELOAD bash scripts/run_rnafold_loop_ablation_experiment.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-rmbase11_full_work/datasets/rmbase11_hg38_41nt_all_sites_cdhit80_strict_splits.tsv}"
SPLIT="${SPLIT:-split_cdhit80_cluster}"
RNAFOLD_MFE="${RNAFOLD_MFE:-rmbase11_full_work/structure/rnafold_41nt_all_sites.tsv}"
RNAFOLD_LOOP="${RNAFOLD_LOOP:-rmbase11_full_work/structure/rnafold_loop_41nt_all_sites.tsv}"
OUT_ROOT="${OUT_ROOT:-rmbase11_full_work/models/structure_ablation_cdhit80}"
METRICS_OUT="${METRICS_OUT:-rmbase11_full_work/metrics/structure_ablation_cdhit80}"

EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
D_MODEL="${D_MODEL:-128}"
GATE_L1="${GATE_L1:-0.001}"
STRUCTURE_SCALE="${STRUCTURE_SCALE:-0.5}"
GATE_INIT_LOGIT="${GATE_INIT_LOGIT:--2.0}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-8}"
EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-0.0001}"
SEED="${SEED:-42}"
FEATURE_MAX_ROWS="${FEATURE_MAX_ROWS:-0}"
TRAIN_MAX_ROWS="${TRAIN_MAX_ROWS:-0}"
DISABLE_POSITION="${DISABLE_POSITION:-1}"

LOOP_EXTRA_STRUCT_FIELDS="${LOOP_EXTRA_STRUCT_FIELDS:-center_loop,central_11nt_loop_fraction,central_21nt_loop_fraction,loop_fraction_window,unpaired_run_around_center,nearest_paired_distance,enclosing_pair_span,loop_segment_count,max_loop_span,mean_loop_span}"
LOOP_NODE_FEATURE_FIELDS="${LOOP_NODE_FEATURE_FIELDS:-paired_flags,loop_flags}"

mkdir -p "${OUT_ROOT}" "${METRICS_OUT}"

if [[ ! -f "${DATASET}" ]]; then
  echo "ERROR: dataset not found: ${DATASET}" >&2
  exit 1
fi

if [[ ! -f "${RNAFOLD_MFE}" ]]; then
  echo "ERROR: RNAfold MFE structure file not found: ${RNAFOLD_MFE}" >&2
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

FEATURE_MAX_ROW_ARGS=()
if [[ "${FEATURE_MAX_ROWS}" != "0" ]]; then
  FEATURE_MAX_ROW_ARGS=(--max-rows "${FEATURE_MAX_ROWS}")
fi

echo "[1/3] Build loop-enhanced RNAfold features"
"${PYTHON_BIN}" -B scripts/build_rnafold_loop_structure.py \
  --rnafold "${RNAFOLD_MFE}" \
  --out "${RNAFOLD_LOOP}" \
  "${FEATURE_MAX_ROW_ARGS[@]}"

run_model() {
  local name="$1"
  shift
  echo "===== Running RNAfold-loop ablation: ${name} ====="
  "${PYTHON_BIN}" -B scripts/train_motif_struct_fusion_v4.py \
    --dataset "${DATASET}" \
    --rnafold "${RNAFOLD_LOOP}" \
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
    --early-stopping-patience "${EARLY_STOPPING_PATIENCE}" \
    --early-stopping-min-delta "${EARLY_STOPPING_MIN_DELTA}" \
    --seed "${SEED}" \
    --resume \
    --extra-struct-fields "${LOOP_EXTRA_STRUCT_FIELDS}" \
    --node-feature-fields "${LOOP_NODE_FEATURE_FIELDS}" \
    "${POSITION_ARGS[@]}" \
    "${MAX_ROW_ARGS[@]}" \
    "$@"
}

echo "[2/3] Train RNAfold-loop variants"
run_model rnafold_loop
run_model rnafold_loop_shuffled --shuffle-structure

echo "[3/3] Summarize all available structure-ablation variants"
"${PYTHON_BIN}" -B scripts/summarize_structure_ablation.py \
  --root "${OUT_ROOT}" \
  --out-dir "${METRICS_OUT}" \
  --variants sequence_only,rnafold_mfe,rnafold_mfe_shuffled,rnafold_prob,rnafold_prob_shuffled,rnafold_loop,rnafold_loop_shuffled,rnaplfold_accessibility,rnaplfold_accessibility_shuffled

echo "Done."
echo "Macro table: ${METRICS_OUT}/structure_ablation_macro_metrics.tsv"
echo "Per-class matrix: ${METRICS_OUT}/structure_ablation_per_class_matrix.tsv"
echo "Decision table: ${METRICS_OUT}/structure_ablation_decision_table.tsv"
