#!/usr/bin/env bash
set -euo pipefail

# Run only the RNAstructure Fold structure-source ablation variants and merge
# them into the existing structure-ablation summary.
#
# Example:
#   env -u LD_PRELOAD bash scripts/run_rnastructure_fold_ablation_experiment.sh
#   RNASTRUCTURE_WORKERS=8 env -u LD_PRELOAD bash scripts/run_rnastructure_fold_ablation_experiment.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-rmbase11_full_work/datasets/rmbase11_hg38_41nt_all_sites_cdhit80_strict_splits.tsv}"
SPLIT="${SPLIT:-split_cdhit80_cluster}"
RNASTRUCTURE_FOLD="${RNASTRUCTURE_FOLD:-rmbase11_full_work/structure/rnastructure_fold_41nt_all_sites.tsv}"
RNASTRUCTURE_FOLD_BIN="${RNASTRUCTURE_FOLD_BIN:-Fold}"
RNASTRUCTURE_FOLD_EXTRA_ARGS="${RNASTRUCTURE_FOLD_EXTRA_ARGS:-}"
RNASTRUCTURE_WORKERS="${RNASTRUCTURE_WORKERS:-1}"
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

ALL_VARIANTS="${ALL_VARIANTS:-sequence_only,rnafold_mfe,rnafold_mfe_shuffled,rnafold_prob,rnafold_prob_shuffled,rnafold_loop,rnafold_loop_shuffled,rnaplfold_accessibility,rnaplfold_accessibility_shuffled,rnastructure_fold,rnastructure_fold_shuffled}"

mkdir -p "${OUT_ROOT}" "${METRICS_OUT}"

if [[ ! -f "${DATASET}" ]]; then
  echo "ERROR: dataset not found: ${DATASET}" >&2
  exit 1
fi

if ! command -v "${RNASTRUCTURE_FOLD_BIN}" >/dev/null 2>&1 && [[ ! -x "${RNASTRUCTURE_FOLD_BIN}" ]]; then
  echo "ERROR: RNAstructure Fold binary not found: ${RNASTRUCTURE_FOLD_BIN}" >&2
  echo "Install RNAstructure, add Fold to PATH, or set RNASTRUCTURE_FOLD_BIN=/path/to/Fold." >&2
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

echo "[1/3] Build RNAstructure Fold MFE features"
"${PYTHON_BIN}" -B scripts/build_rnastructure_fold_structure.py \
  --dataset "${DATASET}" \
  --out "${RNASTRUCTURE_FOLD}" \
  --fold-bin "${RNASTRUCTURE_FOLD_BIN}" \
  --fold-extra-args "${RNASTRUCTURE_FOLD_EXTRA_ARGS}" \
  --workers "${RNASTRUCTURE_WORKERS}" \
  --resume \
  "${FEATURE_MAX_ROW_ARGS[@]}"

run_model() {
  local name="$1"
  shift
  echo "===== Running RNAstructure Fold ablation: ${name} ====="
  "${PYTHON_BIN}" -B scripts/train_motif_struct_fusion_v4.py \
    --dataset "${DATASET}" \
    --rnafold "${RNASTRUCTURE_FOLD}" \
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
    "${POSITION_ARGS[@]}" \
    "${MAX_ROW_ARGS[@]}" \
    "$@"
}

echo "[2/3] Train RNAstructure Fold variants"
run_model rnastructure_fold
run_model rnastructure_fold_shuffled --shuffle-structure

echo "[3/3] Summarize all available structure-ablation variants"
"${PYTHON_BIN}" -B scripts/summarize_structure_ablation.py \
  --root "${OUT_ROOT}" \
  --out-dir "${METRICS_OUT}" \
  --variants "${ALL_VARIANTS}"

echo "Done."
echo "Macro table: ${METRICS_OUT}/structure_ablation_macro_metrics.tsv"
echo "Per-class matrix: ${METRICS_OUT}/structure_ablation_per_class_matrix.tsv"
echo "Decision table: ${METRICS_OUT}/structure_ablation_decision_table.tsv"
