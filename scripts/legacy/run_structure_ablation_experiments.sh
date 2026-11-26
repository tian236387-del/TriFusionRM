#!/usr/bin/env bash
set -euo pipefail

# Structure-source ablation for MotifStructFusionV4.
#
# Main comparison:
#   sequence_only
#   rnafold_mfe
#   rnafold_mfe_shuffled
#   rnafold_prob
#   rnafold_prob_shuffled
# Optional, slower:
#   rnaplfold_accessibility
#   rnaplfold_accessibility_shuffled
#
# Example:
#   env -u LD_PRELOAD bash scripts/run_structure_ablation_experiments.sh
#   RUN_RNAPLFOLD=1 env -u LD_PRELOAD bash scripts/run_structure_ablation_experiments.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-rmbase11_full_work/datasets/rmbase11_hg38_41nt_all_sites_cdhit80_strict_splits.tsv}"
SPLIT="${SPLIT:-split_cdhit80_cluster}"
RNAFOLD_MFE="${RNAFOLD_MFE:-rmbase11_full_work/structure/rnafold_41nt_all_sites.tsv}"
RNAFOLD_PROB="${RNAFOLD_PROB:-rmbase11_full_work/structure/rnafold_prob_41nt_all_sites.tsv}"
RNAPLFOLD_ACCESS="${RNAPLFOLD_ACCESS:-rmbase11_full_work/structure/rnaplfold_accessibility_41nt_all_sites.tsv}"
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
RUN_RNAPLFOLD="${RUN_RNAPLFOLD:-0}"

mkdir -p "${OUT_ROOT}" "${METRICS_OUT}"

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

echo "[1/4] Build probability-style ViennaRNA structure features if possible"
if "${PYTHON_BIN}" - <<'PY'
import RNA  # noqa: F401
PY
then
  "${PYTHON_BIN}" -B scripts/build_viennarna_probability_structure.py \
    --dataset "${DATASET}" \
    --out "${RNAFOLD_PROB}" \
    --resume \
    "${FEATURE_MAX_ROW_ARGS[@]}"
else
  echo "skip rnafold_prob: Python ViennaRNA bindings are unavailable in this environment"
fi

if [[ "${RUN_RNAPLFOLD}" == "1" ]]; then
  echo "[2/4] Build RNAplfold local-accessibility features"
  "${PYTHON_BIN}" -B scripts/build_rnaplfold_accessibility_structure.py \
    --dataset "${DATASET}" \
    --out "${RNAPLFOLD_ACCESS}" \
    --mfe-reference "${RNAFOLD_MFE}" \
    --resume \
    "${FEATURE_MAX_ROW_ARGS[@]}"
else
  echo "[2/4] Skip RNAplfold feature generation by default; set RUN_RNAPLFOLD=1 to enable it"
fi

run_model() {
  local name="$1"
  local structure_file="$2"
  shift 2
  echo "===== Running structure ablation: ${name} ====="
  "${PYTHON_BIN}" -B scripts/train_motif_struct_fusion_v4.py \
    --dataset "${DATASET}" \
    --rnafold "${structure_file}" \
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

echo "[3/4] Train structure ablation variants"
run_model sequence_only "${RNAFOLD_MFE}" --disable-structure --aux-struct-weight 0 --structure-scale 0 --gate-l1-weight 0
run_model rnafold_mfe "${RNAFOLD_MFE}"
run_model rnafold_mfe_shuffled "${RNAFOLD_MFE}" --shuffle-structure

VARIANTS="sequence_only,rnafold_mfe,rnafold_mfe_shuffled"
if [[ -f "${RNAFOLD_PROB}" ]]; then
  run_model rnafold_prob "${RNAFOLD_PROB}"
  run_model rnafold_prob_shuffled "${RNAFOLD_PROB}" --shuffle-structure
  VARIANTS="${VARIANTS},rnafold_prob,rnafold_prob_shuffled"
else
  echo "skip rnafold_prob training: missing ${RNAFOLD_PROB}"
fi

if [[ -f "${RNAPLFOLD_ACCESS}" ]]; then
  run_model rnaplfold_accessibility "${RNAPLFOLD_ACCESS}"
  run_model rnaplfold_accessibility_shuffled "${RNAPLFOLD_ACCESS}" --shuffle-structure
  VARIANTS="${VARIANTS},rnaplfold_accessibility,rnaplfold_accessibility_shuffled"
else
  echo "skip rnaplfold training: missing ${RNAPLFOLD_ACCESS}"
fi

echo "[4/4] Summarize structure ablation"
"${PYTHON_BIN}" -B scripts/summarize_structure_ablation.py \
  --root "${OUT_ROOT}" \
  --out-dir "${METRICS_OUT}" \
  --variants "${VARIANTS}"

echo "Done."
echo "Macro table: ${METRICS_OUT}/structure_ablation_macro_metrics.tsv"
echo "Per-class matrix: ${METRICS_OUT}/structure_ablation_per_class_matrix.tsv"
echo "Decision table: ${METRICS_OUT}/structure_ablation_decision_table.tsv"
