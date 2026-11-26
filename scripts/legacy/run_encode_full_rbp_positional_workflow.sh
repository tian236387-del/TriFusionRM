#!/usr/bin/env bash
set -euo pipefail

# End-to-end RBP positional workflow:
#   1. Download comprehensive ENCODE eCLIP GRCh38 narrowPeak peak files.
#   2. Rank RBP targets by overlap with candidate RNA modification sites.
#   3. Build a dense top-N RBP positional tensor.
#   4. Run the main-model RBP positional ablation.
#
# Example:
#   env -u LD_PRELOAD CUDA_ID=0 TOP_N=30 bash scripts/run_encode_full_rbp_positional_workflow.sh
#
# Notes:
#   TOP_N controls the dense tensor size.  TOP_N=30 is a practical default.
#   TOP_N=50 is more comprehensive but can create a substantially larger TSV
#   and use more CPU RAM during training.

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_ID="${CUDA_ID:-0}"
DATASET="${DATASET:-rmbase11_full_work/datasets/rmbase11_hg38_41nt_all_sites_cdhit80_strict_splits.tsv}"
SPLIT="${SPLIT:-split_cdhit80_cluster}"
RNAFOLD="${RNAFOLD:-rmbase11_full_work/structure/rnafold_41nt_all_sites.tsv}"
RBP_ROOT="${RBP_ROOT:-rmbase11_full_work/rbp_encode_eclip_all}"
TOP_N="${TOP_N:-30}"
MAX_FILES_PER_TARGET="${MAX_FILES_PER_TARGET:-2}"
OVERLAP_WINDOW="${OVERLAP_WINDOW:-200}"
ENCODE_INSECURE="${ENCODE_INSECURE:-0}"
ENCODE_RESOLVE="${ENCODE_RESOLVE:-}"

MANIFEST="${RBP_ROOT}/encode_eclip_file_manifest.tsv"
SELECTED_MANIFEST="${RBP_ROOT}/encode_eclip_top${TOP_N}_manifest.tsv"
COVERAGE_TABLE="${RBP_ROOT}/encode_eclip_target_overlap_${OVERLAP_WINDOW}.tsv"
RBP_POSITIONAL="${RBP_ROOT}/rmbase11_encode_eclip_top${TOP_N}_rbp_positional_features.tsv"
OUT_ROOT="${OUT_ROOT:-rmbase11_full_work/models/rbp_positional_ablation_cdhit80_encode_top${TOP_N}}"
METRICS_OUT="${METRICS_OUT:-rmbase11_full_work/metrics/rbp_positional_ablation_cdhit80_encode_top${TOP_N}}"

mkdir -p "${RBP_ROOT}"

ENCODE_SSL_ARGS=()
if [[ "${ENCODE_INSECURE}" == "1" ]]; then
  ENCODE_SSL_ARGS=(--insecure)
fi

ENCODE_RESOLVE_ARGS=()
if [[ -n "${ENCODE_RESOLVE}" ]]; then
  ENCODE_RESOLVE_ARGS=(--resolve "${ENCODE_RESOLVE}")
fi

echo "[1/4] Download/query ENCODE eCLIP GRCh38 peak files"
"${PYTHON_BIN}" -B scripts/download_encode_eclip_rbp_peaks.py \
  --out-dir "${RBP_ROOT}" \
  --all-targets \
  --max-files-per-target "${MAX_FILES_PER_TARGET}" \
  "${ENCODE_SSL_ARGS[@]}" \
  "${ENCODE_RESOLVE_ARGS[@]}"

echo "[2/4] Select top-${TOP_N} RBP targets by overlap with RMBase sites"
"${PYTHON_BIN}" -B scripts/select_rbp_manifest_by_site_overlap.py \
  --dataset "${DATASET}" \
  --manifest "${MANIFEST}" \
  --out-manifest "${SELECTED_MANIFEST}" \
  --out-coverage "${COVERAGE_TABLE}" \
  --top-n "${TOP_N}" \
  --window "${OVERLAP_WINDOW}"

echo "[3/4] Build top-${TOP_N} positional RBP feature tensor"
"${PYTHON_BIN}" -B scripts/build_rbp_positional_features.py \
  --dataset "${DATASET}" \
  --manifest "${SELECTED_MANIFEST}" \
  --out "${RBP_POSITIONAL}"

echo "[4/4] Train main-model RBP positional ablation using top-${TOP_N} ENCODE RBP features"
CUDA_ID="${CUDA_ID}" \
DATASET="${DATASET}" \
SPLIT="${SPLIT}" \
RNAFOLD="${RNAFOLD}" \
RBP_POSITIONAL="${RBP_POSITIONAL}" \
OUT_ROOT="${OUT_ROOT}" \
METRICS_OUT="${METRICS_OUT}" \
bash scripts/run_main_model_rbp_positional_ablation.sh

echo "Done. Key outputs:"
echo "  ${COVERAGE_TABLE}"
echo "  ${SELECTED_MANIFEST}"
echo "  ${RBP_POSITIONAL}"
echo "  ${METRICS_OUT}/rbp_positional_ablation_macro.tsv"
echo "  ${METRICS_OUT}/rbp_positional_ablation_per_class.tsv"
