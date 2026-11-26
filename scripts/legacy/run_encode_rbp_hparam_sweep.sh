#!/usr/bin/env bash
set -euo pipefail

# RBP positional hyperparameter sweep for the current StructDualRM main model.
#
# The script prepares ENCODE eCLIP top-N positional RBP features once, then runs
# several independent RBP ablation jobs. Each job writes to its own output
# directory so interrupted runs can be resumed without overwriting prior results.
#
# Example:
#   env -u LD_PRELOAD CUDA_ID=0 TOP_N=30 bash scripts/run_encode_rbp_hparam_sweep.sh
#
# Optional custom configs:
#   SWEEP_CONFIGS=$'tag|rbp_scale|rbp_l1|rbp_init|dropout|lr|weight_decay|rbp_dropout\n...' bash scripts/run_encode_rbp_hparam_sweep.sh

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
FORCE_PREPARE="${FORCE_PREPARE:-0}"

SWEEP_NAME="${SWEEP_NAME:-encode_top${TOP_N}_rbp_hparam_sweep}"
OUT_BASE="${OUT_BASE:-rmbase11_full_work/models/${SWEEP_NAME}}"
METRICS_BASE="${METRICS_BASE:-rmbase11_full_work/metrics/${SWEEP_NAME}}"
SUMMARY_OUT="${SUMMARY_OUT:-${METRICS_BASE}/rbp_hparam_sweep_summary.tsv}"

MANIFEST="${RBP_ROOT}/encode_eclip_file_manifest.tsv"
SELECTED_MANIFEST="${RBP_ROOT}/encode_eclip_top${TOP_N}_manifest.tsv"
COVERAGE_TABLE="${RBP_ROOT}/encode_eclip_target_overlap_${OVERLAP_WINDOW}.tsv"
RBP_POSITIONAL="${RBP_ROOT}/rmbase11_encode_eclip_top${TOP_N}_rbp_positional_features.tsv"

mkdir -p "${RBP_ROOT}" "${OUT_BASE}" "${METRICS_BASE}"

ENCODE_SSL_ARGS=()
if [[ "${ENCODE_INSECURE}" == "1" ]]; then
  ENCODE_SSL_ARGS=(--insecure)
fi

ENCODE_RESOLVE_ARGS=()
if [[ -n "${ENCODE_RESOLVE}" ]]; then
  ENCODE_RESOLVE_ARGS=(--resolve "${ENCODE_RESOLVE}")
fi

if [[ "${FORCE_PREPARE}" == "1" || ! -s "${MANIFEST}" ]]; then
  echo "[prepare] Download/query ENCODE eCLIP GRCh38 peak files"
  "${PYTHON_BIN}" -B scripts/download_encode_eclip_rbp_peaks.py \
    --out-dir "${RBP_ROOT}" \
    --all-targets \
    --max-files-per-target "${MAX_FILES_PER_TARGET}" \
    "${ENCODE_SSL_ARGS[@]}" \
    "${ENCODE_RESOLVE_ARGS[@]}"
else
  echo "[prepare] Reusing ${MANIFEST}"
fi

if [[ "${FORCE_PREPARE}" == "1" || ! -s "${SELECTED_MANIFEST}" ]]; then
  echo "[prepare] Select top-${TOP_N} RBP targets by overlap with RMBase sites"
  "${PYTHON_BIN}" -B scripts/select_rbp_manifest_by_site_overlap.py \
    --dataset "${DATASET}" \
    --manifest "${MANIFEST}" \
    --out-manifest "${SELECTED_MANIFEST}" \
    --out-coverage "${COVERAGE_TABLE}" \
    --top-n "${TOP_N}" \
    --window "${OVERLAP_WINDOW}"
else
  echo "[prepare] Reusing ${SELECTED_MANIFEST}"
fi

if [[ "${FORCE_PREPARE}" == "1" || ! -s "${RBP_POSITIONAL}" ]]; then
  echo "[prepare] Build top-${TOP_N} positional RBP feature tensor"
  "${PYTHON_BIN}" -B scripts/build_rbp_positional_features.py \
    --dataset "${DATASET}" \
    --manifest "${SELECTED_MANIFEST}" \
    --out "${RBP_POSITIONAL}"
else
  echo "[prepare] Reusing ${RBP_POSITIONAL}"
fi

if [[ -n "${SWEEP_CONFIGS:-}" ]]; then
  mapfile -t CONFIGS <<< "${SWEEP_CONFIGS}"
else
  CONFIGS=(
    "rbp_s005_l1000_initm2_drop03_lr3e4|0.05|0.001|-2.0|0.3|3e-4|1e-4|0.0"
    "rbp_s010_l1000_initm15_drop03_lr3e4|0.10|0.001|-1.5|0.3|3e-4|1e-4|0.0"
    "rbp_s010_l0100_initm15_drop03_lr3e4|0.10|0.0001|-1.5|0.3|3e-4|1e-4|0.0"
    "rbp_s020_l0100_initm1_drop03_lr3e4|0.20|0.0001|-1.0|0.3|3e-4|1e-4|0.0"
    "rbp_s010_l0000_initm15_drop04_lr3e4|0.10|0.0|-1.5|0.4|3e-4|1e-4|0.0"
    "rbp_s010_l0100_initm15_drop03_lr1e4|0.10|0.0001|-1.5|0.3|1e-4|1e-4|0.0"
    "rbp_s010_l0100_initm15_drop03_lr3e4_mdrbp01|0.10|0.0001|-1.5|0.3|3e-4|1e-4|0.1"
  )
fi

echo -e "tag\trbp_pos_scale\trbp_pos_gate_l1\trbp_pos_gate_init_logit\tdropout\tlr\tweight_decay\tmodality_dropout_rbp\tout_root\tmetrics_out" > "${METRICS_BASE}/rbp_hparam_sweep_configs.tsv"

for config in "${CONFIGS[@]}"; do
  [[ -z "${config// }" ]] && continue
  IFS="|" read -r tag rbp_scale rbp_l1 rbp_init dropout lr weight_decay rbp_dropout <<< "${config}"
  if [[ -z "${tag:-}" || -z "${rbp_scale:-}" || -z "${rbp_l1:-}" || -z "${rbp_init:-}" || -z "${dropout:-}" || -z "${lr:-}" || -z "${weight_decay:-}" || -z "${rbp_dropout:-}" ]]; then
    echo "Invalid config: ${config}" >&2
    exit 1
  fi

  config_out="${OUT_BASE}/${tag}"
  config_metrics="${METRICS_BASE}/${tag}"
  mkdir -p "${config_out}" "${config_metrics}"
  echo -e "${tag}\t${rbp_scale}\t${rbp_l1}\t${rbp_init}\t${dropout}\t${lr}\t${weight_decay}\t${rbp_dropout}\t${config_out}\t${config_metrics}" >> "${METRICS_BASE}/rbp_hparam_sweep_configs.tsv"

  echo "===== Sweep config: ${tag} ====="
  CUDA_ID="${CUDA_ID}" \
  DATASET="${DATASET}" \
  SPLIT="${SPLIT}" \
  RNAFOLD="${RNAFOLD}" \
  RBP_POSITIONAL="${RBP_POSITIONAL}" \
  OUT_ROOT="${config_out}" \
  METRICS_OUT="${config_metrics}" \
  DROPOUT="${dropout}" \
  LR="${lr}" \
  WEIGHT_DECAY="${weight_decay}" \
  RBP_POS_SCALE="${rbp_scale}" \
  RBP_POS_GATE_L1="${rbp_l1}" \
  RBP_POS_GATE_INIT_LOGIT="${rbp_init}" \
  MODALITY_DROPOUT_RBP="${rbp_dropout}" \
  bash scripts/run_main_model_rbp_positional_ablation.sh
done

"${PYTHON_BIN}" -B scripts/summarize_rbp_hparam_sweep.py \
  --metrics-base "${METRICS_BASE}" \
  --config-table "${METRICS_BASE}/rbp_hparam_sweep_configs.tsv" \
  --out "${SUMMARY_OUT}"

echo "Done. Sweep summary:"
echo "  ${SUMMARY_OUT}"
