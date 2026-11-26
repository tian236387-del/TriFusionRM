#!/usr/bin/env bash
set -euo pipefail

# Focused experiments for classes that lag behind baselines.
#
# Default setup:
#   - Keep global ENCODE eCLIP top30 RBP targets from the current final model.
#   - Add class-specific top-k RBP targets for Gm,m1A,m7G.
#   - Train with focal loss and stronger class balancing.
#
# Example:
#   env -u LD_PRELOAD CUDA_ID=0 bash scripts/run_class_specific_rbp_focused_experiments.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_ID="${CUDA_ID:-0}"
DATASET="${DATASET:-rmbase11_full_work/datasets/rmbase11_hg38_41nt_all_sites_cdhit80_strict_splits.tsv}"
SPLIT="${SPLIT:-split_cdhit80_cluster}"
RNAFOLD="${RNAFOLD:-rmbase11_full_work/structure/rnafold_41nt_all_sites.tsv}"
RBP_ROOT="${RBP_ROOT:-rmbase11_full_work/rbp_encode_eclip_all}"
GLOBAL_TOP_N="${GLOBAL_TOP_N:-30}"
FOCUS_CLASSES="${FOCUS_CLASSES:-Gm,m1A,m7G}"
TOP_K_LIST="${TOP_K_LIST:-10 20}"
MAX_FILES_PER_TARGET="${MAX_FILES_PER_TARGET:-2}"
OVERLAP_WINDOW="${OVERLAP_WINDOW:-200}"
ENCODE_INSECURE="${ENCODE_INSECURE:-0}"
ENCODE_RESOLVE="${ENCODE_RESOLVE:-}"
FORCE_PREPARE="${FORCE_PREPARE:-0}"

SWEEP_NAME="${SWEEP_NAME:-class_specific_rbp_focus}"
OUT_BASE="${OUT_BASE:-rmbase11_full_work/models/${SWEEP_NAME}}"
METRICS_BASE="${METRICS_BASE:-rmbase11_full_work/metrics/${SWEEP_NAME}}"

EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
SEED="${SEED:-42}"

mkdir -p "${RBP_ROOT}" "${OUT_BASE}" "${METRICS_BASE}"

MANIFEST="${RBP_ROOT}/encode_eclip_file_manifest.tsv"
GLOBAL_MANIFEST="${RBP_ROOT}/encode_eclip_top${GLOBAL_TOP_N}_manifest.tsv"
GLOBAL_COVERAGE="${RBP_ROOT}/encode_eclip_target_overlap_${OVERLAP_WINDOW}.tsv"

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

if [[ "${FORCE_PREPARE}" == "1" || ! -s "${GLOBAL_MANIFEST}" ]]; then
  echo "[prepare] Select global top-${GLOBAL_TOP_N} RBP targets"
  "${PYTHON_BIN}" -B scripts/select_rbp_manifest_by_site_overlap.py \
    --dataset "${DATASET}" \
    --manifest "${MANIFEST}" \
    --out-manifest "${GLOBAL_MANIFEST}" \
    --out-coverage "${GLOBAL_COVERAGE}" \
    --top-n "${GLOBAL_TOP_N}" \
    --window "${OVERLAP_WINDOW}"
else
  echo "[prepare] Reusing ${GLOBAL_MANIFEST}"
fi

CONFIG_TABLE="${METRICS_BASE}/class_specific_rbp_focus_configs.tsv"
echo -e "tag\tfocus_classes\ttop_k_per_class\tselected_manifest\trbp_positional\tout_root\tmetrics_out\tloss\tfocal_gamma\tclass_weight_power\trbp_pos_scale\trbp_pos_gate_l1\tdropout\tlr" > "${CONFIG_TABLE}"

for top_k in ${TOP_K_LIST}; do
  tag="global${GLOBAL_TOP_N}_focus_top${top_k}_focal_g15"
  selected_manifest="${RBP_ROOT}/encode_eclip_global${GLOBAL_TOP_N}_focus_${FOCUS_CLASSES//,/_}_top${top_k}_manifest.tsv"
  coverage_table="${RBP_ROOT}/encode_eclip_global${GLOBAL_TOP_N}_focus_${FOCUS_CLASSES//,/_}_top${top_k}_overlap.tsv"
  rbp_positional="${RBP_ROOT}/rmbase11_encode_eclip_global${GLOBAL_TOP_N}_focus_${FOCUS_CLASSES//,/_}_top${top_k}_rbp_positional_features.tsv"
  out_root="${OUT_BASE}/${tag}"
  metrics_out="${METRICS_BASE}/${tag}"

  if [[ "${FORCE_PREPARE}" == "1" || ! -s "${selected_manifest}" ]]; then
    echo "[prepare] Select class-specific RBP targets: ${tag}"
    "${PYTHON_BIN}" -B scripts/select_rbp_manifest_by_class_overlap.py \
      --dataset "${DATASET}" \
      --manifest "${MANIFEST}" \
      --base-manifest "${GLOBAL_MANIFEST}" \
      --out-manifest "${selected_manifest}" \
      --out-coverage "${coverage_table}" \
      --classes "${FOCUS_CLASSES}" \
      --top-k-per-class "${top_k}" \
      --window "${OVERLAP_WINDOW}" \
      --rank-by enrichment_log_count
  else
    echo "[prepare] Reusing ${selected_manifest}"
  fi

  if [[ "${FORCE_PREPARE}" == "1" || ! -s "${rbp_positional}" ]]; then
    echo "[prepare] Build positional RBP tensor: ${tag}"
    "${PYTHON_BIN}" -B scripts/build_rbp_positional_features.py \
      --dataset "${DATASET}" \
      --manifest "${selected_manifest}" \
      --out "${rbp_positional}"
  else
    echo "[prepare] Reusing ${rbp_positional}"
  fi

  echo -e "${tag}\t${FOCUS_CLASSES}\t${top_k}\t${selected_manifest}\t${rbp_positional}\t${out_root}\t${metrics_out}\tfocal\t1.5\t0.7\t0.10\t0.0\t0.4\t3e-4" >> "${CONFIG_TABLE}"

  echo "===== Train focused class-specific RBP experiment: ${tag} ====="
  CUDA_ID="${CUDA_ID}" \
  DATASET="${DATASET}" \
  SPLIT="${SPLIT}" \
  RNAFOLD="${RNAFOLD}" \
  RBP_POSITIONAL="${rbp_positional}" \
  OUT_ROOT="${out_root}" \
  METRICS_OUT="${metrics_out}" \
  EPOCHS="${EPOCHS}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  SEED="${SEED}" \
  LOSS="focal" \
  FOCAL_GAMMA="1.5" \
  CLASS_WEIGHT_POWER="0.7" \
  RBP_POS_SCALE="0.10" \
  RBP_POS_GATE_L1="0.0" \
  RBP_POS_GATE_INIT_LOGIT="-1.5" \
  DROPOUT="0.4" \
  LR="3e-4" \
  bash scripts/run_main_model_rbp_positional_ablation.sh
done

SUMMARY_TABLE="${METRICS_BASE}/class_specific_rbp_focus_summary.tsv"
"${PYTHON_BIN}" -B scripts/summarize_rbp_hparam_sweep.py \
  --metrics-base "${METRICS_BASE}" \
  --config-table "${CONFIG_TABLE}" \
  --out "${SUMMARY_TABLE}"

echo "Done. Focused RBP summary:"
echo "  ${SUMMARY_TABLE}"
