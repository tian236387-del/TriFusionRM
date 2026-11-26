#!/usr/bin/env bash
set -euo pipefail

cd "${PROJECT_ROOT:-${HOME}/summer/m1A_m6A}"

PYTHON_BIN="${PYTHON_BIN:-${HOME}/anaconda3/envs/rmbaseline/bin/python}"
CUDA_ID="${CUDA_ID:-0}"
AUX_SEED="${AUX_SEED:-202}"
WORK="${WORK:-rmbase12_work/submission_strict}"
STATE="${STATE:-${WORK}/state}"
RUN_ROOT="${RUN_ROOT:-rmbase12_work/submission_strict_runs_20260806}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"
STATUS_LOG="${STATUS_LOG:-${RUN_ROOT}/job_status.tsv}"
DATASET="${DATASET:-${WORK}/multirm11_hg38_conflict_clean_strict_splits.tsv}"
RNAFOLD="${RNAFOLD:-rmbase12_work/structure/rnafold_41nt_multirm12_all_sites.tsv}"
CLASS_NAMES="Am,Cm,Gm,Um,m1A,m5C,m5U,m6A,m7G,Y,AtoI"

mkdir -p "${STATE}" "${LOG_DIR}"
while [[ ! -s "${STATE}/rbp.ready" ]]; do
  if [[ -s "${STATE}/prepare.failed" ]]; then
    cat "${STATE}/prepare.failed" >&2
    exit 2
  fi
  sleep 60
done
RBP_PRIMARY="$(tr -d '\r\n' < "${STATE}/rbp_primary_path.txt")"

COMMON=(
  scripts/train_motif_struct_fusion_v4.py
  --dataset "${DATASET}"
  --class-names "${CLASS_NAMES}"
  --rnafold "${RNAFOLD}"
  --split split_cluster201
  --input-length 41
  --sequence-column sequence_41nt
  --seq-encoder multi_length
  --multi-length-columns sequence_41nt,sequence_101nt,sequence_201nt
  --multi-length-lengths 41,101,201
  --multi-length-base-encoder motif
  --epochs 50
  --batch-size 256
  --grad-accum-steps 4
  --d-model 128
  --gat-layers 2
  --dropout 0.4
  --lr 3e-4
  --weight-decay 1e-4
  --class-weight-power 0.5
  --loss ce
  --aux-seq-weight 0.1
  --aux-struct-weight 0.05
  --length-aux-weight 0.05
  --gate-l1-weight 0.001
  --structure-scale 0.5
  --gate-init-logit -2.0
  --rbp-pos-scale 0.1
  --rbp-pos-gate-init-logit -1.5
  --rbp-pos-gate-l1-weight 0.0
  --disable-position
  --selection-metric macro_AUPR_ovr
  --early-stopping-patience 8
  --early-stopping-min-delta 1e-4
  --rbp-positional-features "${RBP_PRIMARY}"
  --seed "${AUX_SEED}"
)

run_variant() {
  local variant="$1"
  shift
  local job="main_${variant}_seed${AUX_SEED}"
  local out_dir="${RUN_ROOT}/main_controls/${variant}/seed${AUX_SEED}"
  mkdir -p "${out_dir}"
  if [[ -s "${out_dir}/metrics.json" && -s "${out_dir}/valid_predictions.tsv" && -s "${out_dir}/test_predictions.tsv" ]]; then
    return 0
  fi
  exec 9>"${out_dir}/auxiliary.lock"
  if ! flock -n 9; then
    return 0
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Iseconds)" "${job}" auxiliary_started 0 "${out_dir}" >> "${STATUS_LOG}"
  set +e
  CUDA_VISIBLE_DEVICES="${CUDA_ID}" PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -B "${COMMON[@]}" "$@" \
    --out-dir "${out_dir}" > "${LOG_DIR}/${job}.log" 2>&1
  code=$?
  set -e
  if [[ "${code}" -ne 0 ]]; then
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Iseconds)" "${job}" auxiliary_failed "${code}" "${out_dir}" >> "${STATUS_LOG}"
    printf '%s\n' "${job}" > "${STATE}/aux_main_seed${AUX_SEED}.failed"
    exit "${code}"
  fi
  if [[ ! -s "${out_dir}/metrics.json" || ! -s "${out_dir}/valid_predictions.tsv" || ! -s "${out_dir}/test_predictions.tsv" ]]; then
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Iseconds)" "${job}" auxiliary_incomplete 3 "${out_dir}" >> "${STATUS_LOG}"
    printf '%s\n' "${job}" > "${STATE}/aux_main_seed${AUX_SEED}.failed"
    exit 3
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Iseconds)" "${job}" auxiliary_completed 0 "${out_dir}" >> "${STATUS_LOG}"
}

run_variant sequence_only --disable-structure
run_variant sequence_structure --disable-pair-edges
run_variant sequence_rbp --disable-structure --use-rbp-positional
run_variant full --use-rbp-positional --disable-pair-edges
run_variant shuffled_structure --use-rbp-positional --disable-pair-edges --shuffle-structure
run_variant shuffled_rbp --use-rbp-positional --disable-pair-edges --shuffle-rbp
run_variant shuffled_rbp_bins --use-rbp-positional --disable-pair-edges --shuffle-rbp-bins
run_variant explicit_pair_edges --use-rbp-positional
run_variant capacity_matched_structure_cnn --use-rbp-positional --disable-pair-edges --structure-encoder cnn_control
run_variant parameter_matched_rbp_mlp --use-rbp-positional --disable-pair-edges --rbp-pos-encoder mlp_control

date -Iseconds > "${STATE}/aux_main_seed${AUX_SEED}.ready"
