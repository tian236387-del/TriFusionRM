#!/usr/bin/env bash
set -euo pipefail

cd "${PROJECT_ROOT:-${HOME}/summer/m1A_m6A}"

PYTHON_BIN="${PYTHON_BIN:-${HOME}/anaconda3/envs/rmbaseline/bin/python}"
RNAFM_PYTHON="${RNAFM_PYTHON:-.venvs/evormd_rnafm/bin/python}"
CUDA_ID="${CUDA_ID:-0}"
WORK="${WORK:-rmbase12_work/submission_strict}"
STATE="${STATE:-${WORK}/state}"
RUN_ROOT="${RUN_ROOT:-rmbase12_work/submission_strict_runs_20260806}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"
STATUS_LOG="${STATUS_LOG:-${RUN_ROOT}/job_status.tsv}"
DATASET="${DATASET:-${WORK}/multirm11_hg38_conflict_clean_strict_splits.tsv}"
MULTILABEL_DATASET="${MULTILABEL_DATASET:-${WORK}/multirm12_hg38_coordinate_multilabel_strict_splits.tsv}"
RNAFOLD="${RNAFOLD:-rmbase12_work/structure/rnafold_41nt_multirm12_all_sites.tsv}"
CLASS_NAMES="Am,Cm,Gm,Um,m1A,m5C,m5U,m6A,m7G,Y,AtoI"
SEEDS=(42 17 73 101 202)

mkdir -p "${STATE}" "${RUN_ROOT}" "${LOG_DIR}"
if [[ ! -s "${STATUS_LOG}" ]]; then
  printf 'timestamp\tjob\tstatus\texit_code\toutput_dir\n' > "${STATUS_LOG}"
fi

wait_for_state() {
  local ready="$1"
  local failed="$2"
  while [[ ! -s "${ready}" ]]; do
    if [[ -s "${failed}" ]]; then
      echo "Preparation failed: ${failed}" >&2
      cat "${failed}" >&2
      exit 2
    fi
    sleep 60
  done
}

record_status() {
  local job="$1" status="$2" code="$3" out_dir="$4"
  printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Iseconds)" "${job}" "${status}" "${code}" "${out_dir}" >> "${STATUS_LOG}"
}

run_job() {
  local job="$1" out_dir="$2" python_bin="$3"
  shift 3
  mkdir -p "${out_dir}"
  if [[ -s "${out_dir}/metrics.json" && -s "${out_dir}/valid_predictions.tsv" && -s "${out_dir}/test_predictions.tsv" ]]; then
    record_status "${job}" completed_existing 0 "${out_dir}"
    return 0
  fi
  local resume=()
  if [[ -s "${out_dir}/checkpoint.pt" ]]; then
    resume=(--resume)
  fi
  record_status "${job}" started 0 "${out_dir}"
  set +e
  CUDA_VISIBLE_DEVICES="${CUDA_ID}" PYTHONUNBUFFERED=1 "${python_bin}" -B "$@" \
    --out-dir "${out_dir}" "${resume[@]}" > "${LOG_DIR}/${job}.log" 2>&1
  local code=$?
  set -e
  if [[ "${code}" -ne 0 ]]; then
    record_status "${job}" failed "${code}" "${out_dir}"
    printf '%s\n' "${job}" > "${STATE}/gpu_queue.failed"
    return "${code}"
  fi
  if [[ ! -s "${out_dir}/metrics.json" || ! -s "${out_dir}/valid_predictions.tsv" || ! -s "${out_dir}/test_predictions.tsv" ]]; then
    record_status "${job}" incomplete_output 3 "${out_dir}"
    printf '%s\n' "${job}" > "${STATE}/gpu_queue.failed"
    return 3
  fi
  record_status "${job}" completed 0 "${out_dir}"
}

wait_for_state "${STATE}/splits.ready" "${STATE}/prepare.failed"
rm -f "${STATE}/gpu_queue.failed"

BASELINE_COMMON=(
  scripts/train_rmbase11_official_reproduction_benchmark.py
  --dataset "${DATASET}"
  --class-names "${CLASS_NAMES}"
  --split split_cluster201
  --input-length 41
  --epochs 50
  --batch-size 2048
  --dropout 0.2
  --lr 3e-4
  --lr-decay 0.98
  --weight-decay 1e-4
  --class-weight-power 0.5
  --selection-metric macro_AUPR_ovr
  --early-stopping-patience 8
  --early-stopping-min-delta 1e-4
)

for seed in "${SEEDS[@]}"; do
  for model in multirm_official transrnam_official; do
    job="baseline_${model}_seed${seed}"
    run_job "${job}" "${RUN_ROOT}/baselines/${model}/seed${seed}" "${PYTHON_BIN}" \
      "${BASELINE_COMMON[@]}" --model "${model}" --seed "${seed}"
  done
done

RNAFM_COMMON=(
  scripts/train_rmbase11_official_reproduction_benchmark.py
  --model evormd_rnafm_official_core
  --dataset "${DATASET}"
  --class-names "${CLASS_NAMES}"
  --split split_cluster201
  --input-length 41
  --meta-dim 160
  --mlp-depth 1
  --epochs 30
  --batch-size 256
  --lr 1e-3
  --lr-decay 1.0
  --weight-decay 0
  --class-weight-power 0
  --downsample-train-classes m6A,m5C
  --downsample-alpha 0.6
  --selection-metric macro_AUPR_ovr
  --early-stopping-patience 6
  --early-stopping-min-delta 1e-4
)

for seed in "${SEEDS[@]}"; do
  run_job "baseline_evormd_safe_seed${seed}" "${RUN_ROOT}/baselines/evormd_safe/seed${seed}" "${RNAFM_PYTHON}" \
    "${RNAFM_COMMON[@]}" --context-mode safe --seed "${seed}"
  run_job "baseline_rnafm_sequence_only_seed${seed}" "${RUN_ROOT}/baselines/rnafm_sequence_only/seed${seed}" "${RNAFM_PYTHON}" \
    "${RNAFM_COMMON[@]}" --context-mode sequence_only --sequence-column sequence_201nt \
    --input-length 201 --batch-size 64 --seed "${seed}"
done

wait_for_state "${STATE}/rbp.ready" "${STATE}/prepare.failed"
RBP_PRIMARY="$(tr -d '\r\n' < "${STATE}/rbp_primary_path.txt")"

MAIN_COMMON=(
  scripts/train_motif_struct_fusion_v4.py
  --dataset "${DATASET}"
  --class-names "${CLASS_NAMES}"
  --rnafold "${RNAFOLD}"
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
)

run_main_variant() {
  local variant="$1" seed="$2"
  shift 2
  run_job "main_${variant}_seed${seed}" "${RUN_ROOT}/main_controls/${variant}/seed${seed}" "${PYTHON_BIN}" \
    "${MAIN_COMMON[@]}" --split split_cluster201 \
    --rbp-positional-features "${RBP_PRIMARY}" --seed "${seed}" "$@"
}

for seed in "${SEEDS[@]}"; do
  run_main_variant sequence_only "${seed}" --disable-structure
  run_main_variant sequence_structure "${seed}" --disable-pair-edges
  run_main_variant sequence_rbp "${seed}" --disable-structure --use-rbp-positional
  run_main_variant full "${seed}" --use-rbp-positional --disable-pair-edges
  run_main_variant shuffled_structure "${seed}" --use-rbp-positional --disable-pair-edges --shuffle-structure
  run_main_variant shuffled_rbp "${seed}" --use-rbp-positional --disable-pair-edges --shuffle-rbp
  run_main_variant shuffled_rbp_bins "${seed}" --use-rbp-positional --disable-pair-edges --shuffle-rbp-bins
  run_main_variant explicit_pair_edges "${seed}" --use-rbp-positional
  run_main_variant capacity_matched_structure_cnn "${seed}" --use-rbp-positional --disable-pair-edges --structure-encoder cnn_control
  run_main_variant parameter_matched_rbp_mlp "${seed}" --use-rbp-positional --disable-pair-edges --rbp-pos-encoder mlp_control
done

for split_label in gene_holdout transcript_holdout study_holdout; do
  split_column="split_${split_label}"
  rbp_cache="${WORK}/rbp_train_only/multirm11_${split_label}_top30_train_rbp_positional.memmap"
  for seed in "${SEEDS[@]}"; do
    run_job "holdout_${split_label}_seed${seed}" "${RUN_ROOT}/holdouts/${split_label}/seed${seed}" "${PYTHON_BIN}" \
      "${MAIN_COMMON[@]}" --split "${split_column}" --rbp-positional-features "${rbp_cache}" \
      --use-rbp-positional --disable-pair-edges --seed "${seed}"
  done
done

MULTILABEL_COMMON=(
  scripts/train_structdualrm_multilabel_relational.py
  --dataset "${MULTILABEL_DATASET}"
  --rnafold "${RNAFOLD}"
  --class-names Am,Cm,Gm,Um,m1A,m5C,m5U,m6A,m6Am,m7G,Y,AtoI
  --target-field label_set
  --fold-site-id-column source_site_id
  --sequence-column sequence_41nt
  --split split_cluster201
  --input-length 41
  --epochs 50
  --batch-size 1024
  --d-model 96
  --layers 2
  --heads 4
  --label-heads 4
  --dropout 0.2
  --lr 3e-4
  --weight-decay 1e-4
  --pos-weight-cap 100
  --negative-weight 0.2
  --gamma-pos 0
  --gamma-neg 2
  --center-loss bce
  --lambda-window 1
  --lambda-center 0
  --lambda-relation 0.1
  --lambda-cardinality 0.05
  --selection-metric macro_AUPR
  --early-stopping-patience 8
  --early-stopping-min-delta 1e-4
)

for seed in "${SEEDS[@]}"; do
  run_job "multilabel_seed${seed}" "${RUN_ROOT}/multilabel/seed${seed}" "${PYTHON_BIN}" \
    "${MULTILABEL_COMMON[@]}" --seed "${seed}"
done

date -Iseconds > "${STATE}/gpu_queue.ready"
