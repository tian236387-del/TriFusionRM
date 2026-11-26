#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PYTHON_BIN:-}" && -x "${HOME}/anaconda3/envs/rmbaseline/bin/python" ]]; then
  PYTHON_BIN="${HOME}/anaconda3/envs/rmbaseline/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

CUDA_ID="${CUDA_ID:-0}"
DATASET="${DATASET:-rmbase12_work/datasets/rmbase_multirm12_hg38_41_101_201nt_cdhit80_common.tsv}"
RNAFOLD="${RNAFOLD:-rmbase12_work/structure/rnafold_41nt_multirm12_all_sites.tsv}"
RBP_POSITIONAL="${RBP_POSITIONAL:-rmbase12_work/rbp_encode_eclip_all/rmbase_multirm12_encode_eclip_top30_rbp_positional_features.tsv}"
FULL_MODEL_DIR="${FULL_MODEL_DIR:-rmbase12_work/models/multirm12_multi_length_fusion_cdhit80/multi_length_independent}"
OUT_ROOT="${OUT_ROOT:-rmbase12_work/models/final_modality_factorial_ablation_seed42}"
SUMMARY_DIR="${SUMMARY_DIR:-rmbase12_work/metrics/final_modality_factorial_ablation_seed42}"
EPOCHS="${EPOCHS:-50}"
SEED="${SEED:-42}"
BOOTSTRAP_ITERATIONS="${BOOTSTRAP_ITERATIONS:-500}"

for required in "${DATASET}" "${RNAFOLD}" "${RBP_POSITIONAL}" "${FULL_MODEL_DIR}/metrics.json"; do
  if [[ ! -s "${required}" ]]; then
    echo "Missing required input: ${required}" >&2
    exit 1
  fi
done

mkdir -p "${OUT_ROOT}" "${SUMMARY_DIR}"

COMMON_ARGS=(
  --dataset "${DATASET}"
  --rnafold "${RNAFOLD}"
  --rbp-positional-features "${RBP_POSITIONAL}"
  --split split_cdhit80_cluster
  --class-names Am,Cm,Gm,Um,m1A,m5C,m5U,m6A,m6Am,m7G,Y,AtoI
  --input-length 41
  --sequence-column sequence_41nt
  --seq-encoder multi_length
  --multi-length-columns sequence_41nt,sequence_101nt,sequence_201nt
  --multi-length-lengths 41,101,201
  --multi-length-base-encoder motif
  --epochs "${EPOCHS}"
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
  --early-stopping-min-delta 0.0001
  --seed "${SEED}"
)

run_variant() {
  local name="$1"
  shift
  local out_dir="${OUT_ROOT}/${name}"
  if [[ -s "${out_dir}/metrics.json" && -s "${out_dir}/test_predictions.tsv" ]]; then
    echo "Skip completed variant: ${out_dir}"
    return 0
  fi
  mkdir -p "${out_dir}"
  local resume_args=()
  if [[ -s "${out_dir}/checkpoint.pt" || -s "${out_dir}/model.pt" ]]; then
    resume_args=(--resume)
    echo "Resume interrupted variant: ${out_dir}"
  fi
  echo "===== final modality ablation: ${name} ====="
  CUDA_VISIBLE_DEVICES="${CUDA_ID}" "${PYTHON_BIN}" -B scripts/train_motif_struct_fusion_v4.py \
    "${COMMON_ARGS[@]}" \
    --out-dir "${out_dir}" \
    "${resume_args[@]}" \
    "$@"
}

run_variant sequence_only --disable-structure
run_variant sequence_structure
run_variant sequence_rbp --disable-structure --use-rbp-positional
run_variant full_shuffled_structure --use-rbp-positional --shuffle-structure
run_variant full_shuffled_rbp --use-rbp-positional --shuffle-rbp
run_variant full_no_pair_edges --use-rbp-positional --disable-pair-edges
run_variant full_shuffled_rbp_bins --use-rbp-positional --shuffle-rbp-bins

"${PYTHON_BIN}" -B scripts/summarize_final_modality_factorial_ablation.py \
  --run-root "${OUT_ROOT}" \
  --full-model-dir "${FULL_MODEL_DIR}" \
  --out-dir "${SUMMARY_DIR}"

"${PYTHON_BIN}" -B scripts/paired_bootstrap_final_modality.py \
  --run-root "${OUT_ROOT}" \
  --full-model-dir "${FULL_MODEL_DIR}" \
  --out-dir "${SUMMARY_DIR}" \
  --iterations "${BOOTSTRAP_ITERATIONS}"

"${PYTHON_BIN}" -B scripts/analyze_final_modality_interpretability.py \
  --full-model-dir "${FULL_MODEL_DIR}" \
  --sequence-only-dir "${OUT_ROOT}/sequence_only" \
  --rnafold "${RNAFOLD}" \
  --out-dir "${SUMMARY_DIR}"

echo "Done: ${SUMMARY_DIR}"
