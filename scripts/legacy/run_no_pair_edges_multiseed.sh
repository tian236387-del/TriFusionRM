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
SEED42_MODEL_DIR="${SEED42_MODEL_DIR:-rmbase12_work/models/final_modality_factorial_ablation_seed42_20260726/full_no_pair_edges}"
OUT_ROOT="${OUT_ROOT:-rmbase12_work/models/no_pair_edges_multiseed_20260730}"
SUMMARY_DIR="${SUMMARY_DIR:-rmbase12_work/metrics/no_pair_edges_multiseed_20260730}"
SEEDS="${SEEDS:-17,73}"
EPOCHS="${EPOCHS:-50}"

for required in "${DATASET}" "${RNAFOLD}" "${RBP_POSITIONAL}" "${SEED42_MODEL_DIR}/metrics.json"; do
  if [[ ! -s "${required}" ]]; then
    echo "Missing required input: ${required}" >&2
    exit 1
  fi
done

mkdir -p "${OUT_ROOT}" "${SUMMARY_DIR}"
IFS=',' read -r -a seed_values <<< "${SEEDS}"
for seed in "${seed_values[@]}"; do
  out_dir="${OUT_ROOT}/seed${seed}"
  if [[ -s "${out_dir}/metrics.json" && -s "${out_dir}/test_predictions.tsv" ]]; then
    echo "Skip completed no-pair seed: ${seed}"
    continue
  fi
  mkdir -p "${out_dir}"
  resume_args=()
  if [[ -s "${out_dir}/checkpoint.pt" || -s "${out_dir}/model.pt" ]]; then
    resume_args=(--resume)
    echo "Resume interrupted no-pair seed: ${seed}"
  fi
  echo "===== no-pair three-modal model seed ${seed} ====="
  CUDA_VISIBLE_DEVICES="${CUDA_ID}" "${PYTHON_BIN}" -B scripts/train_motif_struct_fusion_v4.py \
    --dataset "${DATASET}" \
    --rnafold "${RNAFOLD}" \
    --rbp-positional-features "${RBP_POSITIONAL}" \
    --split split_cdhit80_cluster \
    --class-names Am,Cm,Gm,Um,m1A,m5C,m5U,m6A,m6Am,m7G,Y,AtoI \
    --input-length 41 \
    --sequence-column sequence_41nt \
    --seq-encoder multi_length \
    --multi-length-columns sequence_41nt,sequence_101nt,sequence_201nt \
    --multi-length-lengths 41,101,201 \
    --multi-length-base-encoder motif \
    --epochs "${EPOCHS}" \
    --batch-size 256 \
    --grad-accum-steps 4 \
    --d-model 128 \
    --gat-layers 2 \
    --dropout 0.4 \
    --lr 3e-4 \
    --weight-decay 1e-4 \
    --class-weight-power 0.5 \
    --loss ce \
    --aux-seq-weight 0.1 \
    --aux-struct-weight 0.05 \
    --length-aux-weight 0.05 \
    --gate-l1-weight 0.001 \
    --structure-scale 0.5 \
    --gate-init-logit -2.0 \
    --use-rbp-positional \
    --rbp-pos-scale 0.1 \
    --rbp-pos-gate-init-logit -1.5 \
    --rbp-pos-gate-l1-weight 0.0 \
    --disable-pair-edges \
    --disable-position \
    --selection-metric macro_AUPR_ovr \
    --early-stopping-patience 8 \
    --early-stopping-min-delta 0.0001 \
    --seed "${seed}" \
    --out-dir "${out_dir}" \
    "${resume_args[@]}"
done

"${PYTHON_BIN}" -B scripts/summarize_final_main_multiseed.py \
  --seed42-model-dir "${SEED42_MODEL_DIR}" \
  --run-root "${OUT_ROOT}" \
  --seeds "42,${SEEDS}" \
  --out-dir "${SUMMARY_DIR}"

echo "Done: ${SUMMARY_DIR}"
