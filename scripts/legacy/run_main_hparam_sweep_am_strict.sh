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
OUT_ROOT="${OUT_ROOT:-rmbase12_work/models/main_hparam_sweep_am_strict_seed42}"
SUMMARY_DIR="${SUMMARY_DIR:-rmbase12_work/metrics/main_hparam_sweep_am_strict_seed42}"
EPOCHS="${EPOCHS:-35}"
SEED="${SEED:-42}"

for required in "${DATASET}" "${RNAFOLD}" "${RBP_POSITIONAL}"; do
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
  --use-rbp-positional
  --disable-position
  --selection-metric macro_AUPR_ovr
  --selection-focus-class Am
  --selection-focus-metric F1
  --early-stopping-patience 6
  --early-stopping-min-delta 0.0001
  --seed "${SEED}"
)

run_variant() {
  local name="$1"
  local lr="$2"
  local dropout="$3"
  local weight_decay="$4"
  local class_weight_power="$5"
  local focus_weight="$6"
  local out_dir="${OUT_ROOT}/${name}"

  if [[ -e "${out_dir}/checkpoint.pt" || -e "${out_dir}/model.pt" ]]; then
    echo "Refusing to reuse an existing sweep run: ${out_dir}" >&2
    exit 2
  fi
  mkdir -p "${out_dir}"
  echo "===== ${name}: lr=${lr} dropout=${dropout} wd=${weight_decay} class_power=${class_weight_power} ====="
  CUDA_VISIBLE_DEVICES="${CUDA_ID}" "${PYTHON_BIN}" -B scripts/train_motif_struct_fusion_v4.py \
    "${COMMON_ARGS[@]}" \
    --out-dir "${out_dir}" \
    --lr "${lr}" \
    --dropout "${dropout}" \
    --weight-decay "${weight_decay}" \
    --class-weight-power "${class_weight_power}" \
    --selection-focus-weight "${focus_weight}"
}

run_variant lr2e4_d035_wd1e4_cw045 2e-4 0.35 1e-4 0.45 0.10
run_variant lr15e5_d035_wd1e4_cw040 1.5e-4 0.35 1e-4 0.40 0.10
run_variant lr2e4_d030_wd5e4_cw040 2e-4 0.30 5e-4 0.40 0.15
run_variant lr1e4_d040_wd5e5_cw045 1e-4 0.40 5e-5 0.45 0.15

"${PYTHON_BIN}" -B scripts/summarize_main_hparam_sweep.py \
  --run-root "${OUT_ROOT}" \
  --control rmbase12_work/models/multirm12_multi_length_fusion_cdhit80/multi_length_independent/metrics.json \
  --out-dir "${SUMMARY_DIR}"

echo "Done: ${SUMMARY_DIR}"
