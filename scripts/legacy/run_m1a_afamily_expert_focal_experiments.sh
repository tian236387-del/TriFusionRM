#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-${HOME}/anaconda3/envs/rmbaseline/bin/python}"
CUDA_ID="${CUDA_ID:-0}"
ROOT="${ROOT:-${HOME}/summer/m1A_m6A}"
OUT_ROOT="${OUT_ROOT:-rmbase12_work/models/multirm12_multi_length_fusion_cdhit80}"
METRICS_ROOT="${METRICS_ROOT:-rmbase12_work/metrics/multirm12_multi_length_fusion_cdhit80}"
CLASS_NAMES="Am,Cm,Gm,Um,m1A,m5C,m5U,m6A,m6Am,m7G,Y,AtoI"

cd "${ROOT}"

COMMON_ARGS=(
  --dataset rmbase12_work/datasets/rmbase_multirm12_hg38_41_101_201nt_cdhit80_common.tsv
  --rnafold rmbase12_work/structure/rnafold_41nt_multirm12_all_sites.tsv
  --split split_cdhit80_cluster
  --class-names "${CLASS_NAMES}"
  --input-length 41
  --sequence-column sequence_41nt
  --multi-length-columns sequence_41nt,sequence_101nt,sequence_201nt
  --multi-length-lengths 41,101,201
  --multi-length-base-encoder motif
  --seq-encoder multi_length
  --use-rbp-positional
  --rbp-positional-features rmbase12_work/rbp_encode_eclip_all/rmbase_multirm12_encode_eclip_top30_rbp_positional_features.tsv
  --epochs 50
  --batch-size 256
  --grad-accum-steps 4
  --d-model 128
  --dropout 0.4
  --lr 3e-4
  --weight-decay 1e-4
  --class-weight-power 0.5
  --loss ce
  --gat-layers 2
  --disable-position
  --aux-seq-weight 0.1
  --aux-struct-weight 0.05
  --length-aux-weight 0.05
  --gate-l1-weight 0.001
  --structure-scale 0.5
  --gate-init-logit -2.0
  --rbp-pos-scale 0.1
  --rbp-pos-gate-l1-weight 0.0
  --rbp-pos-gate-init-logit -1.5
  --afamily-classes Am,m1A,m6A,m6Am,AtoI
  --afamily-fusion-mode early_class
  --afamily-residual-scale 0.1
  --afamily-m1a-expert-scale 0.5
  --selection-metric macro_AUPR_ovr
  --selection-focus-class m1A
  --selection-focus-weight 0.2
  --early-stopping-patience 8
  --early-stopping-min-delta 0.0001
  --seed 42
  --resume
)

compare_variant() {
  local name="$1"
  "${PYTHON_BIN}" -B scripts/compare_prediction_intersection_per_class.py \
    --pred "current_main=${OUT_ROOT}/multi_length_independent/test_predictions.tsv" \
    --pred "evormd_adapted=rmbase12_work/official_reproduction_benchmark/multirm12_cdhit80_strict_split/evormd_adapted/test_predictions.tsv" \
    --pred "afamily_early=${OUT_ROOT}/afamily_early_class_w02_s01_focus02/test_predictions.tsv" \
    --pred "${name}=${OUT_ROOT}/${name}/test_predictions.tsv" \
    --classes "${CLASS_NAMES}" \
    --out "${METRICS_ROOT}/${name}_intersection_per_class.tsv"
}

run_variant() {
  local name="$1"
  shift
  local out_dir="${OUT_ROOT}/${name}"
  if [[ ! -s "${out_dir}/metrics.json" ]]; then
    mkdir -p "${out_dir}"
    CUDA_VISIBLE_DEVICES="${CUDA_ID}" "${PYTHON_BIN}" -B scripts/train_motif_struct_fusion_v4.py \
      "${COMMON_ARGS[@]}" \
      --out-dir "${out_dir}" \
      "$@"
  fi
  compare_variant "${name}"
}

# Controlled architecture ablation: only the dedicated m1A expert is new.
run_variant afamily_m1a_expert_s05 \
  --afamily-aux-weight 0.2 \
  --afamily-loss ce \
  --afamily-class-weight-power 0.25 \
  --afamily-max-class-weight 8.0 \
  --afamily-m1a-weight-multiplier 1.0

# Stronger local supervision: expert + moderate m1A weighting + focal hard-example emphasis.
run_variant afamily_m1a_expert_s05_focal_g15_w2 \
  --afamily-aux-weight 0.3 \
  --afamily-loss focal \
  --afamily-focal-gamma 1.5 \
  --afamily-class-weight-power 0.5 \
  --afamily-max-class-weight 8.0 \
  --afamily-m1a-weight-multiplier 2.0

