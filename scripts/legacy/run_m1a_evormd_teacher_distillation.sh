#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-${HOME}/anaconda3/envs/rmbaseline/bin/python}"
CUDA_ID="${CUDA_ID:-0}"
ROOT="${ROOT:-${HOME}/summer/m1A_m6A}"
OUT_ROOT="rmbase12_work/models/multirm12_multi_length_fusion_cdhit80"
NAME="m1a_evormd_full_teacher_distill_ft"
OUT_DIR="${OUT_ROOT}/${NAME}"
METRICS_DIR="rmbase12_work/metrics/multirm12_multi_length_fusion_cdhit80"
CLASS_NAMES="Am,Cm,Gm,Um,m1A,m5C,m5U,m6A,m6Am,m7G,Y,AtoI"

cd "${ROOT}"

if [[ ! -s "${OUT_DIR}/metrics.json" ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_ID}" "${PYTHON_BIN}" -B scripts/train_motif_struct_fusion_v4.py \
    --dataset rmbase12_work/datasets/rmbase_multirm12_hg38_41_101_201nt_cdhit80_common.tsv \
    --rnafold rmbase12_work/structure/rnafold_41nt_multirm12_all_sites.tsv \
    --split split_cdhit80_cluster \
    --class-names "${CLASS_NAMES}" \
    --input-length 41 \
    --sequence-column sequence_41nt \
    --multi-length-columns sequence_41nt,sequence_101nt,sequence_201nt \
    --multi-length-lengths 41,101,201 \
    --multi-length-base-encoder motif \
    --seq-encoder multi_length \
    --use-rbp-positional \
    --rbp-positional-features rmbase12_work/rbp_encode_eclip_all/rmbase_multirm12_encode_eclip_top30_rbp_positional_features.tsv \
    --epochs 8 \
    --batch-size 256 \
    --grad-accum-steps 4 \
    --d-model 128 \
    --dropout 0.4 \
    --lr 5e-5 \
    --weight-decay 1e-4 \
    --class-weight-power 0.5 \
    --loss ce \
    --gat-layers 2 \
    --disable-position \
    --aux-seq-weight 0.1 \
    --aux-struct-weight 0.05 \
    --length-aux-weight 0.05 \
    --gate-l1-weight 0.001 \
    --structure-scale 0.5 \
    --gate-init-logit -2.0 \
    --rbp-pos-scale 0.1 \
    --rbp-pos-gate-l1-weight 0.0 \
    --rbp-pos-gate-init-logit -1.5 \
    --distill-teacher-predictions rmbase12_work/metrics/evormd_m1a_counterfactual/train_teacher_m1a.tsv \
    --distill-teacher-field teacher_full_m1A \
    --distill-m1a-weight 0.03 \
    --distill-m1a-rank-weight 0.10 \
    --distill-temperature 2.0 \
    --distill-teacher-positive-weight 20.0 \
    --distill-rank-pairs 32 \
    --distill-min-teacher-gap 0.05 \
    --init-model-path "${OUT_ROOT}/multi_length_independent/model.pt" \
    --selection-metric macro_AUPR_ovr \
    --selection-focus-class m1A \
    --selection-focus-weight 0.5 \
    --early-stopping-patience 4 \
    --early-stopping-min-delta 0.0001 \
    --seed 42 \
    --resume \
    --out-dir "${OUT_DIR}"
fi

"${PYTHON_BIN}" -B scripts/compare_prediction_intersection_per_class.py \
  --pred "current_main=${OUT_ROOT}/multi_length_independent/test_predictions.tsv" \
  --pred "evormd_adapted=rmbase12_work/official_reproduction_benchmark/multirm12_cdhit80_strict_split/evormd_adapted/test_predictions.tsv" \
  --pred "distilled=${OUT_DIR}/test_predictions.tsv" \
  --classes "${CLASS_NAMES}" \
  --out "${METRICS_DIR}/${NAME}_intersection_per_class.tsv"

