#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PYTHON_BIN:-}" && -x "${HOME}/anaconda3/envs/rmbaseline/bin/python" ]]; then
  PYTHON_BIN="${HOME}/anaconda3/envs/rmbaseline/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

CUDA_ID="${CUDA_ID:-0}"
SWEEP_PID="${SWEEP_PID:?SWEEP_PID is required}"
SWEEP_ROOT="${SWEEP_ROOT:?SWEEP_ROOT is required}"
SWEEP_SUMMARY="${SWEEP_SUMMARY:?SWEEP_SUMMARY is required}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT is required}"
SUMMARY_DIR="${SUMMARY_DIR:?SUMMARY_DIR is required}"
EPOCHS="${EPOCHS:-35}"

while [[ ! -s "${SWEEP_SUMMARY}/best_validation_run.txt" ]]; do
  if ! kill -0 "${SWEEP_PID}" 2>/dev/null; then
    echo "Sweep process ${SWEEP_PID} ended without a best-run file" >&2
    exit 1
  fi
  sleep 300
done

BEST_RUN=$(tr -d '\r\n' < "${SWEEP_SUMMARY}/best_validation_run.txt")
case "${BEST_RUN}" in
  lr2e4_d035_wd1e4_cw045) LR=2e-4; DROPOUT=0.35; WEIGHT_DECAY=1e-4; CLASS_POWER=0.45; FOCUS_WEIGHT=0.10 ;;
  lr15e5_d035_wd1e4_cw040) LR=1.5e-4; DROPOUT=0.35; WEIGHT_DECAY=1e-4; CLASS_POWER=0.40; FOCUS_WEIGHT=0.10 ;;
  lr2e4_d030_wd5e4_cw040) LR=2e-4; DROPOUT=0.30; WEIGHT_DECAY=5e-4; CLASS_POWER=0.40; FOCUS_WEIGHT=0.15 ;;
  lr1e4_d040_wd5e5_cw045) LR=1e-4; DROPOUT=0.40; WEIGHT_DECAY=5e-5; CLASS_POWER=0.45; FOCUS_WEIGHT=0.15 ;;
  *) echo "Unknown best run: ${BEST_RUN}" >&2; exit 2 ;;
esac

mkdir -p "${OUT_ROOT}" "${SUMMARY_DIR}"

COMMON_ARGS=(
  --dataset rmbase12_work/datasets/rmbase_multirm12_hg38_41_101_201nt_cdhit80_common.tsv
  --rnafold rmbase12_work/structure/rnafold_41nt_multirm12_all_sites.tsv
  --rbp-positional-features rmbase12_work/rbp_encode_eclip_all/rmbase_multirm12_encode_eclip_top30_rbp_positional_features.tsv
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
  --dropout "${DROPOUT}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --class-weight-power "${CLASS_POWER}"
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
  --selection-focus-weight "${FOCUS_WEIGHT}"
  --early-stopping-patience 6
  --early-stopping-min-delta 0.0001
)

for seed in 17 73; do
  out_dir="${OUT_ROOT}/${BEST_RUN}_seed${seed}"
  if [[ -e "${out_dir}/checkpoint.pt" || -e "${out_dir}/model.pt" ]]; then
    echo "Refusing to reuse an existing multi-seed run: ${out_dir}" >&2
    exit 3
  fi
  mkdir -p "${out_dir}"
  echo "===== best=${BEST_RUN} seed=${seed} ====="
  CUDA_VISIBLE_DEVICES="${CUDA_ID}" "${PYTHON_BIN}" -B scripts/train_motif_struct_fusion_v4.py \
    "${COMMON_ARGS[@]}" \
    --seed "${seed}" \
    --out-dir "${out_dir}"
done

"${PYTHON_BIN}" -B scripts/summarize_main_hparam_multiseed.py \
  --seed42-metrics "${SWEEP_ROOT}/${BEST_RUN}/metrics.json" \
  --additional-root "${OUT_ROOT}" \
  --out-dir "${SUMMARY_DIR}" \
  --best-run "${BEST_RUN}"

echo "Done: ${SUMMARY_DIR}"
