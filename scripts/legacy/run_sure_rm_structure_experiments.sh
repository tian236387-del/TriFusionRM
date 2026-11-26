#!/usr/bin/env bash
set -euo pipefail

# SURE-RM staged structure experiments on the strict hg38 matched-candidate task.
# Pilot usage:
#   CUDA_ID=0 SEEDS=42 RUN_CONTROLS=1 RUN_MULTISCALE=0 bash scripts/run_sure_rm_structure_experiments.sh
# Five-seed confirmation:
#   CUDA_ID=0 SEEDS="42 52 62 72 82" RUN_CONTROLS=1 RUN_MULTISCALE=1 bash scripts/run_sure_rm_structure_experiments.sh

ROOT="${ROOT:-${HOME}/summer/m1A_m6A}"
TORCH_PYTHON="${TORCH_PYTHON:-${HOME}/anaconda3/envs/rmbaseline/bin/python}"
RNA_PYTHON="${RNA_PYTHON:-${HOME}/anaconda3/envs/RNAfold/bin/python}"
CUDA_ID="${CUDA_ID:-0}"
SEEDS="${SEEDS:-42}"
RUN_CONTROLS="${RUN_CONTROLS:-1}"
RUN_MULTISCALE="${RUN_MULTISCALE:-0}"
RNA_WORKERS="${RNA_WORKERS:-8}"

OUT="rmbase12_work/detection_strict_20260814"
DATA="${OUT}/hg38_modified_vs_unannotated_strict_splits.tsv"
FEATURES="${OUT}/features/sure_rm"
MODELS="${OUT}/models"
LOGS="${OUT}/logs/sure_rm"
STATE="${OUT}/state/sure_rm"
SPLIT="split_detection_cluster201"

cd "${ROOT}"
mkdir -p "${FEATURES}" "${MODELS}" "${LOGS}" "${STATE}"
test -s "${DATA}"

build_probability_features() {
  local length="$1"
  local sequence_column="sequence_${length}nt"
  local output="${FEATURES}/viennarna_probability_${length}nt.tsv"
  local marker="${STATE}/viennarna_probability_${length}nt.ready"
  if [[ -s "${marker}" && -s "${output}" ]]; then
    return
  fi
  PYTHONUNBUFFERED=1 "${RNA_PYTHON}" -B scripts/build_viennarna_probability_structure.py \
    --dataset "${DATA}" \
    --out "${output}" \
    --sequence-column "${sequence_column}" \
    --pair-threshold 0.5 \
    --candidate-threshold 0.05 \
    --max-pair-candidates 3 \
    --workers "${RNA_WORKERS}" \
    --chunk-size 5000 \
    --worker-chunksize 32 \
    --cache-db "${FEATURES}/viennarna_probability_${length}nt.cache.sqlite3" \
    --progress-every 10000 \
    --resume 2>&1 | tee "${LOGS}/build_probability_${length}nt.log"
  test -s "${output}"
  date -Iseconds > "${marker}"
}

build_probability_features 41

PROB41="${FEATURES}/viennarna_probability_41nt.tsv"
MULTISCALE="${FEATURES}/viennarna_probability_multiscale_41_101_201.tsv"
MULTISCALE_MISALIGNED="${FEATURES}/viennarna_probability_multiscale_misaligned_control.tsv"
if [[ "${RUN_MULTISCALE}" == "1" ]]; then
  build_probability_features 101
  build_probability_features 201
  if [[ ! -s "${STATE}/viennarna_probability_multiscale.ready" || ! -s "${MULTISCALE}" ]]; then
    "${TORCH_PYTHON}" -B scripts/build_multiscale_probability_structure.py \
      --structure-files "${PROB41},${FEATURES}/viennarna_probability_101nt.tsv,${FEATURES}/viennarna_probability_201nt.tsv" \
      --window-lengths 41,101,201 \
      --target-length 41 \
      --out "${MULTISCALE}" \
      --progress-every 100000 2>&1 | tee "${LOGS}/build_multiscale.log"
    test -s "${MULTISCALE}"
    date -Iseconds > "${STATE}/viennarna_probability_multiscale.ready"
  fi
  if [[ "${RUN_CONTROLS}" == "1" && ( ! -s "${STATE}/viennarna_probability_multiscale_misaligned.ready" || ! -s "${MULTISCALE_MISALIGNED}" ) ]]; then
    "${TORCH_PYTHON}" -B scripts/build_multiscale_probability_structure.py \
      --structure-files "${PROB41},${FEATURES}/viennarna_probability_101nt.tsv,${FEATURES}/viennarna_probability_201nt.tsv" \
      --window-lengths 41,101,201 \
      --target-length 41 \
      --shuffle-non-target-scales \
      --shuffle-offset 997 \
      --out "${MULTISCALE_MISALIGNED}" \
      --progress-every 100000 2>&1 | tee "${LOGS}/build_multiscale_misaligned.log"
    test -s "${MULTISCALE_MISALIGNED}"
    date -Iseconds > "${STATE}/viennarna_probability_multiscale_misaligned.ready"
  fi
fi

BASE_NODE_FIELDS="paired_flags,unpaired_probability_flags,pair_entropy_flags,expected_partner_distance_flags,pair_edge_probability"
BASE_SCALAR_FIELDS="ensemble_free_energy,ensemble_diversity,center_unpaired_probability,mean_unpaired_probability,center_pair_entropy,central_11nt_pair_entropy,central_21nt_pair_entropy,mean_pair_entropy,center_expected_partner_distance,central_11nt_expected_partner_distance,central_21nt_expected_partner_distance,mean_expected_partner_distance"
MULTISCALE_NODE_FIELDS="${BASE_NODE_FIELDS},multiscale_mean_paired_flags,multiscale_mean_unpaired_flags,multiscale_mean_entropy_flags,multiscale_mean_expected_partner_distance_flags,multiscale_pair_variance_flags"
MULTISCALE_SCALAR_FIELDS="${BASE_SCALAR_FIELDS},multiscale_center_paired_mean,multiscale_center_paired_variance,multiscale_center_unpaired_mean,multiscale_center_entropy_mean,multiscale_center_entropy_max,multiscale_profile_jsd,multiscale_pairing_agreement,multiscale_structure_confidence,multiscale_window_count"

run_model() {
  local name="$1"
  local seed="$2"
  shift 2
  local model_out="${MODELS}/${name}_seed${seed}"
  local log="${LOGS}/${name}_seed${seed}.log"
  if [[ -s "${model_out}/audited/detection_metrics.json" ]]; then
    return
  fi
  mkdir -p "${model_out}"
  CUDA_VISIBLE_DEVICES="${CUDA_ID}" PYTHONUNBUFFERED=1 "${TORCH_PYTHON}" -B \
    scripts/train_motif_struct_fusion_v4.py \
    --dataset "${DATA}" \
    --class-names unmodified,modified \
    --rbp-features "" \
    --rbp-positional-features "" \
    --split "${SPLIT}" \
    --out-dir "${model_out}" \
    --input-length 41 \
    --sequence-column sequence_41nt \
    --epochs 30 \
    --batch-size 256 \
    --grad-accum-steps 4 \
    --d-model 128 \
    --gat-layers 2 \
    --dropout 0.4 \
    --seq-encoder multi_length \
    --multi-length-columns sequence_41nt,sequence_101nt,sequence_201nt \
    --multi-length-lengths 41,101,201 \
    --multi-length-base-encoder motif \
    --afamily-classes "" \
    --lr 3e-4 \
    --weight-decay 1e-4 \
    --class-weight-power 0.0 \
    --loss ce \
    --aux-seq-weight 0.1 \
    --aux-struct-weight 0.05 \
    --length-aux-weight 0.05 \
    --sampling-group-column original_mod_type \
    --sampling-group-power 1.0 \
    --sampling-group-max-weight 300 \
    --evaluation-group-column original_mod_type \
    --group-positive-class modified \
    --group-min-positive 20 \
    --selection-metric macro_group_AUPR \
    --selection-focus-weight 0.0 \
    --early-stopping-patience 5 \
    --early-stopping-min-delta 1e-4 \
    --structure-scale 0.5 \
    --gate-l1-weight 0.001 \
    --gate-init-logit -2.0 \
    --max-pair-candidates 3 \
    --confidence-gated-structure \
    --structure-confidence-floor 0.1 \
    --disable-position \
    --seed "${seed}" \
    "$@" 2>&1 | tee "${log}"
  "${TORCH_PYTHON}" -B scripts/evaluate_hg38_detection_predictions.py \
    --dataset "${DATA}" \
    --valid-predictions "${model_out}/valid_predictions.tsv" \
    --test-predictions "${model_out}/test_predictions.tsv" \
    --output-dir "${model_out}/audited" \
    --split-column "${SPLIT}" 2>&1 | tee "${LOGS}/${name}_audit_seed${seed}.log"
}

for seed in ${SEEDS}; do
  run_model sure_s2_probability_nodes "${seed}" \
    --rnafold "${PROB41}" \
    --node-feature-fields "${BASE_NODE_FIELDS}" \
    --extra-struct-fields "${BASE_SCALAR_FIELDS}" \
    --disable-pair-edges

  run_model sure_s3_probability_graph "${seed}" \
    --rnafold "${PROB41}" \
    --node-feature-fields "${BASE_NODE_FIELDS}" \
    --extra-struct-fields "${BASE_SCALAR_FIELDS}"

  if [[ "${RUN_CONTROLS}" == "1" ]]; then
    run_model sure_control_sample_shuffled "${seed}" \
      --rnafold "${PROB41}" \
      --node-feature-fields "${BASE_NODE_FIELDS}" \
      --extra-struct-fields "${BASE_SCALAR_FIELDS}" \
      --shuffle-structure
    run_model sure_control_node_shuffled "${seed}" \
      --rnafold "${PROB41}" \
      --node-feature-fields "${BASE_NODE_FIELDS}" \
      --extra-struct-fields "${BASE_SCALAR_FIELDS}" \
      --shuffle-structure-nodes
    run_model sure_control_weight_shuffled "${seed}" \
      --rnafold "${PROB41}" \
      --node-feature-fields "${BASE_NODE_FIELDS}" \
      --extra-struct-fields "${BASE_SCALAR_FIELDS}" \
      --shuffle-pair-weights
    run_model sure_control_degree_rewired "${seed}" \
      --rnafold "${PROB41}" \
      --node-feature-fields "${BASE_NODE_FIELDS}" \
      --extra-struct-fields "${BASE_SCALAR_FIELDS}" \
      --rewire-pair-edges
    run_model sure_control_capacity_cnn "${seed}" \
      --rnafold "${PROB41}" \
      --node-feature-fields "${BASE_NODE_FIELDS}" \
      --extra-struct-fields "${BASE_SCALAR_FIELDS}" \
      --structure-encoder cnn_control
  fi

  if [[ "${RUN_MULTISCALE}" == "1" ]]; then
    run_model sure_s4_multiscale_probability "${seed}" \
      --rnafold "${MULTISCALE}" \
      --node-feature-fields "${MULTISCALE_NODE_FIELDS}" \
      --extra-struct-fields "${MULTISCALE_SCALAR_FIELDS}"
    if [[ "${RUN_CONTROLS}" == "1" ]]; then
      run_model sure_control_multiscale_misaligned "${seed}" \
        --rnafold "${MULTISCALE_MISALIGNED}" \
        --node-feature-fields "${MULTISCALE_NODE_FIELDS}" \
        --extra-struct-fields "${MULTISCALE_SCALAR_FIELDS}"
    fi
  fi
done

"${TORCH_PYTHON}" -B scripts/summarize_hg38_detection_experiments.py \
  --models-dir "${MODELS}" \
  --output-dir "${OUT}/summary" 2>&1 | tee "${LOGS}/summarize.log"
"${TORCH_PYTHON}" -B scripts/summarize_sure_rm_structure_experiments.py \
  --models-dir "${MODELS}" \
  --output-dir "${OUT}/summary/sure_rm" 2>&1 | tee "${LOGS}/summarize_sure_rm.log"

date -Iseconds > "${STATE}/experiments.complete"
printf '%s sure_rm_structure_experiments_complete\n' "$(date -Iseconds)"
