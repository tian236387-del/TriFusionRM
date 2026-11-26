#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-${HOME}/summer/m1A_m6A}"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/anaconda3/envs/rmbaseline/bin/python}"
CUDA_ID="${CUDA_ID:-0}"
OUT="rmbase12_work/detection_strict_20260814"
DATA="${OUT}/hg38_modified_vs_unannotated_strict_splits.tsv"
RNAFOLD="${OUT}/features/rnafold_41nt_modified_vs_unannotated.tsv"
RBP_MEMMAP="${OUT}/features/modified_vs_unannotated_top30_rbp_positional.memmap"
SPLIT="split_detection_cluster201"
SEED=42
GROUP_MAX_WEIGHT=100

cd "${ROOT}"
mkdir -p "${OUT}/logs" "${OUT}/state" "${OUT}/models"

while [[ ! -s "${OUT}/state/initial_models_seed42.complete" ]]; do
  printf '%s waiting_for_sequence_seed42\n' "$(date -Iseconds)"
  sleep 60
done
while [[ ! -s "${OUT}/state/rnafold.ready" ]]; do
  printf '%s waiting_for_rnafold\n' "$(date -Iseconds)"
  sleep 60
done

run_model() {
  local name="$1"
  shift
  local model_out="${OUT}/models/${name}_seed${SEED}"
  local log="${OUT}/logs/${name}_seed${SEED}.log"
  if [[ -s "${model_out}/audited/detection_metrics.json" ]]; then
    return
  fi
  mkdir -p "${model_out}"
  CUDA_VISIBLE_DEVICES="${CUDA_ID}" PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -B \
    scripts/train_motif_struct_fusion_v4.py \
    --dataset "${DATA}" \
    --class-names unmodified,modified \
    --rbp-features "" \
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
    --length-aux-weight 0.05 \
    --sampling-group-column original_mod_type \
    --sampling-group-power 1.0 \
    --sampling-group-max-weight "${GROUP_MAX_WEIGHT}" \
    --evaluation-group-column original_mod_type \
    --group-positive-class modified \
    --group-min-positive 20 \
    --selection-metric macro_group_AUPR \
    --selection-focus-weight 0.0 \
    --early-stopping-patience 5 \
    --early-stopping-min-delta 1e-4 \
    --disable-position \
    --seed "${SEED}" \
    "$@" 2>&1 | tee "${log}"
  "${PYTHON_BIN}" -B scripts/evaluate_hg38_detection_predictions.py \
    --dataset "${DATA}" \
    --valid-predictions "${model_out}/valid_predictions.tsv" \
    --test-predictions "${model_out}/test_predictions.tsv" \
    --output-dir "${model_out}/audited" \
    --split-column "${SPLIT}" 2>&1 | tee "${OUT}/logs/${name}_audit_seed${SEED}.log"
}

GROUP_MAX_WEIGHT=100
run_model sequence_group_balanced_cap100 \
  --rnafold "" \
  --disable-structure \
  --rbp-positional-features "" \
  --aux-struct-weight 0.0 \
  --gate-l1-weight 0.0

GROUP_MAX_WEIGHT=300
run_model sequence_group_balanced_cap300 \
  --rnafold "" \
  --disable-structure \
  --rbp-positional-features "" \
  --aux-struct-weight 0.0 \
  --gate-l1-weight 0.0

GROUP_MAX_WEIGHT=1000
run_model sequence_group_balanced_cap1000 \
  --rnafold "" \
  --disable-structure \
  --rbp-positional-features "" \
  --aux-struct-weight 0.0 \
  --gate-l1-weight 0.0

GROUP_MAX_WEIGHT="$("${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path

root = Path("rmbase12_work/detection_strict_20260814")
scores = {}
for cap in (100, 300, 1000):
    path = root / "models" / f"sequence_group_balanced_cap{cap}_seed42" / "audited" / "detection_metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    scores[cap] = float(payload["macro_original_modification_validation"]["macro_stable_AUPR"])
best = max(scores, key=lambda cap: (scores[cap], -cap))
(root / "state" / "group_sampling_selection.json").write_text(
    json.dumps(
        {
            "selection_rule": "maximum validation macro-stable AUPR; test metrics excluded",
            "candidate_max_weights": scores,
            "selected_max_weight": best,
        },
        indent=2,
    ),
    encoding="utf-8",
)
print(best)
PY
)"
printf '%s\n' "${GROUP_MAX_WEIGHT}" > "${OUT}/state/selected_group_max_weight.txt"
printf '%s selected_group_max_weight=%s\n' "$(date -Iseconds)" "${GROUP_MAX_WEIGHT}"

run_model sequence_structure \
  --rnafold "${RNAFOLD}" \
  --rbp-positional-features "" \
  --aux-struct-weight 0.05 \
  --structure-scale 0.5 \
  --gate-l1-weight 0.001

while [[ ! -s "${OUT}/state/rbp_memmap.ready" ]]; do
  printf '%s waiting_for_rbp_memmap\n' "$(date -Iseconds)"
  sleep 60
done

run_model sequence_rbp \
  --rnafold "" \
  --disable-structure \
  --rbp-positional-features "${RBP_MEMMAP}" \
  --use-rbp-positional \
  --aux-struct-weight 0.0 \
  --gate-l1-weight 0.0 \
  --rbp-pos-scale 0.1 \
  --rbp-pos-gate-init-logit -1.5 \
  --rbp-pos-gate-l1-weight 0.0

run_model full \
  --rnafold "${RNAFOLD}" \
  --rbp-positional-features "${RBP_MEMMAP}" \
  --use-rbp-positional \
  --aux-struct-weight 0.05 \
  --structure-scale 0.5 \
  --gate-l1-weight 0.001 \
  --rbp-pos-scale 0.1 \
  --rbp-pos-gate-init-logit -1.5 \
  --rbp-pos-gate-l1-weight 0.0

run_model full_no_pair_edges \
  --rnafold "${RNAFOLD}" \
  --rbp-positional-features "${RBP_MEMMAP}" \
  --use-rbp-positional \
  --disable-pair-edges \
  --aux-struct-weight 0.05 \
  --structure-scale 0.5 \
  --gate-l1-weight 0.001 \
  --rbp-pos-scale 0.1 \
  --rbp-pos-gate-init-logit -1.5 \
  --rbp-pos-gate-l1-weight 0.0

date -Iseconds > "${OUT}/state/seed42_modalities.complete"
printf '%s detection_seed42_modalities_complete\n' "$(date -Iseconds)"
