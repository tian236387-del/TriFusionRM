# StructDualRM: final no-pair model and historical inference bundle

This bundle contains the final no-pair StructDualRM seed42 model, not the older pair-edge reference and not the shuffled-RBP control. No manuscript files are included or modified.

## Contents

- `src/train_motif_struct_fusion_v4.py`: model, multi-length/RBP dataloaders, training and fusion.
- `src/train_motif_struct_fusion_v3.py`: base/2-mer/3-mer sequence embeddings and CNN-BiGRU encoder.
- `src/train_paired_gat_rbp_fusion.py`: shared dataset, graph layers, feature normalization and utilities. Its filename does not imply pairing edges are enabled in this checkpoint.
- `checkpoints/seed42/model.pt`: validation-selected inference checkpoint. Best epoch: 18. Test Macro AUPR: 0.9297232736546807, over 11 supported classes.
- `checkpoints/seed42/training_config.json`: original training configuration, including original server paths. `centralized_config.json` records relocated server paths; neither is a portable local path configuration.
- `historical/test_predictions_seed42.tsv`: all 108,211 archived test predictions, with true/predicted labels and twelve probabilities.
- `examples/`: 22 real test sites (two per supported class), matched RNAfold features, RBP tensors in TSV form, RBP ordering metadata, historical predictions and a runnable inference script.
- `audits/`, `source_provenance.json`, `SHA256SUMS.json`: identity, source and file-integrity records.

## Run the checked example

Use Python 3.9 or a compatible environment with PyTorch, NumPy and scikit-learn. `environment.json` and `requirements.txt` record the environment in which the example was verified, not the unverified historical training environment. Install an appropriate PyTorch build for your OS/device; CPU is sufficient for the example.

From the extracted bundle root:

```sh
python examples/run_inference.py
```

The script loads checkpoint normalization statistics and original class order, disables pairing-partner edges explicitly, runs all three sequence lengths and compares probabilities with historical outputs (absolute tolerance 1e-5). It writes `examples/reproduced_predictions.tsv` and `examples/reproduction_check.json`. This verifies a real-input example, not full-dataset reproducibility or performance on new data.

The archived `src/export_motif_struct_fusion_v4_predictions.py` is preserved unchanged for provenance. Its generic `build_dataset` does not forward `disable_pair_edges`; do not use its standalone CLI unmodified for final no-pair inference. Use the checked example loader, which explicitly sets `disable_pair_edges=True`.

## Preprocessing specification

1. Dataset and labels: human GRCh38/hg38 RMBase-derived sites. Coordinate/strand records were deduplicated within modification type; sequence and central-base checks were applied. Source Tm, Y and A-to-I labels map to Um, Y (displayed as Psi) and AtoI (displayed as I). The model has twelve outputs, but m6Am has no positives in the frozen test set. See `preprocessing/label_mapping.json` and the dataset audit.
2. Split: exact-unique 41-nt sequences were clustered with CD-HIT-est v4.8.1, `-c 0.8 -n 5 -G 1 -aS 0.8 -aL 0.8`. Whole clusters were assigned with seed42, approximately 80:10:10. Centered 101-nt and 201-nt windows were extracted from GRCh38, reverse-complemented on the minus strand and required to agree with the 41-nt center. All lengths inherit `split_cdhit80_cluster`; they were not independently reclustered. Frozen counts: 865,582 train, 108,251 validation, 108,211 test.
3. Sequence tokens: A=1, C=2, G=3, U/T=4, unknown/padding=0. Overlapping 2-/3-mer IDs are base-5 encodings anchored at the starting base, with right-boundary zero padding. The 5/25/125-entry embedding tables map to 128 dimensions; embeddings are added and layer-normalized. Preserve the original code rather than replacing this with a different tokenizer.
4. Structure: RNAfold on the centered 41-nt sequence supplies per-base paired flags and six descriptors (center pairing, paired fractions over 11/21/41 nt, MFE per nucleotide, center loop span). Use checkpoint `struct_stats`, never fit normalization on test data. Node features are pairing flags, not nucleotide embeddings. The final graph retains self/backbone neighbors and disables explicit pairing-partner edges. `src/run_rmbase11_rnafold.py` is the archived preprocessing implementation; generating new features requires the external RNAfold executable.
5. RBP: ENCODE eCLIP evidence for 30 RBPs, six signed genomic bins [-200,-100), [-100,-50), [-50,0), [0,50), [50,100), [100,200). Bin order is not reversed on minus-strand sites. Seven features: peak presence, log1p peak count, log1p maximum signal, log1p mean signal, summit closeness, capped fractional overlap and strand-match fraction. Preserve the exact RBP/bin/feature order in `examples/rbp_positional.tsv.json`. `src/build_rbp_positional_features.py` is the original feature builder. Genome-wide eCLIP peaks are not included.
6. Inference: load `model_state_dict` strictly, apply twelve-way softmax, retain the checkpoint label order. Do not renormalize over the classes appearing in the example. RBP tensors and RNAfold features must be joined by `site_id`, never by row position alone.

## Scope

The archive includes a complete seed42 inference checkpoint and small, fully paired example inputs; it does not include all training sequences, genome reference, original RMBase downloads, genome-wide RBP features or optimizer/resume states. Source server locations are recorded in `source_provenance.json` and original configs. The complete historical test prediction table is included. The 22-site example is a deterministic demonstration and must not be reported as a new evaluation set.
