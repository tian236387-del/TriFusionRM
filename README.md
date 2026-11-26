# TriFusionRM

Code, processed data and supporting materials for RNA modification-type classification using multi-length sequence, predicted RNA structure and positional RBP evidence.

## Contents

- `src/`: final no-pair model and prediction utilities.
- `examples/`: real archived inputs, historical predictions and an inference check.
- `scripts/`: preprocessing, experiments, statistics and plotting; `legacy/` and `experiments/` retain historical workflows.
- `configs/`: archived parameter configurations and historical environment records.
- `preprocessing/`: label mapping and dataset provenance.
- `data/`: dataset documentation, release links and SHA256 checksums.
- `results/`: available motif, bootstrap and t-SNE tables.
- `figures/`: manuscript figures and alternative t-SNE parameter settings.
- `manuscript/`: current LaTeX source and OUP template.
- `supplementary/`: supplementary Word and PDF files.

## Installation and example inference

Use Python 3.10 or 3.11 and install `pip install -r requirements.txt`.
Download the final no-pair bundle from the [v1.0.0 release](https://github.com/tian236387-del/TriFusionRM/releases/tag/v1.0.0).
Extract its `checkpoints/` directory at this repository root, then run `python examples/run_inference.py`.
The bundle also retains its original folder layout for standalone use.

The portable dependency ranges have not yet been validated in a fresh environment. Historical dependency records are retained in `configs/`; the original list contains a NumPy/SciPy version conflict and should not be installed unchanged.

## Reproducibility and provenance

The final model is the no-pair seed42 configuration. Internal identifiers containing `StructDualRM` and original class names are retained for checkpoint compatibility. Source archives preserve server paths; historical experiment scripts require their input/output paths to be adapted to the local installation. They are not a single turnkey pipeline.

The four-model t-SNE figure uses 12 final logits on matched test sites, not a common intermediate embedding. No models were retrained for that visualization.

Dataset counts and split definitions are in `data/README.md`. Checkpoints and full datasets are Release assets, not Git-tracked files. The repository does not include all upstream reference genomes or raw eCLIP downloads.

The current manuscript uses an inline bibliography in `manuscript.tex`, so a separate `reference.bib` is not required to compile it. Compile from `manuscript/` with pdfLaTeX twice. Author metadata and the documented class-by-length-prior wording omission still require editorial completion.

Third-party methods, template and data retain their original attribution and applicable terms. This repository does not grant a new license for third-party materials.
