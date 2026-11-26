# Processed datasets and checkpoints

Download assets from https://github.com/tian236387-del/TriFusionRM/releases/tag/v1.0.0 and verify against `SHA256SUMS.txt`.

## Dataset 1: RMBase-derived frozen_v1

`StructDualRM_Dataset1.tar.gz` contains the processed frozen master table, structure inputs, positional RBP inputs and provenance. The common-window dataset contains 1,082,044 sites: 865,582 training, 108,251 validation and 108,211 test sites. Exact-unique 41-nt sequences were clustered with CD-HIT-est at 80% identity; cluster-level assignments were inherited by the 101-nt and 201-nt windows. Longer windows were not independently reclustered.

Master-table fields include `site_id`, `mod_type`, `class_index`, `cdhit80_cluster_id`, `split_cdhit80_cluster`, and `sequence_41nt`, `sequence_101nt`, `sequence_201nt`. See the packaged provenance for upstream RMBase and GRCh38 processing. Twelve output labels are Am, Cm, Gm, Um, m1A, m5C, m5U, m6A, m6Am, m7G, Y (pseudouridine) and AtoI (I). No m6Am positives occur in the internal test set.

## Dataset 2: NaRMBench-derived external set

`StructDualRM_Dataset2.tar.gz` contains 96,352 external sites across six classes, the processed sequence table, structure inputs and positional RBP inputs. It is used for external evaluation only, not training or validation. The external MMseqs2 filtering is distinct from the internal CD-HIT-est split. Archive paths retain the original project layout.

## Final model

`StructDualRM_final_no_pair_20261106.zip` contains the frozen seed42 checkpoint, original inference example, code and preprocessing information. Old file names are retained deliberately for provenance.

These are processed research inputs, not complete mirrors of upstream databases, reference genomes or eCLIP peak collections. Follow original source attribution and redistribution terms.
