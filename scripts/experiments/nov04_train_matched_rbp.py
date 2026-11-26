#!/usr/bin/env python3
"""Train a matched-stratum RBP permutation control from the frozen-v1 config."""
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path('/home/nefu1020220084/summer/StructDualRM/BIB_strengthening_20261028')
sys.path.insert(0, str(ROOT / 'code'))
import torch
import numpy as np
import train_motif_struct_fusion_v4 as trainer

AUDITS = []
AUDIT_PATH = None


def matched_permutation(dataset, seed):
    groups = defaultdict(list)
    split_name = 'unknown'
    for i, row in enumerate(dataset.rows):
        split_name = row.get('split_cdhit80_cluster', split_name)
        gene = row.get('gene_id', '').strip()
        biotype = row.get('gene_type', '').strip()
        region = row.get('region_type', '').strip()
        missing = {'', 'na', 'nan', 'none', 'null', '.', '-', 'unknown'}
        if gene.lower() in missing | {'intergenic'} or biotype.lower() in missing | {'intergenic'} or region.lower() in missing:
            key = ('singleton', str(i))
        else:
            key = (gene, biotype, region)
        groups[key].append(i)
    digest = hashlib.sha256(str(split_name).encode('utf-8')).digest()
    rng = np.random.default_rng(seed + int.from_bytes(digest[:4], 'little'))
    donor = np.arange(len(dataset.rows))
    eligible_groups = eligible_rows = 0
    for indices in groups.values():
        if len(indices) < 2:
            continue
        eligible_groups += 1
        eligible_rows += len(indices)
        ordered = rng.permutation(indices)
        donor[ordered] = np.roll(ordered, int(rng.integers(1, len(indices))))
    before = dataset.rbp_pos_tensor
    dataset.rbp_pos_tensor = before[torch.as_tensor(donor, dtype=torch.long)]
    changed = int((before != dataset.rbp_pos_tensor).flatten(start_dim=1).any(dim=1).sum().item())
    audit = {
        'split': split_name,
        'sites': len(dataset.rows),
        'strata': len(groups),
        'exchangeable_strata': eligible_groups,
        'exchangeable_sites': eligible_rows,
        'exchangeable_site_fraction': eligible_rows / max(1, len(dataset.rows)),
        'sites_with_changed_RBP_tensor': changed,
        'matched_on': ['gene_id', 'gene_type', 'region_type'],
        'permutation_seed': int(seed + int.from_bytes(digest[:4], 'little')),
        'missing_metadata_rows_left_unshuffled': sum(1 for x in groups if x[0] == 'singleton'),
    }
    AUDITS.append(audit)
    if AUDIT_PATH is not None:
        AUDIT_PATH.write_text(json.dumps({'splits': AUDITS}, indent=2))
    print('MATCHED_SHUFFLE_AUDIT', json.dumps(audit), flush=True)
    return audit


class MatchedPositionalDataset(trainer.RMBaseFusionPositionalDataset):
    def __init__(self, *args, **kwargs):
        seed = int(kwargs.get('seed', 42))
        super().__init__(*args, **kwargs)
        matched_permutation(self, seed)


class MatchedMultiLengthPositionalDataset(trainer.RMBaseFusionMultiLengthPositionalDataset):
    def __init__(self, *args, **kwargs):
        seed = int(kwargs.get('seed', 42))
        super().__init__(*args, **kwargs)
        matched_permutation(self, seed)


trainer.RMBaseFusionPositionalDataset = MatchedPositionalDataset
trainer.RMBaseFusionMultiLengthPositionalDataset = MatchedMultiLengthPositionalDataset


def main(job_path):
    global AUDIT_PATH
    job = json.loads(Path(job_path).read_text())
    config_path = ROOT / job['template']
    config = json.loads(config_path.read_text())
    args = trainer.build_parser().parse_args([])
    unknown = []
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)
        else:
            unknown.append(key)
    for key, value in job.get('overrides', {}).items():
        if not hasattr(args, key):
            raise ValueError('Unknown training override: ' + key)
        setattr(args, key, value)
    args.dataset = str(ROOT / 'data/frozen_v1/master.tsv')
    args.rnafold = str(ROOT / 'data/frozen_v1/rnafold.tsv')
    args.rbp_positional_features = str(ROOT / 'data/frozen_v1/rbp_positional.tsv')
    args.rbp_features = ''
    args.out_dir = str(ROOT / job['out_dir'])
    args.resume = False
    args.shuffle_rbp = False
    args.shuffle_rbp_bins = False
    torch.set_num_threads(4)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH = out / 'matched_rbp_shuffle_audit.json'
    (out / 'resolved_requested_config.json').write_text(json.dumps(vars(args), indent=2))
    (out / 'launch_environment.json').write_text(json.dumps({
        'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'cuda_available': torch.cuda.is_available(), 'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'unknown_legacy_keys': unknown, 'time': time.time(),
        'RBP_control': 'within split, permute whole RBP positional tensors within exact gene_id x gene_type x region_type strata',
    }, indent=2))
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this run')
    trainer.run(args)
    (out / 'matched_rbp_shuffle_audit.json').write_text(json.dumps({
        'job': job, 'splits': AUDITS,
        'interpretation': 'Tests site-specific RBP correspondence conditional on matched gene, gene_type and region_type. Singleton/missing-metadata strata remain unshuffled and are quantified above.',
    }, indent=2))
    (out / 'TRAINING_COMPLETED.json').write_text(json.dumps({'job': job, 'time': time.time()}))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: nov04_train_matched_rbp.py JOB.json')
    main(sys.argv[1])
