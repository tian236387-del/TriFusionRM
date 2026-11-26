"""Extract matched-site benchmark features from frozen paper seed42 checkpoints."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path('/home/nefu1020220084/summer/StructDualRM/BIB_strengthening_20261028')
CODE = ROOT / 'code/train_rmbase11_official_reproduction_benchmark.py'
MODELS = {
    'MultiRM': ('baseline_multirm', 'multirm_official'),
    'TransRNAm': ('baseline_transrnam', 'transrnam_official'),
    'EvoRMD': ('baseline_evormd_safe', 'evormd_rnafm_official_core'),
}


def module_from_file(path):
    spec = importlib.util.spec_from_file_location('benchmark_reproduction', path)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_rows(master, sample):
    with sample.open(encoding='utf-8', newline='') as handle:
        sample_rows = list(csv.DictReader(handle, delimiter='\t'))
    ids = {row['site_id'] for row in sample_rows}
    found = {}
    with master.open(encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            if row['site_id'] in ids:
                found[row['site_id']] = row
    if set(found) != ids:
        raise RuntimeError(f'Missing {len(ids - set(found))} selected test sites')
    rows = [found[row['site_id']] for row in sample_rows]
    for selected, row in zip(sample_rows, rows):
        if row['mod_type'] != selected['modification'] or row['split_cdhit80_cluster'] != 'test':
            raise RuntimeError(f'Manifest disagreement at {selected["site_id"]}')
    return sample_rows, rows


def historical_probs(path, site_ids, class_names):
    wanted = set(site_ids)
    found = {}
    with path.open(encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            if row['site_id'] in wanted:
                found[row['site_id']] = [float(row[f'prob_{name}']) for name in class_names]
    if set(found) != wanted:
        raise RuntimeError(f'Historical prediction mismatch: {path}: {len(wanted - set(found))} missing')
    return np.asarray([found[site_id] for site_id in site_ids], dtype=np.float32)


def extract_one(name, kind, rows, sample_rows, module, device):
    model_dir = ROOT / 'models' / kind / 'seed42'
    payload = torch.load(model_dir / 'model.pt', map_location='cpu', weights_only=False)
    class_names = payload['class_names']
    module.configure_classes(class_names)
    args = argparse.Namespace(**payload['training_args'])
    state = payload['model_state_dict']
    if name == 'MultiRM':
        matrix = state['embedding.weight'].numpy()
        layer = 'class-conditioned Bahdanau attention contexts averaged across classes'
    elif name == 'TransRNAm':
        matrix = state['transformer.embedding.weight'].numpy()
        layer = 'Transformer and CNN concatenation before class heads'
    else:
        matrix = None
        layer = 'RNA-FM and safe-context attention pool before classifier'
    context_spec = module.ContextSpec(**payload['context_spec']) if payload['context_spec'] else None
    model, objective, _ = module.build_model(args, matrix, context_spec)
    module.load_checkpoint_model_state(model, state)
    model.to(device).eval()
    dataset = module.OfficialBenchmarkDataset(
        rows, payload['model'], args.input_length,
        sequence_column=getattr(args, 'sequence_column', 'sequence_41nt'),
        trimer_vocab=payload['trimer_vocab'],
        context_spec=context_spec,
        context_mode=getattr(args, 'context_mode', 'full'),
    )
    batch_size = 32 if name == 'EvoRMD' else 128
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=module.collate_batch)
    representations, logits = [], []
    captured = []
    if name == 'MultiRM':
        hook = model.attention.register_forward_hook(
            lambda _, __, output: captured.append(output[0].mean(dim=1).detach().cpu().numpy()))
    else:
        classifier = model.heads[0] if name == 'TransRNAm' else model.classifier
        hook = classifier.register_forward_pre_hook(
            lambda _, inputs: captured.append(inputs[0].detach().cpu().numpy()))
    try:
        with torch.inference_mode():
            for batch in loader:
                x = batch['x'].to(device)
                context = batch.get('context')
                if context is not None:
                    context = context.to(device)
                out = model(x, context)['logits']
                logits.append(out.detach().cpu().numpy())
                representations.append(captured.pop(0))
    finally:
        hook.remove()
    features = np.concatenate(representations).astype(np.float32)
    scores = np.concatenate(logits).astype(np.float32)
    probabilities = 1 / (1 + np.exp(-scores)) if objective == 'bce' else torch.softmax(torch.from_numpy(scores), dim=-1).numpy()
    site_ids = [row['site_id'] for row in sample_rows]
    prior = historical_probs(model_dir / 'test_predictions_normalized.tsv', site_ids, class_names)
    diff = np.abs(prior - probabilities)
    top1 = float(np.mean(prior.argmax(axis=1) == probabilities.argmax(axis=1)))
    # The normalized prediction archive may differ slightly from float32 inference.
    if diff.max() > 0.003 or top1 < 0.999:
        raise RuntimeError(f'{name} inference does not match archived predictions: max={diff.max()}, top1={top1}')
    return features, scores, {
        'checkpoint': str(model_dir / 'model.pt'),
        'checkpoint_sha256': hashlib.sha256((model_dir / 'model.pt').read_bytes()).hexdigest(),
        'feature_layer': layer,
        'feature_dimensions': int(features.shape[1]),
        'logit_dimensions': int(scores.shape[1]),
        'objective': objective,
        'max_prediction_difference': float(diff.max()),
        'top1_agreement': top1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    module = module_from_file(CODE)
    sample_rows, rows = load_rows(ROOT / 'data/frozen_v1/master.tsv', args.sample)
    output = {}
    audit = {'n_sites': len(rows), 'models': {}}
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    for name, (kind, expected) in MODELS.items():
        print(f'Extracting {name} on {device}', flush=True)
        features, logits, detail = extract_one(name, kind, rows, sample_rows, module, device)
        output[f'{name}_features'] = features
        output[f'{name}_logits'] = logits
        audit['models'][name] = detail
        print(f'{name}: features={features.shape}, logits={logits.shape}, agreement={detail["top1_agreement"]}', flush=True)
        torch.cuda.empty_cache()
    output['site_id'] = np.asarray([row['site_id'] for row in sample_rows])
    output['label'] = np.asarray([row['modification'] for row in sample_rows])
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / 'benchmark_features.npz', **output)
    (args.output / 'extraction_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
