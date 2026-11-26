#!/usr/bin/env python3
"""Paired cluster bootstrap for external A-centered modification classification."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

CLASSES = ['m1A', 'm6A', 'I']
FAMILY5 = ['Am', 'm1A', 'm6A', 'm6Am', 'I']
PRED_COL = {'I': 'AtoI'}


def read_tsv(path):
    with open(path, encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def bh(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 1.0
    for rank in range(len(p) - 1, -1, -1):
        idx = order[rank]
        running = min(running, p[idx] * len(p) / (rank + 1))
        adjusted[idx] = running
    return adjusted


def probs_for(rows, classes, prefix='p_'):
    columns = [prefix + PRED_COL.get(c, c) for c in classes]
    p = np.asarray([[float(row[c]) for c in columns] for row in rows], dtype=np.float64)
    return p / np.maximum(p.sum(axis=1, keepdims=True), 1e-300)


def scores(y, p, weights=None):
    out = {}
    for i, name in enumerate(CLASSES):
        out[name] = average_precision_score(y == i, p[:, i], sample_weight=weights)
    out['MacroAUPR'] = float(np.mean([out[c] for c in CLASSES]))
    return out


def p_empirical(values):
    values = np.asarray(values)
    return min(1.0, (2 * (1 + min(np.sum(values <= 0), np.sum(values >= 0)))) / (len(values) + 1))


def run_pair(full_rows, seq_rows, data_rows, outdir, replicates, seed):
    assert len(full_rows) == len(seq_rows) == len(data_rows)
    ids = [r['site_id'] for r in data_rows]
    assert ids == [r['site_id'] for r in full_rows] == [r['site_id'] for r in seq_rows]
    keep = [i for i, r in enumerate(data_rows) if r['mod_type'] in {'m6A', 'm1A', 'I', 'AtoI'}]
    data = [data_rows[i] for i in keep]
    full = [full_rows[i] for i in keep]
    seq = [seq_rows[i] for i in keep]
    names = ['m1A', 'm6A', 'I']
    y = np.asarray([names.index('I' if r['mod_type'] == 'AtoI' else r['mod_type']) for r in data], dtype=np.int8)
    cluster = np.asarray([r['external_cluster_id'] for r in data], dtype=object)
    cluster_ids, cluster_inverse = np.unique(cluster, return_inverse=True)

    scenarios = {
        'closed3': (names, [0, 1, 2]),
        'open5': (FAMILY5, [1, 2, 4]),
    }
    rng = np.random.default_rng(seed)
    summary, point_rows, audit, all_boots = [], [], {}, {}
    for scenario, (class_set, target_cols) in scenarios.items():
        pf = probs_for(full, class_set)
        ps = probs_for(seq, class_set)
        pf = pf[:, target_cols]
        ps = ps[:, target_cols]
        # open5 leaves probability assigned to absent Am/m6Am as competing mass.
        point_f = scores(y, pf)
        point_s = scores(y, ps)
        metrics = ['MacroAUPR'] + names
        delta_point = {m: point_f[m] - point_s[m] for m in metrics}
        boots = {m: [] for m in metrics}
        n_clusters = len(cluster_ids)
        accepted = 0
        while accepted < replicates:
            sampled = rng.integers(0, n_clusters, size=n_clusters)
            counts = np.bincount(sampled, minlength=n_clusters)
            weights = counts[cluster_inverse]
            if any(np.sum(weights[y == c]) == 0 for c in range(3)):
                continue
            sf = scores(y, pf, weights)
            ss = scores(y, ps, weights)
            for metric in metrics:
                boots[metric].append(sf[metric] - ss[metric])
            accepted += 1
        pvals = [p_empirical(boots[m]) for m in metrics]
        fdrs = bh(pvals)
        all_boots[scenario] = boots
        for j, metric in enumerate(metrics):
            vals = np.asarray(boots[metric])
            summary.append({
                'scenario': scenario, 'comparison': 'full_minus_sequence_only', 'class': metric,
                'full_AUPR': point_f[metric], 'sequence_only_AUPR': point_s[metric],
                'delta_AUPR': delta_point[metric], 'CI_low': np.quantile(vals, .025),
                'CI_high': np.quantile(vals, .975), 'p_empirical_two_sided': pvals[j],
                'BH_FDR_within_scenario': fdrs[j], 'bootstrap_replicates': replicates,
                'accepted_draws': accepted, 'clusters': n_clusters,
                'sites': len(data), 'support_positive': int(np.sum(y == names.index(metric))) if metric in names else int(len(y)),
            })
            point_rows.append({
                'scenario': scenario, 'class': metric,
                'full_AUPR': point_f[metric], 'sequence_only_AUPR': point_s[metric],
                'delta_AUPR': delta_point[metric], 'support_positive': int(np.sum(y == names.index(metric))) if metric in names else int(len(y)),
            })
        audit[scenario] = {'classes_normalized_over': class_set, 'evaluated_classes': names,
                           'n_sites': len(data), 'n_clusters': n_clusters,
                           'supports': {n: int(np.sum(y == i)) for i, n in enumerate(names)},
                           'bootstrap_replicates': replicates, 'seed': seed,
                           'resampling_unit': 'external_cluster_id',
                           'condition': 'external observed A-centered labels only; Am/m6Am have no positives'}
    outdir.mkdir(parents=True, exist_ok=True)
    for filename, rows in [('external_A_centered_paired_bootstrap.tsv', summary),
                           ('external_A_centered_point_metrics.tsv', point_rows)]:
        with (outdir / filename).open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t')
            writer.writeheader()
            writer.writerows(rows)
    (outdir / 'external_A_centered_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    with (outdir / 'external_A_centered_bootstrap_differences.tsv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['scenario', 'metric', 'replicate', 'delta_AUPR'])
        for scenario in scenarios:
            for metric in metrics:
                vals = all_boots[scenario][metric]
                writer.writerows((scenario, metric, i + 1, value) for i, value in enumerate(vals))
    print(json.dumps({'audit': audit, 'point_metrics': point_rows, 'bootstrap_summary': summary}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full-predictions', type=Path, required=True)
    parser.add_argument('--sequence-predictions', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--replicates', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20261104)
    args = parser.parse_args()
    run_pair(read_tsv(args.full_predictions), read_tsv(args.sequence_predictions),
             read_tsv(args.data), args.outdir, args.replicates, args.seed)


if __name__ == '__main__':
    main()
