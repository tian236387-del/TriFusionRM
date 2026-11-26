"""Matched-site t-SNE comparison for TriFusionRM and three paper benchmarks."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


ROOT = Path(r'F:\pythonprogram\result\附件\修改\latex修改')
SOURCE = ROOT / 'tsne_four_models_20261125'
FULL = Path(r'F:\pythonprogram\result\附件\文件\tSNE特征可视化_20261117\extracted_features.npz')
CLASSES = ['Am', 'Cm', 'Gm', 'Um', 'm1A', 'm5C', 'm5U', 'm6A', 'm7G', 'Y', 'AtoI']
COLORS = ['#456f80', '#a7756d', '#84915e', '#8b78a3', '#409686', '#bd8d54',
          '#9aa4b2', '#b46e85', '#a3a259', '#5b9ca9', '#786b63']
COLOR = dict(zip(CLASSES, COLORS))
MODELS = ['TriFusionRM', 'MultiRM', 'TransRNAm', 'EvoRMD']


def project(x, seed, perplexity):
    if x.shape[1] > 50:
        x = PCA(n_components=50, random_state=seed).fit_transform(x)
    xy = TSNE(n_components=2, perplexity=perplexity, init='pca',
              learning_rate='auto', random_state=seed, metric='euclidean').fit_transform(x)
    return xy.astype(np.float32)


def render(data, labels, output, kind, perps=(30,)):
    coordinates = []
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.6,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    for perplexity in perps:
        fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.75))
        for row, subset in enumerate([np.ones(len(labels), dtype=bool),
                                       np.isin(labels, ['m6A', 'm1A', 'AtoI'])]):
            for col, model in enumerate(MODELS):
                ax = axes[row, col]
                xy = project(data[model][subset], 42, perplexity)
                these_labels = labels[subset]
                for class_name in CLASSES:
                    mask = these_labels == class_name
                    if mask.any():
                        ax.scatter(xy[mask, 0], xy[mask, 1], s=6.2,
                                   c=COLOR[class_name], alpha=0.58, edgecolors='none', rasterized=True)
                ax.set_title(f'{"ABCDEFGH"[row*4+col]}  {model}', loc='left', fontsize=10, pad=8)
                ax.set_xlabel('t-SNE 1', fontsize=8)
                ax.set_ylabel('t-SNE 2', fontsize=8)
                ax.spines[['top', 'right']].set_visible(False)
                ax.tick_params(labelsize=7.5, length=2)
                for idx, point in zip(np.flatnonzero(subset), xy):
                    coordinates.append((kind, perplexity, 'all' if row == 0 else 'A_centered',
                                        model, int(idx), float(point[0]), float(point[1])))
        fig.subplots_adjust(left=0.055, right=0.99, top=0.91, bottom=0.13,
                            hspace=0.48, wspace=0.31)
        fig.text(0.055, 0.966, 'All evaluable modification types  |  n = 1,912',
                 fontsize=10.5, weight='bold')
        fig.text(0.055, 0.515, 'A-centered modifications  |  m6A, m1A and I  |  n = 874',
                 fontsize=10.5, weight='bold')
        handles = [Line2D([], [], marker='o', linestyle='', color=COLOR[c],
                          markersize=4.4, label={'Y': r'$\Psi$', 'AtoI': 'I'}.get(c, c))
                   for c in CLASSES]
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.018),
                   ncol=11, frameon=False, handletextpad=0.3, columnspacing=1.0)
        suffix = '' if perplexity == 30 else f'_p{perplexity}'
        for ext in ['pdf', 'svg', 'png']:
            fig.savefig(output / f'Figure6_{kind}_four_models{suffix}.{ext}', dpi=300, facecolor='white')
        plt.close(fig)
        print(f'{kind} p={perplexity}: complete', flush=True)
    return coordinates


def main():
    SOURCE.mkdir(exist_ok=True)
    benchmark = np.load(SOURCE / 'benchmark_features.npz')
    full = np.load(FULL)
    assert np.array_equal(benchmark['site_id'], full['site_id'])
    assert np.array_equal(benchmark['label'], full['label'])
    labels = benchmark['label']
    assert len(labels) == 1912 and int(np.isin(labels, ['m6A', 'm1A', 'AtoI']).sum()) == 874
    logits = {'TriFusionRM': full['C_final_logits_12']}
    features = {'TriFusionRM': full['B_sequence_structure_128']}
    for name in MODELS[1:]:
        logits[name] = benchmark[f'{name}_logits']
        features[name] = benchmark[f'{name}_features']
    for name in MODELS:
        assert logits[name].shape == (1912, 12)
        assert np.isfinite(logits[name]).all() and np.isfinite(features[name]).all()
    output = ROOT / 'TriFusionRM_BIB_four_model_tsne_20261125' / 'figure'
    output.mkdir(parents=True, exist_ok=True)
    coords = render(logits, labels, output, 'logits', perps=(20, 30, 50))
    # The manuscript comparison uses the same 12 logit coordinates for every model.
    # Penultimate representations have different dimensions and are retained as
    # extraction artifacts rather than mixed into this cross-model figure.
    with (SOURCE / 'tsne_coordinates.tsv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow(['representation', 'perplexity', 'subset', 'model', 'sample_index', 'site_id', 'label', 'tsne_1', 'tsne_2'])
        writer.writerows((kind, p, subset, model, i, benchmark['site_id'][i], labels[i], x, y)
                         for kind, p, subset, model, i, x, y in coords)
    audit = json.loads((SOURCE / 'extraction_audit.json').read_text(encoding='utf-8'))
    audit['class_counts'] = dict(Counter(labels.tolist()))
    audit['same_base_count'] = 874
    audit['tsne'] = {'seed': 42, 'perplexities': [20, 30, 50], 'init': 'pca',
                     'learning_rate': 'auto', 'metric': 'euclidean',
                     'PCA_max_dimensions': 50, 'same_sites_for_every_model': True,
                     'embeddings_fitted_independently': True,
                     'main_figure': 'logits, not penultimate features',
                     'TriFusionRM_penultimate_note': 'sequence+structure main_context before RBP logit residual'}
    (SOURCE / 'experiment_summary.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
