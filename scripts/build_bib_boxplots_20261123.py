from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json
import re
import shutil
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(r'F:\pythonprogram\result\附件\修改\latex修改')
DATA = ROOT / 'boxplot_sources_20261123'
FIG = ROOT / 'figure'
WORK = ROOT / 'TriFusionRM_BIB_boxplots_20261123'
OLD = ROOT / 'TriFusionRM_BIB_updated_figures_20261123'
CLASSES = ['Am','Cm','Gm','Um','m1A','m5C','m5U','m6A','m6Am','m7G','Y','AtoI']
DISPLAY = {'Y': r'$\Psi$', 'Psi': r'$\Psi$', 'AtoI':'I'}
COLORS = ['#c89d79', '#76aaa5', '#9baabd']


def box(ax, arrays, labels, horizontal=False, colors=None):
    b = ax.boxplot(arrays, vert=not horizontal, patch_artist=True, widths=.52,
                   labels=labels, whis=1.5, showfliers=True,
                   flierprops=dict(marker='.', markersize=1.5, alpha=.3, markeredgecolor='#888888'),
                   medianprops=dict(color='#303030', linewidth=1.1),
                   boxprops=dict(edgecolor='#606060', linewidth=.8),
                   whiskerprops=dict(color='#606060', linewidth=.8), capprops=dict(color='#606060', linewidth=.8))
    for i, patch in enumerate(b['boxes']):
        patch.set_facecolor((colors or COLORS)[i % len(colors or COLORS)])
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='x' if horizontal else 'y', alpha=.16)
    ax.set_axisbelow(True)
    if horizontal:
        ax.invert_yaxis()


def save(fig, name):
    for ext in ['pdf','png','svg']:
        fig.savefig(FIG / f'{name}.{ext}', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    plt.rcParams.update({'font.size':9, 'font.family':'DejaVu Sans', 'pdf.fonttype':42, 'svg.fonttype':'none'})
    base = DATA / 'final_modality_bootstrap_2000'
    meta = json.loads((base/'COMPLETED.json').read_text())
    names = list(meta['models'])
    dist = np.load(base/'bootstrap_distributions.npz')
    ids = [i for i,c in enumerate(CLASSES) if c != 'm6Am']
    ap = {name: np.stack([dist[f'standard__{j}__{names.index(name)}'] for j in ids]) for name in names}
    macro = {name: values.mean(axis=0) for name,values in ap.items()}
    original = pd.read_csv(base/'bootstrap_macro.tsv', sep='\t')
    for comp,a,b in meta['comparisons']:
        values = (ap[a]-ap[b]).mean(axis=0)
        row = original[(original.comparison==comp)&(original.scope=='standard')&(original.class_subset=='all_evaluable')].iloc[0]
        np.testing.assert_allclose(np.quantile(values,[.025,.975]),[row.ci95_low,row.ci95_high],atol=2e-7)
    fig = plt.figure(figsize=(11.5,5.5))
    gs=fig.add_gridspec(2,2,width_ratios=[1,1.2],wspace=.45,hspace=.65)
    a=fig.add_subplot(gs[0,0]); b=fig.add_subplot(gs[1,0]); c=fig.add_subplot(gs[:,1])
    models=['final_no_pair','no_pair_shuffled_rbp','no_pair_shuffled_bins']
    box(a,[macro[n] for n in models],['TriFusionRM','Shuffled RBP','Shuffled intervals'])
    a.set_title('A  Bootstrap Macro AUPR',loc='left',weight='bold'); a.set_ylabel('Macro AUPR')
    contrast=[(ap['final_no_pair']-ap[n]).mean(axis=0) for n in models[1:]]
    box(b,contrast,['Real minus shuffled RBP','Real minus shuffled intervals'])
    b.axhline(0,color='#888888',ls='--',lw=.7); b.set_ylabel(r'$\Delta$ Macro AUPR')
    b.tick_params(axis='x',labelsize=8)
    b.set_title('B  Paired bootstrap differences',loc='left',weight='bold')
    per=ap['final_no_pair']-ap['no_pair_shuffled_rbp']
    box(c,list(per),[DISPLAY.get(CLASSES[j],CLASSES[j]) for j in ids],True,[COLORS[1]])
    c.axvline(0,color='#888888',ls='--',lw=.7); c.set_xlabel(r'$\Delta$ AUPR: real minus shuffled RBP')
    c.set_title('C  Per-class bootstrap differences',loc='left',weight='bold')
    fig.subplots_adjust(left=.075,right=.98,bottom=.13,top=.92)
    save(fig,'Figure4_boxplot')
    motifbase=DATA/'motif_distance_matched_seed42'
    sites=pd.read_csv(motifbase/'motif_attribution_per_site.tsv',sep='\t')
    group=sites.groupby(['modification','cluster_id'],sort=False).delta_logit_contrast_mean.mean()
    motif=sites.groupby(['modification','motif','cluster_id'],sort=False).delta_logit_contrast_mean.mean()
    ms=pd.read_csv(motifbase/'motif_attribution_summary.tsv',sep='\t')
    gs=pd.read_csv(motifbase/'modification_attribution_summary.tsv',sep='\t')
    for row in gs.itertuples():
        assert len(group.loc[row.modification]) == row.n_independent_clusters
        np.testing.assert_allclose(group.loc[row.modification].mean(),row.mean_delta_logit_contrast,atol=1e-6)
    for row in ms.itertuples():
        assert len(motif.loc[(row.modification,row.motif)]) == row.n_clusters
        np.testing.assert_allclose(motif.loc[(row.modification,row.motif)].mean(),row.mean_delta_logit_contrast,atol=1e-6)
    order=['Am','Um','m1A','m5C','m6A','Psi','I']
    assert set(order)==set(sites.modification)
    pairs=[(cl,m) for cl in order for m in sorted(ms.loc[ms.modification==cl,'motif'])]
    fig,axes=plt.subplots(1,2,figsize=(11.5,7),gridspec_kw={'width_ratios':[1,1.35]})
    vals=[group.loc[cl].values for cl in order]
    box(axes[0],vals,[f'{DISPLAY.get(cl,cl)} (n={len(v):,})' for cl,v in zip(order,vals)],True,[COLORS[1]])
    vals2=[motif.loc[p].values for p in pairs]
    box(axes[1],vals2,[f'{DISPLAY.get(cl,cl)} {m} (n={len(v):,})' for (cl,m),v in zip(pairs,vals2)],True,[COLORS[0]])
    for ax, data in zip(axes,[vals,vals2]):
        ax.scatter([v.mean() for v in data],np.arange(1,len(data)+1),marker='D',s=13,c='#234b59',zorder=4)
        ax.axvline(0,color='#888888',ls='--',lw=.7)
        ax.set_xlabel('Target-class logit decrease: motif minus random')
    axes[0].set_title('A  Cluster-level effects by modification',loc='left',weight='bold')
    axes[1].set_title('B  Cluster-level effects by motif',loc='left',weight='bold')
    fig.tight_layout(w_pad=2.5)
    save(fig,'Figure5_boxplot')
    group.rename('contrast').to_csv(DATA/'modification_cluster_values.tsv',sep='\t')
    motif.rename('contrast').to_csv(DATA/'motif_cluster_values.tsv',sep='\t')
    WORK.mkdir(exist_ok=True)
    for p in OLD.rglob('*'):
        if p.is_file() and p.relative_to(OLD).parts[0]!='figure' and p.suffix not in ['.aux','.out','.log','.pdf']:
            out=WORK/p.relative_to(OLD);out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,out)
    shutil.copytree(FIG,WORK/'figure',dirs_exist_ok=True)
    shutil.copytree(DATA,WORK/'boxplot_source',dirs_exist_ok=True)
    shutil.copy2(__file__,WORK/'boxplot_source'/Path(__file__).name)
    tex=(WORK/'manuscript.tex').read_text(encoding='utf-8')
    tex=tex.replace(r'width=0.82\textwidth]{figure/figure3_TriFusionRM.png}',r'width=0.50\textwidth]{figure/figure3_TriFusionRM.png}')
    captions={
      4:r'RBP controls in no-pair TriFusionRM. A shows bootstrap Macro AUPR for the real-RBP model and trained shuffled-RBP and shuffled-distance-interval controls. B shows paired Macro AUPR differences, and C shows per-class real-minus-shuffled-RBP differences. Each box summarizes 2,000 matched cluster-bootstrap draws from the frozen test set, not independent training runs. Boxes span the 25th to 75th percentiles, lines mark medians, whiskers extend to the most extreme observations within 1.5 interquartile ranges, and dots show observations beyond the whiskers. Only draws retaining positive sites for all 11 evaluable classes were accepted. These boxplots are not 95\% confidence intervals; paired intervals and adjusted P values remain in Supplementary Tables S16 and S18. m6Am has no test positives.',
      5:r'Cluster-level sensitivity to motif disruption in no-pair TriFusionRM. A summarizes contrasts after averaging motif-site observations within each modification and cluster. B summarizes the fixed set of 20 test-replicated motifs after averaging within each motif and cluster. The contrast is the target-class logit decrease after motif disruption minus that after matched random disruption, with 10 perturbations averaged per condition and site. Each observation is a cluster-level contrast. Boxes span the 25th to 75th percentiles, center lines mark medians, whiskers extend to the most extreme observations within 1.5 interquartile ranges, and dots show observations beyond the whiskers. Diamonds mark means, and n denotes contributing clusters. These distributions show cluster-to-cluster variation rather than confidence intervals of the mean. The center base was protected, edits were synchronized across windows, and structure and RBP inputs were fixed. Confidence intervals and adjusted P values remain in Supplementary Tables S27 and S28. Am is supported by seven clusters overall.'}
    for n in [4,5]:
        pattern=r'\\begin\{figure\*\}.*?\\end\{figure\*\}'
        def substitute(match):
            if f'figure{n}' not in match.group(0):return match.group(0)
            return '\\begin{figure*}[!ht]\n\\centering\n'+f'\\includegraphics[width=\\textwidth]{{figure/Figure{n}_boxplot.pdf}}\n\\caption{{{captions[n]}}}\n\\end{{figure*}}'
        tex=re.sub(pattern,substitute,tex,flags=re.S)
    (WORK/'manuscript.tex').write_text(tex,encoding='utf-8')
    for _ in range(3):
        result=subprocess.run(['pdflatex','-halt-on-error','-interaction=nonstopmode','manuscript.tex'],cwd=WORK,capture_output=True)
        if result.returncode:raise RuntimeError(result.stdout.decode(errors='replace')[-3000:])
    out=ROOT/'TriFusionRM_BIB_boxplots_20261123.zip'
    with ZipFile(out,'w',ZIP_DEFLATED) as z:
        for p in WORK.rglob('*'):
            if p.is_file() and p.suffix not in ['.aux','.log','.out','.gz']:
                z.write(p,p.relative_to(WORK))
    with ZipFile(out) as z:assert z.testzip() is None
    print(out)
    print('Verified bootstrap quantiles and all motif/cluster means against archived summaries.')


if __name__=='__main__':main()
