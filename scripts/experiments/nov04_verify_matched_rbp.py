"""Independently check matched-RBP donors, predictions, and bootstrap tables."""
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score
from sklearn.metrics import matthews_corrcoef, precision_recall_fscore_support, roc_auc_score
from scipy.stats import false_discovery_control

from nov04_followup_prepare import CLASSES, O, R, read, write

FAMILIES = [list(x) for x in [('Am','m1A','m6A','m6Am','AtoI'), ('Cm','m5C'), ('Gm','m7G'), ('Um','m5U','Y')]]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_donors():
    metadata = defaultdict(dict)
    for row in read(R/'data/frozen_v1/master.tsv'):
        metadata[row['split_cdhit80_cluster']][row['site_id']] = (
            row['gene_id'].strip(), row['gene_type'].strip(), row['region_type'].strip(),
            row['mod_type'], row['cdhit80_cluster_id'])
    result = []
    audit_path = O/'matched_rbp/matched_rbp_shuffle_audit.json'
    if not audit_path.exists():
        audit_path = R/'models/matched_rbp_shuffle_gene_validated/seed42/matched_rbp_shuffle_audit.json'
    audit = json.loads(audit_path.read_text())
    missing = {'','na','nan','none','null','.','-','unknown'}
    for split, sites in metadata.items():
        rows = list(read(O/'matched_rbp'/(split+'_donor_mapping.tsv')))
        assert len(rows) == len(sites)
        assert {r['site_id'] for r in rows} == set(sites)
        assert {r['donor_site_id'] for r in rows} == set(sites)
        eligible = invalid = different_label_donors = 0
        group_sizes = Counter(value[:3] for value in sites.values())
        for row in rows:
            original, donor = sites[row['site_id']], sites[row['donor_site_id']]
            bad = (original[0].lower() in missing|{'intergenic'} or
                   original[1].lower() in missing|{'intergenic'} or original[2].lower() in missing)
            expected = not bad and group_sizes[original[:3]] > 1
            assert (row['exchangeable']=='True') == expected
            assert (row['site_id'] != row['donor_site_id']) == expected
            assert original[3] == row['modification']
            if expected:
                assert original[:3] == donor[:3]
                eligible += 1
                different_label_donors += int(original[3] != donor[3])
            if bad:
                invalid += 1
        recorded = next(x for x in audit['splits'] if x['split']==split)
        assert recorded['sites'] == len(rows)
        assert recorded['exchangeable_sites'] == eligible
        assert recorded['missing_metadata_rows_left_unshuffled'] == invalid
        result.append(dict(split=split, sites=len(rows), eligible_sites=eligible,
                           invalid_annotation_sites=invalid,eligible_different_label_donors=different_label_donors,
                           eligible_different_label_fraction=different_label_donors/max(1,eligible),
                           donor_map_sha256=digest(O/'matched_rbp'/(split+'_donor_mapping.tsv'))))
    return result, metadata['test']


def evaluate_subset(subset, master):
    directory = O/'matched_rbp'/subset
    stats = directory/'cluster_bootstrap_2000'
    completed = json.loads((stats/'COMPLETED.json').read_text())
    assert completed['replicates']==2000
    names = list(completed['models'])
    distributions = np.load(stats/'bootstrap_distributions.npz')
    probabilities, truth = {}, None
    ids = None
    macro_metrics = []
    for name in names:
        rows = list(read(directory/(name+'_test_predictions.tsv')))
        current_ids = [r['site_id'] for r in rows]
        assert len(set(current_ids)) == len(rows)
        assert all(master[r['site_id']][3] == r['true_mod_type'] for r in rows)
        index = {r['site_id']:r for r in rows}
        if ids is None:
            ids = current_ids
        else:
            assert set(ids) == set(current_ids)
            rows = [index[x] for x in ids]
        y = np.array([CLASSES.index(r['true_mod_type']) for r in rows])
        p = np.array([[float(r['prob_'+c]) for c in CLASSES] for r in rows], dtype=np.float32)
        assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        assert np.allclose(p.sum(1), 1, atol=2e-6)
        if truth is None:
            truth = y
        else:
            assert np.array_equal(truth, y)
        probabilities[name] = p
        active = np.unique(y)
        yp = p.argmax(1)
        precision, recall, f1, _ = precision_recall_fscore_support(y,yp,labels=active,zero_division=0)
        macro_metrics.append(dict(subset=subset,model=name,n_sites=len(y),n_clusters=len({master[x][4] for x in ids}),
                                  evaluable_classes=len(active),Macro_AUPR=np.mean([average_precision_score(y==j,p[:,j]) for j in active]),
                                  Macro_AUC=np.mean([roc_auc_score(y==j,p[:,j]) for j in active]),Macro_Precision=np.mean(precision),
                                  Macro_Recall=np.mean(recall),Macro_F1=np.mean(f1),
                                  Macro_MCC=np.mean([matthews_corrcoef(y==j,yp==j) for j in active]),Global_MCC=matthews_corrcoef(y,yp),
                                  Balanced_accuracy=balanced_accuracy_score(y,yp)))
    positive_clusters = {j:len({master[x][4] for x,t in zip(ids,truth) if t==j}) for j in np.unique(truth)}
    observed = {}
    score_vectors = {}
    for scope in ['standard','same_base']:
        for j in positive_clusters:
            family = next(f for f in FAMILIES if CLASSES[j] in f)
            jj = [CLASSES.index(c) for c in family]
            mask = np.ones(len(truth),dtype=bool) if scope=='standard' else np.isin(truth,jj)
            for ki,name in enumerate(names):
                p = probabilities[name]
                if scope=='standard':
                    score = p[:,j]
                else:
                    q = p[mask][:,jj]
                    score = q[:,jj.index(j)] / q.sum(1).clip(min=1e-7)
                observed[scope,j,ki] = average_precision_score(truth[mask]==j,score)
                score_vectors[scope,j,ki] = (mask,score)
    # Check the optimized weighted-AP engine against sklearn on real cluster draws.
    clusters,inverse = np.unique(np.array([master[x][4] for x in ids],dtype=object),return_inverse=True)
    nclusters = len(clusters)
    assert completed['n_clusters']==nclusters
    rng = np.random.default_rng(completed['random_seed'])
    counts = rng.multinomial(nclusters,np.full(nclusters,1/nclusters),size=256)
    accepted = np.ones(len(counts),dtype=bool)
    for j in positive_clusters:
        accepted &= counts[:,np.unique(inverse[truth==j])].sum(1)>0
    draws = counts[accepted][:2]
    assert len(draws)==2
    max_error = 0.0
    for (scope,j,ki),(mask,score) in score_vectors.items():
        saved = distributions[f'{scope}__{j}__{ki}']
        for b,cluster_counts in enumerate(draws):
            weight = cluster_counts[inverse][mask]
            independent = average_precision_score(truth[mask]==j,score,sample_weight=weight)
            error = abs(independent-float(saved[b]))
            assert error<2e-5,(subset,scope,CLASSES[j],names[ki],b,error)
            max_error = max(max_error,error)
    verified_rows = 0
    for filename in ['bootstrap_macro.tsv','bootstrap_per_class.tsv']:
        rows = list(read(stats/filename))
        correction = defaultdict(list)
        for i,row in enumerate(rows):
            scope = row['scope']
            ai,bi = names.index(row['model_a']),names.index(row['model_b'])
            if filename=='bootstrap_macro.tsv':
                js = [j for j in positive_clusters if row['class_subset']=='all_evaluable' or positive_clusters[j]>=10]
                assert int(row['classes']) == len(js)
            else:
                js = [CLASSES.index(row['class'])]
                assert int(row['positive_clusters']) == positive_clusters[js[0]]
            values = np.mean([distributions[f'{scope}__{j}__{ai}']-distributions[f'{scope}__{j}__{bi}'] for j in js],axis=0)
            assert len(values)==2000 and np.isfinite(values).all()
            interval = np.quantile(values,[.025,.975])
            assert np.allclose(interval,[float(row['ci95_low']),float(row['ci95_high'])],atol=1e-8)
            pvalue = min(1,2*(1+min(np.count_nonzero(values<=0),np.count_nonzero(values>=0)))/2001)
            assert abs(pvalue-float(row['p_two_sided'])) < 1e-12
            for ki,key in [(ai,'AP_a'),(bi,'AP_b')]:
                actual = np.mean([observed[scope,j,ki] for j in js])
                assert abs(actual-float(row[key]))<1e-10
            assert abs(float(row['AP_a'])-float(row['AP_b'])-float(row['delta_AUPR']))<1e-10
            correction[(scope,row.get('class_subset','per_class'))].append(i)
            verified_rows += 1
        for indices in correction.values():
            adjusted = false_discovery_control([float(rows[i]['p_two_sided']) for i in indices],method='bh')
            assert np.allclose(adjusted,[float(rows[i]['FDR']) for i in indices],atol=1e-12)
    write(directory/'all_models_macro_metrics.tsv',macro_metrics)
    return dict(subset=subset,n_sites=len(ids),n_clusters=len({master[x][4] for x in ids}),
                verified_statistical_rows=verified_rows,positive_clusters={CLASSES[j]:n for j,n in positive_clusters.items()},
                weighted_AP_spotcheck_draws=2,weighted_AP_spotcheck_max_abs_error=max_error,
                model_order=names,macro_metrics=macro_metrics)


def main():
    donors, master = validate_donors()
    subsets = [evaluate_subset(s,master) for s in ['all_test','exchangeable_test_only']]
    result = dict(status='VERIFIED',replicates=2000,donor_checks=donors,subsets=subsets,
                  checks=['Donors stay within split and exact gene/biotype/region strata; placeholders and singleton strata fixed.',
                          'Prediction IDs and labels align; probabilities finite and normalized.',
                          'Per-class and macro AP independently recomputed with sklearn.',
                          'Two actual cluster draws per subset checked against sklearn sample-weighted AP for all models, classes and scopes.',
                          'All CIs and two-sided empirical P values reproduced from saved 2000-draw distributions.',
                          'BH-FDR independently checked with scipy.stats.false_discovery_control.'],
                  BH_family='Separately per subset and scope: three comparisons per macro class subset; three comparisons times evaluable classes for per-class tests.',
                  macro_support='Exclude classes without test positives; report additional macro excluding classes with fewer than ten positive clusters.',
                  limitation='One training seed (42). Fixed-checkpoint bootstrap describes test-cluster sampling variability, not between-training-seed variability. Conditional predictive comparison, not causal RBP regulation.')
    (O/'matched_rbp/VERIFIED.json').write_text(json.dumps(result,indent=2))
    write(O/'matched_rbp/macro_metrics_summary.tsv',[r for s in subsets for r in s['macro_metrics']])
    write(O/'matched_rbp/bootstrap_macro_summary.tsv',[dict(subset=s,**r) for s in ['all_test','exchangeable_test_only'] for r in read(O/'matched_rbp'/s/'cluster_bootstrap_2000/bootstrap_macro.tsv')])
    write(O/'matched_rbp/bootstrap_per_class_summary.tsv',[dict(subset=s,**r) for s in ['all_test','exchangeable_test_only'] for r in read(O/'matched_rbp'/s/'cluster_bootstrap_2000/bootstrap_per_class.tsv')])
    notes = ['# Matched-stratum RBP control: completed results', '',
             'The final no-pair StructDualRM seed42 configuration was retrained from scratch. Within each split, whole RBP positional tensors were exchanged within exact gene_id x gene_type x region_type strata. Sequence and structural inputs were kept unchanged. Class labels did not define strata. Invalid annotations and singleton strata were left unchanged.', '',
             'Checkpoint selection uses validation Macro AUPR. Both complete-test and exchangeable-site-only analyses use 2,000 paired CD-HIT80-cluster bootstrap draws. These are single-training-seed comparisons, not three-seed means. Classes without test positives are excluded from macro averages. The additional positive_clusters_ge10 summary excludes classes with insufficient independent support.', '',
             '| Test subset | Authentic RBP Macro AUPR | Matched shuffle Macro AUPR | Difference | 95% CI | Two-sided P | BH-FDR |',
             '| --- | ---: | ---: | ---: | --- | ---: | ---: |']
    for subset in ['all_test','exchangeable_test_only']:
        row = next(r for r in read(O/'matched_rbp'/subset/'cluster_bootstrap_2000/bootstrap_macro.tsv')
                   if r['comparison']=='authentic_minus_gene_matched_shuffle' and r['scope']=='standard' and r['class_subset']=='all_evaluable')
        notes.append('| {} | {:.6f} | {:.6f} | {:+.6f} | [{:.6f}, {:.6f}] | {:.7f} | {:.7f} |'.format(
            subset,*[float(row[k]) for k in ['AP_a','AP_b','delta_AUPR','ci95_low','ci95_high','p_two_sided','FDR']]))
    notes += ['', 'The complete per-class results and all comparisons are in bootstrap_per_class_summary.tsv and bootstrap_macro_summary.tsv. Standard and same-base scopes and macro support definitions are recorded in separate columns; do not mix them. Sequence-only comparisons change both RBP and structural inputs, whereas authentic-minus-matched-shuffle holds the architecture and non-RBP input sources fixed.', '',
              'Inference is limited to conditional predictive value under this matching scheme. Gene, biotype and region matching does not remove every contextual confound and does not establish biological causality. Bootstrap uncertainty describes sampling of test clusters, not training-seed variability. Identical donor tensors can remain numerically unchanged despite a valid donor exchange; eligible coverage and changed-tensor coverage are reported separately.', '',
              'VERIFIED.json checks donor constraints, prediction alignment, AP calculations, every saved confidence interval and P value, and BH correction. checkpoint_and_training_identity_audit.json checks checkpoint architecture, configuration, hashes and validation selection. Manuscript and supplementary DOCX files were not modified.', '']
    (O/'matched_rbp/RESULTS_README.md').write_text('\n'.join(notes))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    if '--donors-only' in sys.argv:
        print(json.dumps(validate_donors()[0],indent=2))
    else:
        main()
