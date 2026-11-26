"""Prepare auditable long-context sensitivity and matched-RBP follow-ups."""
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support, matthews_corrcoef

R = Path('/home/nefu1020220084/summer/StructDualRM/BIB_strengthening_20261028')
O = R / 'results/reviewer_followup_20261104'
CLASSES = ['Am','Cm','Gm','Um','m1A','m5C','m5U','m6A','m6Am','m7G','Y','AtoI']


def read(path):
    with Path(path).open(newline='') as f:
        yield from csv.DictReader(f, delimiter='\t')


def write(path, rows):
    rows = list(rows)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t', lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def fasta_sequences(path):
    sequences = set()
    seq = []
    with Path(path).open() as f:
        for line in f:
            if line.startswith('>'):
                if seq:
                    sequences.add(''.join(seq))
                seq = []
            else:
                seq.append(line.strip().upper().replace('U', 'T'))
    if seq:
        sequences.add(''.join(seq))
    return sequences


def point_metrics(prediction, model, subset, output):
    rows = list(read(prediction))
    y = np.array([CLASSES.index(r['true_mod_type']) for r in rows])
    p = np.array([[float(r['prob_'+c]) for c in CLASSES] for r in rows])
    yp = p.argmax(1)
    precision, recall, f1, support = precision_recall_fscore_support(y, yp, labels=list(range(12)), zero_division=0)
    values = []
    for j, c in enumerate(CLASSES):
        b = y == j
        values.append(dict(model=model, subset=subset, modification=c, support=int(support[j]),
                           AUPR=average_precision_score(b,p[:,j]) if b.any() else '',
                           AUC=roc_auc_score(b,p[:,j]) if b.any() and (~b).any() else '',
                           Precision=precision[j], Recall=recall[j], F1=f1[j],
                           MCC=matthews_corrcoef(b,yp==j) if b.any() else ''))
    write(output, values)
    return values


def main():
    O.mkdir(parents=True, exist_ok=True)
    near = R / 'audits/near_sequence_audit'
    no_match = {n: fasta_sequences(near / ('train_vs_test_'+str(n))) & fasta_sequences(near / ('valid_vs_test_'+str(n))) for n in (101,201)}
    master = R / 'data/frozen_v1/master.tsv'
    counts = defaultdict(Counter)
    clean = []
    flags = []
    metadata = defaultdict(list)
    current_by_id = {}
    for row in read(master):
        split = row['split_cdhit80_cluster']
        counts[split][row['mod_type']] += 1
        current_by_id[row['site_id']] = (split, row['mod_type'])
        metadata[split].append({k:row[k] for k in ('site_id','mod_type','gene_id','gene_type','region_type')})
        if split != 'test':
            continue
        passes = {n:row['sequence_'+str(n)+'nt'].upper().replace('U','T') in no_match[n] for n in (101,201)}
        eligible = all(passes.values())
        flags.append(dict(site_id=row['site_id'], cluster_id=row['cdhit80_cluster_id'], modification=row['mod_type'],
                          no_detected_train_or_validation_match_101nt=passes[101],
                          no_detected_train_or_validation_match_201nt=passes[201], retained=eligible))
        if eligible:
            clean.append(row)
    write(O/'long_context/test_screen_flags.tsv', flags)
    write(O/'long_context/screened_test_master.tsv', clean)
    retained = Counter(row['mod_type'] for row in clean)
    write(O/'long_context/per_class_retention.tsv', [dict(modification=c, original_test_sites=counts['test'][c], retained_sites=retained[c], retained_fraction=retained[c]/max(1,counts['test'][c])) for c in CLASSES])
    summaries = list(read(R/'audits/cross_split_sequence_similarity.tsv'))
    for row in summaries:
        row['detected_match_fraction'] = int(row['query_with_detected_cross_split_match']) / int(row['query_unique_sequences'])
    write(O/'long_context/cross_split_similarity_with_fractions.tsv',summaries)
    selected = {r['site_id'] for r in clean}
    models = {'final_no_pair':'final_no_pair', 'sequence_only':'sequence_only', 'MultiRM':'baseline_multirm', 'TransRNAm':'baseline_transrnam', 'EvoRMD_safe':'baseline_evormd_safe'}
    paths = {}
    all_metrics = []
    for name, folder in models.items():
        source = R/'models'/folder/'seed42/test_predictions.tsv'
        rows = [r for r in read(source) if r['site_id'] in selected]
        assert len(rows) == len(selected) and len({r['site_id'] for r in rows}) == len(selected)
        path = O/'long_context/predictions'/ (name+'.tsv')
        write(path,rows)
        paths[name]=str(path)
        all_metrics.extend(point_metrics(path,name,'no_detected_101_and_201_train_or_validation_match',O/'long_context'/ (name+'_per_class_metrics.tsv')))
    write(O/'long_context/all_models_per_class_metrics.tsv',all_metrics)
    spec = dict(output=str(O/'long_context/cluster_bootstrap_2000'),dataset=str(O/'long_context/screened_test_master.tsv'),models=paths,
                comparisons=[['final_vs_'+n,'final_no_pair',n] for n in models if n!='final_no_pair'])
    (O/'long_context/bootstrap_spec.json').write_text(json.dumps(spec,indent=2))
    audit = dict(original_test_sites=len(flags),retained_sites=len(clean),retained_clusters=len({r['cdhit80_cluster_id'] for r in clean}),
                 screen_source=str(near),screen_conditions='CD-HIT-est-2d, identity=.8, short/long coverage=.8, same strand',
                 selection='Require no detected train match and no detected validation match at BOTH 101 and 201 nt',
                 limitation='Post hoc test-subset sensitivity using frozen checkpoints, not a newly trained strict split or an exhaustive all-pairs independence proof.')
    (O/'long_context/audit.json').write_text(json.dumps(audit,indent=2))
    # Reconstruct the deterministic permutation without loading the RBP feature tensor.
    coverage = []
    for split, rows in metadata.items():
        groups=defaultdict(list)
        for i,row in enumerate(rows):
            values=tuple(row[k].strip() for k in ('gene_id','gene_type','region_type'))
            missing={'','na','nan','none','null','.','-','unknown'}
            invalid=values[0].lower() in missing|{'intergenic'} or values[1].lower() in missing|{'intergenic'} or values[2].lower() in missing
            key=('singleton',str(i)) if invalid else values
            groups[key].append(i)
        digest=hashlib.sha256(split.encode()).digest()
        rng=np.random.default_rng(42+int.from_bytes(digest[:4],'little'))
        donor=np.arange(len(rows))
        for indices in groups.values():
            if len(indices)<2:continue
            ordered=rng.permutation(indices)
            donor[ordered]=np.roll(ordered,int(rng.integers(1,len(indices))))
        mapping=[dict(site_id=row['site_id'], donor_site_id=rows[int(donor[i])]['site_id'], modification=row['mod_type'], exchangeable=bool(donor[i]!=i)) for i,row in enumerate(rows)]
        write(O/'matched_rbp'/ (split+'_donor_mapping.tsv'),mapping)
        for c in CLASSES:
            total=sum(r['mod_type']==c for r in rows)
            changed=sum(r['modification']==c and r['exchangeable'] for r in mapping)
            coverage.append(dict(split=split,modification=c,sites=total,exchangeable_sites=changed,fraction=changed/max(1,total)))
    write(O/'matched_rbp/per_class_exchangeable_coverage.tsv',coverage)
    (O/'PREPARATION_COMPLETED.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':
    main()
