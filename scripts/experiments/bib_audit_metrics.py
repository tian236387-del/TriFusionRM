"""Audit frozen benchmark and recompute every class for all staged runs."""
import csv, hashlib, json, sys, time
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support, matthews_corrcoef, balanced_accuracy_score
R=Path('/home/nefu1020220084/summer/StructDualRM/BIB_strengthening_20261028')
while not (R/'audits/STAGING_COMPLETED.json').exists():time.sleep(10)
C=['Am','Cm','Gm','Um','m1A','m5C','m5U','m6A','m6Am','m7G','Y','AtoI']
data=pd.read_csv(R/'data/frozen_v1/master.tsv',sep='\t',dtype=str,keep_default_na=False)
assert data.site_id.is_unique
split='split_cdhit80_cluster'; cluster='cdhit80_cluster_id'
assert data.groupby(cluster)[split].nunique().max()==1
audit={'rows':len(data),'columns':list(data),'split_counts':data[split].value_counts().to_dict(),'class_by_split':pd.crosstab(data.mod_type,data[split]).to_dict(),'cluster_cross_split_overlap':int((data.groupby(cluster)[split].nunique()>1).sum())}
keys=[x for x in ['chrom','chr','chromosome','start','pos','position','strand'] if x in data]
audit['coordinate_columns']=keys
if 'strand' in keys and len(keys)>=3:
    coords=data.groupby(keys).mod_type.nunique()
    conflict_keys=set(coords[coords>1].index)
    conflicts=data[data[keys].apply(tuple,axis=1).isin(conflict_keys)]
    conflicts.to_csv(R/'audits/label_conflicts.tsv',sep='\t',index=False)
    audit['conflict_coordinates']=len(conflict_keys)
    conflict_ids=set(conflicts.site_id)
else:conflict_ids=set()
exact=[]
for length in [41,101,201]:
    col=f'sequence_{length}nt'
    seq=data[col].str.upper().str.replace('U','T',regex=False)
    group=data.assign(sequence=seq).groupby('sequence')[split].agg(lambda s:','.join(sorted(set(s))))
    overlaps=group[group.str.contains(',')]
    for partitions,count in overlaps.value_counts().items():
        exact.append({'length':length,'partitions':partitions,'overlapping_unique_sequences':int(count)})
    if overlaps.empty:exact.append({'length':length,'partitions':'none','overlapping_unique_sequences':0})
    if length>41:
        mismatch=int((seq.str.slice(length//2-20,length//2+21)!=data.sequence_41nt.str.upper().str.replace('U','T',regex=False)).sum())
        audit[f'center_crop_{length}_mismatches']=mismatch
        assert mismatch==0
    # Export unique sequences with split-encoded ids for a separate near-duplicate audit.
    rows=data.loc[~seq.duplicated(),[split]].copy();rows['seq']=seq.loc[rows.index]
    with (R/f'data/frozen_v1/unique_{length}nt.fasta').open('w') as f:
        for idx,row in rows.iterrows():f.write(f'>{row[split]}__{idx}\n{row.seq}\n')
pd.DataFrame(exact).to_csv(R/'audits/cross_split_exact_sequences.tsv',sep='\t',index=False)
(R/'audits/dataset_identity_audit.json').write_text(json.dumps(audit,indent=2))
print('DATASET_AUDIT',json.dumps(audit),flush=True)

master=data.set_index('site_id')
all_overall=[];all_classes=[]
for pred_path in sorted((R/'models').rglob('test_predictions.tsv')):
    if 'statistics_inputs' in str(pred_path) or '/v2/' in str(pred_path):continue
    pred=pd.read_csv(pred_path,sep='\t')
    if not pred.site_id.is_unique:raise ValueError('Duplicate predictions '+str(pred_path))
    subset=master.loc[pred.site_id]
    if not (subset[split]=='test').all():raise ValueError('Non-test predictions '+str(pred_path))
    labels=np.array([C.index(str(x)) for x in pred.true_mod_type])
    assert np.array_equal(labels,np.array([C.index(x) for x in subset.mod_type]))
    probs=pred[[f'prob_{c}' for c in C]].to_numpy(float)
    assert np.isfinite(probs).all() and ((probs>=0)&(probs<=1)).all()
    if 'baseline_' not in str(pred_path):assert np.allclose(probs.sum(1),1,atol=2e-5)
    hard=probs.argmax(1)
    run=str(pred_path.parent.relative_to(R/'models'))
    seed=int(run.rsplit('seed',1)[-1]) if 'seed' in run else -1
    model=run.rsplit('/seed',1)[0]
    prec,rec,f1,_=precision_recall_fscore_support(labels,hard,labels=list(range(12)),zero_division=0)
    class_rows=[]
    for j,c in enumerate(C):
        y=labels==j; support=int(y.sum())
        row={'model':model,'seed':seed,'class':c,'AP':average_precision_score(y,probs[:,j]) if support else np.nan,'AUC':roc_auc_score(y,probs[:,j]) if 0<support<len(y) else np.nan,'Precision':prec[j],'Recall':rec[j],'F1':f1[j],'MCC':matthews_corrcoef(y,hard==j) if support else np.nan,'support':support,'positive_clusters':subset.loc[y,cluster].nunique()}
        class_rows.append(row)
    all_classes.extend(class_rows)
    supported=[x for x in class_rows if x['support']>0]
    summary={'model':model,'seed':seed,'rows':len(labels),'Macro_AP':np.mean([x['AP'] for x in supported]),'Macro_AUC':np.mean([x['AUC'] for x in supported]),'Macro_F1':np.mean([x['F1'] for x in supported]),'Macro_Precision':np.mean([x['Precision'] for x in supported]),'Macro_Recall':np.mean([x['Recall'] for x in supported]),'Macro_MCC':np.mean([x['MCC'] for x in supported]),'Global_MCC':matthews_corrcoef(labels,hard),'Balanced_accuracy':balanced_accuracy_score(labels,hard),'stable_classes_Macro_AP':np.mean([x['AP'] for x in supported if x['positive_clusters']>=10])}
    all_overall.append(summary)
    pd.DataFrame(class_rows).to_csv(pred_path.parent/'test_per_class_recomputed.tsv',sep='\t',index=False)
    normalized=pd.DataFrame({'site_id':pred.site_id,'cluster_id':subset[cluster].to_numpy(),'y_true':labels,'y_pred':hard})
    for j,c in enumerate(C):normalized[f'prob_{c}']=probs[:,j]
    normalized.to_csv(pred_path.parent/'test_predictions_normalized.tsv',sep='\t',index=False)
    print('METRICS',json.dumps(summary),flush=True)
pd.DataFrame(all_classes).to_csv(R/'results/all_runs_per_class_metrics.tsv',sep='\t',index=False)
pd.DataFrame(all_overall).to_csv(R/'results/all_runs_metrics.tsv',sep='\t',index=False)
df=pd.DataFrame(all_overall)
df.groupby('model').agg({c:['count','mean','std'] for c in ['Macro_AP','Macro_F1','Macro_MCC','Global_MCC','Balanced_accuracy']}).to_csv(R/'results/multiseed_summary.tsv',sep='\t')
pd.DataFrame(all_classes).groupby(['model','class']).agg({c:['count','mean','std'] for c in ['AP','AUC','Precision','Recall','F1','MCC']}).to_csv(R/'results/per_class_multiseed_summary.tsv',sep='\t')
(R/'audits/METRICS_COMPLETED.json').write_text(json.dumps({'runs':len(all_overall),'time':time.time()}))
