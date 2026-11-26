"""Uniform point estimates for each available run, split and dataset version."""
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score,precision_recall_fscore_support,matthews_corrcoef
R=Path('/home/nefu1020220084/summer/StructDualRM/BIB_strengthening_20261028');C=['Am','Cm','Gm','Um','m1A','m5C','m5U','m6A','m6Am','m7G','Y','AtoI']
overall=[];perclass=[]
for version in ['frozen_v1','frozen_v2']:
    path=R/f'data/{version}/master.tsv'
    if not path.exists():continue
    split='split_v2_cluster' if version=='frozen_v2' else 'split_cdhit80_cluster'
    frame=pd.read_csv(path,sep='\t',usecols=['site_id','mod_type',split,'cdhit80_cluster_id'],dtype=str).set_index('site_id')
    for p in sorted((R/'models').rglob('test_predictions.tsv')):
        if ('/v2/' in str(p))!=(version=='frozen_v2'):continue
        if (p.parent/'resolved_requested_config.json').exists() and not (p.parent/'TRAINING_COMPLETED.json').exists():continue
        df=pd.read_csv(p,sep='\t');assert df.site_id.is_unique
        ref=frame.loc[df.site_id];assert ref[split].eq('test').all()
        assert np.array_equal(df.true_mod_type.values,ref.mod_type.values)
        y=np.array([C.index(x) for x in df.true_mod_type]);prob=df[['prob_'+c for c in C]].values
        assert np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all()
        hard=prob.argmax(1);prec,rec,f1,_=precision_recall_fscore_support(y,hard,labels=range(12),zero_division=0)
        name=str(p.parent.relative_to(R/'models'));rows=[]
        for j,c in enumerate(C):
            pos=y==j;n=int(pos.sum());row={'dataset_version':version,'run':name,'class':c,'AP':average_precision_score(pos,prob[:,j]) if n else np.nan,'AUC':roc_auc_score(pos,prob[:,j]) if 0<n<len(y) else np.nan,'Precision':prec[j],'Recall':rec[j],'F1':f1[j],'MCC':matthews_corrcoef(pos,hard==j) if n else np.nan,'support':n,'positive_clusters':ref.loc[pos,'cdhit80_cluster_id'].nunique()}
            rows.append(row)
        perclass.extend(rows);valid=[r for r in rows if r['support']]
        overall.append({'dataset_version':version,'run':name,'n_test':len(y),'classes_evaluable':len(valid),**{'Macro_'+k:np.mean([r[k] for r in valid]) for k in ['AP','AUC','Precision','Recall','F1','MCC']},'Global_MCC':matthews_corrcoef(y,hard),'Balanced_accuracy':np.mean([r['Recall'] for r in valid]),'prediction_sha256':__import__('hashlib').sha256(p.read_bytes()).hexdigest()})
        pd.DataFrame(rows).to_csv(p.parent/'test_per_class_recomputed.tsv',sep='\t',index=False)
pd.DataFrame(overall).to_csv(R/'results/unified_all_runs_metrics.tsv',sep='\t',index=False)
pd.DataFrame(perclass).to_csv(R/'results/unified_all_runs_per_class_metrics.tsv',sep='\t',index=False)
(R/'audits/unified_metrics_status.json').write_text(json.dumps({'runs':len(overall),'time':time.time()}))
print('UNIFIED_METRICS',len(overall),flush=True)
