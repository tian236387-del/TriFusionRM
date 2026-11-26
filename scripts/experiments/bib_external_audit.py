import hashlib,json,shutil,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score,precision_recall_fscore_support,matthews_corrcoef
R=Path('/home/nefu1020220084/summer/StructDualRM/BIB_strengthening_20261028');E=R/'data/external/narmbench';O=R/'results/external_verified';O.mkdir(exist_ok=True)
manifest=json.loads((E/'outputs/external_stage3_manifest.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(R/'models/final_no_pair/seed42/model.pt')==manifest['checkpoints']['D-NP']['sha256']
sequence_identity=sha(R/'models/sequence_only/seed42/model.pt')==manifest['checkpoints']['Model_A']['sha256']
external=pd.read_csv(E/'input_data/external_strict_cdhit80.tsv',sep='\t',dtype=str)
print('EXTERNAL_COLUMNS',list(external),flush=True)
master=pd.read_csv(R/'data/frozen_v1/master.tsv',sep='\t',usecols=['site_id','chr','start','strand','sequence_41nt','sequence_101nt','sequence_201nt'],dtype=str)
audit={'status':'DNP_IDENTITY_VERIFIED_SEQUENCE_COMPARATOR_REQUIRES_RECONCILIATION' if not sequence_identity else 'CHECKPOINT_IDENTITIES_VERIFIED','DNP_matches_final_no_pair_seed42':True,'external_Model_A_matches_current_sequence_only_seed42':sequence_identity,'historical_manifest':manifest,'external_rows':len(external),'exact_sequence_overlap_with_any_V1_split':{},'comparison_policy':'If Model_A hash differs, retain historical external comparison under its archived identity; do not present it as the current final-versus-sequence matched comparison.'}
for n in [41,101,201]:
    col=f'sequence_{n}nt' if f'sequence_{n}nt' in external else f'sequence_{n}'
    if col in external:
        seq=external[col].str.upper().str.replace('U','T',regex=False)
        audit['exact_sequence_overlap_with_any_V1_split'][str(n)]=int(seq.isin(set(master[f'sequence_{n}nt'].str.upper().str.replace('U','T',regex=False))).sum())
C=['Am','Cm','Gm','Um','m1A','m5C','m5U','m6A','m6Am','m7G','Y','AtoI'];rows=[]
for name in ['DNP','A']:
    path=E/f'outputs/external_{name}_predictions.tsv';df=pd.read_csv(path,sep='\t')
    label='external_mod_type'
    labels=df[label].replace({'I':'AtoI','Psi':'Y','\u03a8':'Y'}).map({c:i for i,c in enumerate(C)}).values
    assert np.isfinite(labels).all()
    prob=df[['p_'+c for c in C]].values;hard=prob.argmax(1);pr,re,f1,_=precision_recall_fscore_support(labels,hard,labels=range(12),zero_division=0)
    for j,c in enumerate(C):
        y=labels==j;n=int(y.sum())
        rows.append({'model':name,'class':c,'support':n,'AP':average_precision_score(y,prob[:,j]) if n else np.nan,'AUC':roc_auc_score(y,prob[:,j]) if 0<n<len(y) else np.nan,'Precision':pr[j],'Recall':re[j],'F1':f1[j],'MCC':matthews_corrcoef(y,hard==j) if n else np.nan})
pd.DataFrame(rows).to_csv(O/'external_per_class_recomputed.tsv',sep='\t',index=False)
(O/'external_identity_and_overlap_audit.json').write_text(json.dumps(audit,indent=2))
for name in ['external_paired_bootstrap_macro.tsv','external_paired_bootstrap_per_class.tsv','external_restricted6_metrics.tsv','external_rbp_coverage_by_class.tsv']:
    shutil.copy2(E/'outputs'/name,O/name)
print('EXTERNAL_AUDIT_COMPLETED',json.dumps({k:v for k,v in audit.items() if k!='historical_manifest'}),flush=True)
