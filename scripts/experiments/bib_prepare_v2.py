"""Versioned conflict-clean benchmark; no changes to historical frozen V1."""
import csv,hashlib,json,os,re,shutil,subprocess,time
from pathlib import Path
import numpy as np
import pandas as pd
R=Path('/home/nefu1020220084/summer/StructDualRM/BIB_strengthening_20261028')
O=R/'data/candidate_v2';O.mkdir(exist_ok=True)
CD='/home/nefu1020220084/anaconda3/envs/RNAfold/bin/cd-hit-est'
plan={'status':'PLAN_FIXED_BEFORE_V2_MODEL_RESULTS','seed':20261028,'classes':['Am','Cm','Gm','Um','m1A','m5C','m5U','m6A','m6Am','m7G','Y','AtoI'],'exclude_all_multilabel_coordinates':True,'clustering':'union of CD-HIT-est representative clusters at 41 and 201 nt; greedy clusters do not prove all cross-split pairs <80%','cdhit_parameters':['-c','0.8','-n','5','-G','1','-aS','0.8','-aL','0.8','-r','0','-T','12','-M','24000','-d','0'],'assignment':'rare-label clusters first, optimize normalized class-size quota deviation; target 80/10/10; enforce independent cluster support per class where feasible','m6Am_min_sites_valid_test':50,'m6Am_min_clusters_valid_test':10,'source_bias_limitation':'New m6Am derives from CROWN-seq HEK293T and may encode source/cap-position confounds; this is a new retrospective benchmark, not independent prospective validation.'}
(O/'V2_PLAN.json').write_text(json.dumps(plan,indent=2))
old=pd.read_csv(R/'data/frozen_v1/master.tsv',sep='\t',keep_default_na=False)
key=['chr','start','strand']
counts=old.groupby(key).mod_type.nunique()
conflict_keys=counts[counts>1].index
bad=pd.MultiIndex.from_frame(old[key]).isin(conflict_keys)
old[bad].to_csv(R/'audits/label_conflicts.tsv',sep='\t',index=False)
old=old[~bad].copy();old['source_dataset']='RMBase_v3_hg38'
new=pd.read_csv(R/'data/m6Am_candidates/CROWNseq_m6Am_nonconflicting_hg38.tsv',sep='\t',keep_default_na=False)
new['source_dataset']='GSE233655_CROWNseq_PCIF1_dependent'
new['strand']=new['strand'].astype(str)
combined=pd.concat([old,new],ignore_index=True).fillna('')
combined=combined.drop_duplicates(key+['mod_type']).reset_index(drop=True)
assert combined.site_id.is_unique
assert combined.groupby(key).mod_type.nunique().max()==1
for n in [41,101,201]:
    seq=combined[f'sequence_{n}nt']
    assert seq.str.len().eq(n).all()
    assert seq.eq(combined.sequence_201nt.str.slice(100-n//2,101+n//2)).all()
combined.to_csv(O/'master_before_split.tsv',sep='\t',index=False)
new.to_csv(O/'new_m6Am_input.tsv',sep='\t',index=False)
parent=np.arange(len(combined))
def find(a):
    while parent[a]!=a:
        parent[a]=parent[parent[a]];a=int(parent[a])
    return a
def union(a,b):
    a=find(a);b=find(b)
    if a!=b:parent[max(a,b)]=min(a,b)
for n in [201,41]:
    fasta=O/f'all_{n}nt.fasta';output=O/f'cdhit80_{n}nt'
    with fasta.open('w') as f:
        for i,s in enumerate(combined[f'sequence_{n}nt']):f.write(f'>r{i}\n{s}\n')
    command=[CD,'-i',str(fasta),'-o',str(output)]+plan['cdhit_parameters']
    (O/f'command_{n}nt.json').write_text(json.dumps(command))
    if not Path(str(output)+'.SUCCESS.json').exists():
        with (R/'logs'/f'v2_cdhit_{n}nt.txt').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        Path(str(output)+'.SUCCESS.json').write_text(json.dumps({'time':time.time(),'command':command}))
    seen=0;first=None
    for line in Path(str(output)+'.clstr').open():
        if line.startswith('>'):first=None;continue
        match=re.search(r'>r(\d+)\.\.\.',line);assert match,line
        i=int(match[1]);seen+=1
        if first is None:first=i
        else:union(first,i)
    assert seen==len(combined),(seen,len(combined))
combined['v2_cluster_id']=[f'v2_{find(i)}' for i in range(len(combined))]
combined['v1_cdhit80_cluster_id']=combined['cdhit80_cluster_id']
combined['cdhit80_cluster_id']=combined['v2_cluster_id']
classes=plan['classes'];groups=pd.crosstab(combined.v2_cluster_id,combined.mod_type).reindex(columns=classes,fill_value=0)
available=(groups>0).sum(axis=0)
totals=groups.sum(axis=0).values.astype(float)
rng=np.random.default_rng(plan['seed'])
rarity=(groups.values/np.maximum(totals,1)).max(axis=1)
order=np.lexsort((rng.random(len(groups)),-groups.sum(axis=1).values,-rarity))
target=np.array([.8,.1,.1])[:,None]*totals[None,:]
assigned=np.zeros((3,len(classes)));clcount=np.zeros_like(assigned);assignment={}
for i in order:
    row=groups.iloc[i].values.astype(float);present=row>0
    # Scarce labels get one whole cluster in each split before quota optimization.
    scarce=present & (available.values<=6) & (available.values>=3)
    choices=np.arange(3)
    if scarce.any():
        need=np.sum(clcount[:,scarce]==0,axis=1)
        if need.max()>0:choices=choices[need==need.max()]
    costs=[]
    for s in choices:
        before=(assigned[s]-target[s])/np.maximum(totals,1)
        after=(assigned[s]+row-target[s])/np.maximum(totals,1)
        costs.append(np.sum(after**2-before**2))
    s=int(choices[int(np.argmin(costs))]);assigned[s]+=row;clcount[s]+=present
    assignment[groups.index[i]]=['train','valid','test'][s]
combined['split_v2_cluster']=combined.v2_cluster_id.map(assignment)
assert combined.groupby('v2_cluster_id').split_v2_cluster.nunique().max()==1
support=combined.groupby(['split_v2_cluster','mod_type']).agg(sites=('site_id','size'),clusters=('v2_cluster_id','nunique')).reset_index()
support.to_csv(O/'split_class_support.tsv',sep='\t',index=False)
checks={}
for split in ['valid','test']:
    v=support[(support.split_v2_cluster==split)&(support.mod_type=='m6Am')]
    checks[split]=len(v)==1 and int(v.iloc[0].sites)>=50 and int(v.iloc[0].clusters)>=10
for n in [41,101,201]:assert combined.groupby(f'sequence_{n}nt').split_v2_cluster.nunique().max()==1
combined.to_csv(O/'master.tsv',sep='\t',index=False)
audit={'status':'CLUSTERED_QC_PASSED_AWAITING_FEATURES' if all(checks.values()) else 'INSUFFICIENT_m6Am_CLUSTER_SUPPORT','sites':len(combined),'clusters':len(groups),'removed_conflict_records':int(bad.sum()),'new_source_sites':int((combined.source_dataset!='RMBase_v3_hg38').sum()),'m6Am_checks':checks,'plan':plan,'master_sha256':hashlib.sha256((O/'master.tsv').read_bytes()).hexdigest()}
(O/'V2_CLUSTER_AUDIT.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit),flush=True)
