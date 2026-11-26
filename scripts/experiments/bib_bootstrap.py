"""Paired cluster bootstrap using the audited weighted-AP implementation."""
import argparse,csv,json,multiprocessing as mp,os,time
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score
import cluster_bootstrap_multimodal_parallel as m

def load(path):
    rows=m.read_tsv(Path(path));ids=np.array([x['site_id'] for x in rows],dtype=object)
    if len(set(ids))!=len(ids):raise ValueError('duplicate ids')
    y=np.array([m.CLASSES.index(x['true_mod_type']) for x in rows],dtype=np.int16)
    p=np.array([[float(x['prob_'+c]) for c in m.CLASSES] for x in rows],dtype=np.float32)
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    return ids,y,p

def run(spec):
    R=Path('/home/nefu1020220084/summer/StructDualRM/BIB_strengthening_20261028')
    out=Path(spec['output']);out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETED.json').exists():return
    entries=[];ids=None;y=None
    names=list(spec['models'])
    for name,path in spec['models'].items():
        ci,cy,cp=load(path)
        if ids is None:ids,y=ci,cy
        else:
            index={v:i for i,v in enumerate(ci)}
            assert set(index)==set(ids)
            order=np.array([index[v] for v in ids]);cp=cp[order];cy=cy[order]
            assert np.array_equal(y,cy)
        entries.append(cp)
    p=np.array(entries,dtype=np.float32)
    pp=out/'probabilities.dat';yp=out/'labels.dat';wp=out/'cluster_weights.npy'
    pm=np.memmap(pp,mode='w+',dtype=np.float32,shape=p.shape);pm[:]=p;pm.flush();del pm
    ym=np.memmap(yp,mode='w+',dtype=np.int16,shape=y.shape);ym[:]=y;ym.flush();del ym
    inverse,support,nclusters,attempts=m.generate_weights(Path(spec.get('dataset',R/'data/frozen_v1/master.tsv')),ids,y,spec.get('split','split_cdhit80_cluster'),'test',2000,20261028,wp)
    shapes=(p.shape,y.shape,(2000,len(y)))
    tasks=[(scope,j,k) for scope in ['standard','same_base'] for j,c in enumerate(m.CLASSES) if support[c] for k in range(len(names))]
    observed={(scope,j,k):m.observed_scope_ap(scope,j,p[k],y) for scope,j,k in tasks}
    distributions={}
    with mp.get_context('fork').Pool(processes=min(16,len(tasks)),initializer=m.initialise_worker,initargs=(str(pp),str(yp),str(wp),shapes,16)) as pool:
        for i,(scope,j,k,values) in enumerate(pool.imap_unordered(m.worker,tasks)):
            distributions[(scope,j,k)]=values
            if i%12==0:print('BOOTSTRAP',out.name,i+1,'/',len(tasks),flush=True)
    counts={j:len(np.unique(inverse[y==j])) for j,c in enumerate(m.CLASSES) if support[c]}
    classes=[];macros=[]
    for scope in ['standard','same_base']:
        for comparison,a,b in spec['comparisons']:
            ai=names.index(a);bi=names.index(b)
            delta={j:distributions[(scope,j,ai)]-distributions[(scope,j,bi)] for j in counts}
            for j,values in delta.items():
                classes.append(dict(comparison=comparison,scope=scope,**{'class':m.CLASSES[j]},support=support[m.CLASSES[j]],positive_clusters=counts[j],model_a=a,model_b=b,AP_a=observed[(scope,j,ai)],AP_b=observed[(scope,j,bi)],delta_AUPR=observed[(scope,j,ai)]-observed[(scope,j,bi)],**m.distribution_summary(values)))
            for subset in ['all_evaluable','positive_clusters_ge10']:
                js=[j for j in counts if subset=='all_evaluable' or counts[j]>=10]
                values=np.mean([delta[j] for j in js],axis=0)
                ap_a=np.mean([observed[(scope,j,ai)] for j in js]);ap_b=np.mean([observed[(scope,j,bi)] for j in js])
                macros.append(dict(comparison=comparison,scope=scope,class_subset=subset,classes=len(js),model_a=a,model_b=b,AP_a=ap_a,AP_b=ap_b,delta_AUPR=ap_a-ap_b,**m.distribution_summary(values)))
    for rows in [classes,macros]:
        groups={}
        for i,row in enumerate(rows):groups.setdefault((row['scope'],row.get('class_subset','per_class')),[]).append(i)
        for indices in groups.values():
            for i,q in zip(indices,m.bh([rows[i]['p_two_sided'] for i in indices])):rows[i]['FDR']=float(q)
    m.write_tsv(out/'bootstrap_macro.tsv',macros);m.write_tsv(out/'bootstrap_per_class.tsv',classes)
    np.savez_compressed(out/'bootstrap_distributions.npz',**{f'{s}__{j}__{k}':v for (s,j,k),v in distributions.items()})
    audit=dict(spec,replicates=2000,attempted_replicates=attempts,n_rows=len(y),n_clusters=nclusters,resampling_unit='cdhit80_cluster_id',conditioning='Reject draws omitting positives for any evaluable class.',probabilities='Raw archived float32 probabilities; no probability-floor clipping.',random_seed=20261028,time=time.time())
    (out/'COMPLETED.json').write_text(json.dumps(audit,indent=2))
    pp.unlink();yp.unlink();wp.unlink()
    print('BOOTSTRAP_COMPLETED',json.dumps(macros),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('spec');args=parser.parse_args()
    run(json.loads(Path(args.spec).read_text()))
