"""Verify external bootstrap artifacts and metadata assumptions."""
import hashlib
import json
import platform
from collections import Counter
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support, matthews_corrcoef
from nov04_followup_prepare import R,O,read,write
from nov04_external_a_centered import probs_for,CLASSES,FAMILY5,bh,p_empirical


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    original=R/'results/external_A_centered_cluster_bootstrap_2000'
    dest=O/'external_A_centered'
    dest.mkdir(parents=True,exist_ok=True)
    import shutil
    for path in original.iterdir():
        if path.is_file():shutil.copy2(path,dest/path.name)
    external=Path('/home/nefu1020220084/summer/external_stage3_narmbench_20261017')
    paths={'full':external/'outputs/external_DNP_predictions.tsv','sequence_only':external/'outputs/external_A_predictions.tsv','data':external/'input/data/external_strict_cdhit80.tsv'}
    data=list(read(paths['data']))
    predictions={k:list(read(paths[k])) for k in ('full','sequence_only')}
    assert all([r['site_id'] for r in data]==[r['site_id'] for r in p] for p in predictions.values())
    for p in predictions.values():
        assert all(('I' if r['mod_type']=='AtoI' else r['mod_type'])==('I' if q['external_mod_type']=='AtoI' else q['external_mod_type']) for r,q in zip(data,p))
    indices=[i for i,r in enumerate(data) if r['mod_type'] in ('m1A','m6A','I','AtoI')]
    selected=[data[i] for i in indices]
    seq_col=next((k for k in selected[0] if k.startswith('sequence_') and '41' in k),None)
    if seq_col is None:
        seq_col=next((k for k in ('sequence','seq','sequence_201nt') if k in selected[0]),None)
    assert seq_col is not None, list(selected[0])
    centers=Counter(r[seq_col][len(r[seq_col])//2].upper() for r in selected)
    assert set(centers)=={'A'},centers
    labels=np.array([CLASSES.index('I' if r['mod_type']=='AtoI' else r['mod_type']) for r in selected])
    metrics=[]
    for scenario, family in [('closed3',CLASSES),('open5',FAMILY5)]:
        target_columns=[family.index(c) for c in CLASSES]
        for name,pred in predictions.items():
            rows=[pred[i] for i in indices]
            p=probs_for(rows,family)
            predicted=p.argmax(1)
            y=np.array([family.index(CLASSES[i]) for i in labels])
            prec,rec,f1,support=precision_recall_fscore_support(y,predicted,labels=target_columns,zero_division=0)
            for j,c in enumerate(CLASSES):
                column=family.index(c)
                b=labels==j
                metrics.append(dict(scenario=scenario,model=name,modification=c,support=int(support[j]),
                                    AUPR=average_precision_score(b,p[:,column]),AUC=roc_auc_score(b,p[:,column]),
                                    Precision=prec[j],Recall=rec[j],F1=f1[j],MCC=matthews_corrcoef(b,predicted==column)))
            output=[]
            for i,row in enumerate(rows):
                output.append(dict(site_id=row['site_id'],cluster_id=selected[i]['external_cluster_id'],y_true=CLASSES[labels[i]],y_pred=family[predicted[i]],**{'prob_'+c:p[i,j] for j,c in enumerate(family)}))
            write(dest/(scenario+'_'+name+'_predictions.tsv'),output)
    write(dest/'external_A_centered_all_class_metrics.tsv',metrics)
    distributions=list(read(dest/'external_A_centered_bootstrap_differences.tsv'))
    grouped={}
    for row in distributions:grouped.setdefault((row['scenario'],row['metric']),[]).append(float(row['delta_AUPR']))
    summary=list(read(dest/'external_A_centered_paired_bootstrap.tsv'))
    for row in summary:
        values=grouped[(row['scenario'],row['class'])]
        assert len(values)==2000
        assert np.allclose(np.quantile(values,[.025,.975]),[float(row['CI_low']),float(row['CI_high'])],atol=1e-12)
        row['p_empirical_two_sided']=p_empirical(values)
    for scenario in ('closed3','open5'):
        subset=[row for row in summary if row['scenario']==scenario]
        for row,q in zip(subset,bh([row['p_empirical_two_sided'] for row in subset])):
            row['BH_FDR_within_scenario']=float(q)
    for directory in (original,dest):
        write(directory/'external_A_centered_paired_bootstrap.tsv',summary)
    metadata={k:Counter() for k in ('gene_id','gene_type','region_type')}
    sentinel=Counter()
    for row in read(R/'data/frozen_v1/master.tsv'):
        for key in metadata:
            value=row[key].strip()
            metadata[key][value]+=1
            if value.lower() in ('','na','nan','none','null','.','-','unknown','intergenic'):
                sentinel[(key,value)]+=1
    audit=dict(inputs={k:{'path':str(p),'sha256':sha(p)} for k,p in paths.items()},
               checkpoint_identity_audit=str(R/'results/external_verified/external_effective_checkpoint_identity_audit.json'),
               central_bases=dict(centers),site_order_and_labels_verified=True,bootstrap_quantiles_verified=True,
               python=platform.python_version(),numpy=np.__version__,sklearn=sklearn.__version__,
               BH_family='Macro AUPR and three class-wise tests, corrected independently within each normalization scenario',
               p_formula='min(1,2*(1+min(number(delta<=0),number(delta>=0)))/(2000+1))',
               score_warning='closed3 and open5 use different probability denominators; their absolute AP values answer different tasks.')
    (dest/'VERIFIED.json').write_text(json.dumps(audit,indent=2))
    (O/'matched_rbp/metadata_value_audit.json').write_text(json.dumps({'frequent_values':{k:v.most_common(8) for k,v in metadata.items()},'sentinel_counts':[dict(field=k,value=v,sites=n) for (k,v),n in sentinel.items()]},indent=2))
    print(json.dumps(audit,indent=2),flush=True)
    print('METADATA_SENTINELS',str(sentinel),flush=True)


if __name__=='__main__':main()
