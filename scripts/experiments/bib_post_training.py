"""Complete registered analyses when their exact prediction inputs exist."""
import fcntl,json,os,subprocess,time
from pathlib import Path
R=Path('/home/nefu1020220084/summer/StructDualRM/BIB_strengthening_20261028')
lock=(R/'audits/post_training.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
env=dict(os.environ,LD_PRELOAD='/home/nefu1020220084/anaconda3/envs/RNAfold/lib/libstdc++.so.6',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
PY='/home/nefu1020220084/anaconda3/bin/python'
status={}
def record(key,state,**kw):
    status[key]=dict(state=state,time=time.time(),**kw)
    (R/'audits/post_training_status.json').write_text(json.dumps(status,indent=2))

def bootstrap(name,models,comparisons):
    out=R/'results'/name
    if (out/'COMPLETED.json').exists():record(name,'completed');return
    spec=R/'code'/(name+'.json');spec.write_text(json.dumps({'analysis':name,'models':{n:str(p) for n,p in models.items()},'comparisons':comparisons,'output':str(out)},indent=2))
    record(name,'running')
    with (R/'logs'/(name+'.txt')).open('a') as f:
        rc=subprocess.run([PY,'-u',str(R/'code/bib_bootstrap.py'),str(spec)],env=env,cwd=R,stdout=f,stderr=subprocess.STDOUT).returncode
    record(name,'completed' if rc==0 else 'failed',exit_code=rc)

def ensemble(model):
    import numpy as np
    import pandas as pd
    files=[R/'models'/model/f'seed{s}/test_predictions.tsv' for s in [17,42,73]]
    frames=[pd.read_csv(p,sep='\t').set_index('site_id') for p in files]
    ref=frames[0]
    for df in frames[1:]:
        assert set(df.index)==set(ref.index)
        assert (df.loc[ref.index,'true_mod_type']==ref.true_mod_type).all()
    fields=[c for c in ref if c.startswith('prob_')]
    probs=np.mean([df.loc[ref.index,fields].to_numpy(float) for df in frames],axis=0)
    out=ref.copy();out[fields]=probs;out['pred_class_index']=probs.argmax(1)
    out['pred_mod_type']=[fields[j][5:] for j in probs.argmax(1)]
    target=R/'models'/model/'ensemble3';target.mkdir(exist_ok=True)
    out.reset_index().to_csv(target/'test_predictions.tsv',sep='\t',index=False)
    (target/'ensemble_audit.json').write_text(json.dumps({'seeds':[17,42,73],'operation':'arithmetic mean of class probabilities before metrics','inputs':[str(p) for p in files]},indent=2))
    return target/'test_predictions.tsv'

while True:
    if all((R/f'models/sequence_only/seed{s}/TRAINING_COMPLETED.json').exists() for s in [17,73]):
        for seed in [17,73]:
            bootstrap(f'final_vs_seq_seed{seed}_bootstrap_2000',{'final_no_pair':R/f'models/final_no_pair/seed{seed}/test_predictions.tsv','sequence_only':R/f'models/sequence_only/seed{seed}/test_predictions.tsv'},[['final_vs_sequence','final_no_pair','sequence_only']])
        models={n:ensemble(n) for n in ['final_no_pair','sequence_only']}
        bootstrap('final_vs_seq_ensemble3_bootstrap_2000',models,[['final_vs_sequence','final_no_pair','sequence_only']])
        record('multiseed','completed')
    else:record('multiseed','waiting_for_sequence_only_seeds')
    controls=['no_pair_shuffled_rbp','no_pair_shuffled_structure','no_pair_shuffled_bins','no_pair_sequence_structure']
    if all((R/f'models/{n}/seed42/TRAINING_COMPLETED.json').exists() for n in controls):
        models={n:R/f'models/{n}/seed42/test_predictions.tsv' for n in controls+['final_no_pair','sequence_only','reference_sequence_rbp']}
        bootstrap('final_modality_bootstrap_2000',models,[['final_vs_sequence','final_no_pair','sequence_only'],['structure_increment','no_pair_sequence_structure','sequence_only'],['rbp_increment','reference_sequence_rbp','sequence_only'],['real_vs_shuffled_rbp','final_no_pair','no_pair_shuffled_rbp'],['real_vs_shuffled_structure','final_no_pair','no_pair_shuffled_structure'],['real_vs_shuffled_bins','final_no_pair','no_pair_shuffled_bins'],['structure_given_rbp','final_no_pair','reference_sequence_rbp'],['rbp_given_structure','final_no_pair','no_pair_sequence_structure']])
        record('final_controls','completed')
    else:record('final_controls','waiting_for_matched_training')
    if status.get('multiseed',{}).get('state')=='completed' and status.get('final_controls',{}).get('state')=='completed':
        with (R/'logs/recompute_final_metrics.txt').open('w') as f:
            rc=subprocess.run([PY,'-u','/home/nefu1020220084/summer/StructDualRM/bib_audit_metrics.py'],env=env,stdout=f,stderr=subprocess.STDOUT).returncode
        record('metrics_refresh','completed' if rc==0 else 'failed')
        break
    queue=R/'audits/training_queue_status.json'
    if queue.exists() and any(x['state']=='failed' for x in json.loads(queue.read_text()).values()):
        record('training_dependency','failed_requires_repair');break
    time.sleep(30)
