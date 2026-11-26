"""Run fixed paired tests as registered seed predictions become available."""
import fcntl,json,os,subprocess,time
from pathlib import Path
R=Path('/home/nefu1020220084/summer/StructDualRM/BIB_strengthening_20261028')
lock=(R/'audits/followup_statistics.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
PY='/home/nefu1020220084/anaconda3/bin/python';env=dict(os.environ,LD_PRELOAD='/home/nefu1020220084/anaconda3/envs/RNAfold/lib/libstdc++.so.6',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
states={};seen=set()
def bootstrap(version,seed):
    prefix='v2/' if version=='frozen_v2' else ''
    names=['final_no_pair','sequence_only','baseline_multirm','baseline_transrnam']
    files={n:R/f'models/{prefix}{n}/seed{seed}/test_predictions.tsv' for n in names}
    if not all(p.exists() for p in files.values()):return False
    if any((p.parent/'resolved_requested_config.json').exists() and not (p.parent/'TRAINING_COMPLETED.json').exists() for p in files.values()):return False
    name=f'{version}_seed{seed}_matched_bootstrap_2000';out=R/'results'/name
    if (out/'COMPLETED.json').exists():return True
    spec={'models':{n:str(p) for n,p in files.items()},'comparisons':[[f'final_vs_{n}','final_no_pair',n] for n in names[1:]],'output':str(out),'dataset':str(R/f'data/{version}/master.tsv'),'split':'split_v2_cluster' if version=='frozen_v2' else 'split_cdhit80_cluster'}
    config=R/'code'/(name+'.json');config.write_text(json.dumps(spec,indent=2))
    with (R/'logs'/(name+'.txt')).open('w') as f:subprocess.run([PY,'-u',str(R/'code/bib_bootstrap.py'),str(config)],env=env,cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
    return True
while True:
    available={str(p) for p in (R/'models').rglob('test_predictions.tsv')}
    if available!=seen:
        with (R/'logs/unified_metrics.txt').open('a') as f:subprocess.run([PY,'-u',str(R/'code/bib_refresh_metrics.py')],env=env,cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
        seen=available
    for version in ['frozen_v1','frozen_v2']:
        for seed in [42,17,73]:states[f'{version}_seed{seed}']='completed' if bootstrap(version,seed) else 'waiting_for_predictions'
    (R/'audits/followup_statistics_status.json').write_text(json.dumps(states,indent=2))
    if all(v=='completed' for v in states.values()):break
    time.sleep(30)
