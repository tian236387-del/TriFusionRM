"""Two GPU lanes; durable status files, no duplicate queue execution."""
import concurrent.futures,fcntl,json,os,subprocess,time
from pathlib import Path
R=Path('/home/nefu1020220084/summer/StructDualRM/BIB_strengthening_20261028')
lock=(R/'audits/training_queue.lock').open('w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
while not (R/'audits/STAGING_COMPLETED.json').exists():time.sleep(10)
jobs=[
 {'name':'sequence_only_seed17','gpu':0,'template':'models/sequence_only/seed42/training_config.json','out_dir':'models/sequence_only/seed17','overrides':{'seed':17}},
 {'name':'sequence_only_seed73','gpu':1,'template':'models/sequence_only/seed42/training_config.json','out_dir':'models/sequence_only/seed73','overrides':{'seed':73}},
 {'name':'no_pair_shuffled_rbp_seed42','gpu':0,'template':'models/final_no_pair/seed42/training_config.json','out_dir':'models/no_pair_shuffled_rbp/seed42','overrides':{'seed':42,'shuffle_rbp':True}},
 {'name':'no_pair_shuffled_structure_seed42','gpu':1,'template':'models/final_no_pair/seed42/training_config.json','out_dir':'models/no_pair_shuffled_structure/seed42','overrides':{'seed':42,'shuffle_structure':True}},
 {'name':'no_pair_sequence_structure_seed42','gpu':0,'template':'models/final_no_pair/seed42/training_config.json','out_dir':'models/no_pair_sequence_structure/seed42','overrides':{'seed':42,'use_rbp_positional':False}},
 {'name':'no_pair_shuffled_bins_seed42','gpu':1,'template':'models/final_no_pair/seed42/training_config.json','out_dir':'models/no_pair_shuffled_bins/seed42','overrides':{'seed':42,'shuffle_rbp_bins':True}},
]
(R/'audits/training_plan.json').write_text(json.dumps({'fixed_before_new_results':True,'jobs':jobs,'policy':'Validation checkpoint selection; do not select jobs, seeds or epochs using test performance.'},indent=2))
states={x['name']:{'state':'queued','gpu':x['gpu']} for x in jobs}
def status():
    temp=R/'audits/training_queue_status.tmp';temp.write_text(json.dumps(states,indent=2));temp.replace(R/'audits/training_queue_status.json')
import threading
mutex=threading.Lock()
def lane(gpu):
    for job in [x for x in jobs if x['gpu']==gpu]:
        out=R/job['out_dir']
        if (out/'TRAINING_COMPLETED.json').exists():
            states[job['name']]['state']='completed';continue
        spec=R/'code'/(job['name']+'.json');spec.write_text(json.dumps(job,indent=2))
        log=R/'logs'/(job['name']+'.txt')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4',LD_PRELOAD='/home/nefu1020220084/anaconda3/envs/RNAfold/lib/libstdc++.so.6',PYTHONUNBUFFERED='1')
        with log.open('a') as f:
            child=subprocess.Popen(['/home/nefu1020220084/anaconda3/envs/RNAfold/bin/python','-u',str(R/'code/bib_train_one.py'),str(spec)],env=env,cwd=R,stdout=f,stderr=subprocess.STDOUT)
            with mutex:
                states[job['name']].update(state='running',pid=child.pid,started=time.time(),log=str(log));status()
            rc=child.wait()
        with mutex:
            states[job['name']].update(state='completed' if rc==0 else 'failed',exit_code=rc,ended=time.time());status()
        if rc!=0:
            # Stop this lane on environment or configuration errors; do not burn through jobs.
            return
status()
with concurrent.futures.ThreadPoolExecutor(2) as ex:list(ex.map(lane,[0,1]))
print('QUEUE_FINISHED',json.dumps(states),flush=True)
