"""Dependency-aware continuation: fixed motif checks, baseline seeds, V2 retraining."""
import concurrent.futures,csv,fcntl,hashlib,json,os,shutil,subprocess,threading,time
from pathlib import Path
R=Path('/home/nefu1020220084/summer/StructDualRM/BIB_strengthening_20261028')
lock=(R/'audits/followup_queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
PY='/home/nefu1020220084/anaconda3/envs/RNAfold/bin/python'
states={};mutex=threading.Lock()
def record(name,state,**kw):
    with mutex:
        states[name]=dict(state=state,time=time.time(),**kw)
        tmp=R/'audits/followup_queue_status.tmp';tmp.write_text(json.dumps(states,indent=2));tmp.replace(R/'audits/followup_queue_status.json')
def run(name,cmd,gpu):
    done=R/'audits'/(name+'.COMPLETED.json')
    if done.exists():record(name,'completed_previously');return
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),LD_PRELOAD='/home/nefu1020220084/anaconda3/envs/RNAfold/lib/libstdc++.so.6',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    with (R/'logs'/(name+'.txt')).open('a') as f:
        child=subprocess.Popen(cmd,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT)
        record(name,'running',gpu=gpu,pid=child.pid,command=cmd)
        rc=child.wait()
    record(name,'completed' if rc==0 else 'failed',exit_code=rc)
    if rc:raise RuntimeError(f'{name} failed: {rc}; inspect log')
    done.write_text(json.dumps({'time':time.time(),'command':cmd}))
record('dependency','waiting_for_initial_GPU_queue')
while True:
    status=json.loads((R/'audits/training_queue_status.json').read_text())
    if any(v['state']=='failed' for v in status.values()):raise RuntimeError('Initial queue failed; no further GPU jobs launched')
    if all(v['state']=='completed' for v in status.values()):break
    time.sleep(30)
# Keep sample selection and editing seeds identical across checkpoint seeds.
import pandas as pd
df=pd.read_csv(R/'data/frozen_v1/master.tsv',sep='\t',keep_default_na=False)
test=df[df.split_cdhit80_cluster=='test'].copy();test['split']='test';test['cluster_id']=test.cdhit80_cluster_id;test['y_true']=test.class_index
testpath=R/'data/frozen_v1/test_inputs_attribution.tsv';test.to_csv(testpath,sep='\t',index=False)
old_manifest=R.parent.parent/'m1A_m6A/rmbase12_work/motif_attribution_experiment1_20261011/motif_attribution_site_manifest.tsv'
shutil.copy2(old_manifest,R/'data/frozen_v1/original_motif_attribution_site_manifest.tsv')
del df,test
def first_lane(gpu):
    for seed in ([42,73] if gpu==0 else [17]):
        name=f'motif_distance_matched_seed{seed}';out=R/'results'/name
        cmd=[PY,'-u',str(R/'code/run_motif_distance_matched.py'),'--test-dataset',str(testpath),'--common-dataset',str(R/'data/frozen_v1/master.tsv'),'--structure',str(R/'data/frozen_v1/rnafold.tsv'),'--rbp-positional',str(R/'data/frozen_v1/rbp_positional.tsv'),'--config',str(R/f'models/final_no_pair/seed{seed}/centralized_config.json'),'--model',str(R/f'models/final_no_pair/seed{seed}/model.pt'),'--out-dir',str(out),'--seed','20261011','--bootstrap','2000']
        run(name,cmd,gpu)
        original=pd.read_csv(old_manifest,sep='\t');current=pd.read_csv(out/'motif_attribution_site_manifest.tsv',sep='\t')
        fields=['source_class','motif','cluster_id','site_id']
        assert set(map(tuple,original[fields].values))==set(map(tuple,current[fields].values)),'Motif selection changed'
    for seed in [17,73]:
        model=['multirm','transrnam'][gpu]
        job={'name':f'baseline_{model}_seed{seed}','seed':seed,'template':f'models/baseline_{model}/seed42/training_config.json','out_dir':f'models/baseline_{model}/seed{seed}'}
        spec=R/'code'/(job['name']+'.json');spec.write_text(json.dumps(job,indent=2))
        run(job['name'],[PY,'-u',str(R/'code/bib_train_baseline_one.py'),str(spec)],gpu)
with concurrent.futures.ThreadPoolExecutor(2) as ex:list(ex.map(first_lane,[0,1]))
record('v2_dependency','waiting_for_clusters_and_features')
O=R/'data/candidate_v2';F=R/'data/frozen_v2'
while not (F/'FROZEN.json').exists():
    if (O/'V2_CLUSTER_AUDIT.json').exists() and (O/'FEATURES_COMPLETED.json').exists():
        audit=json.loads((O/'V2_CLUSTER_AUDIT.json').read_text());assert audit['status']=='CLUSTERED_QC_PASSED_AWAITING_FEATURES',audit
        # A producer already running during code update can retain the V1 column;
        # normalize it explicitly before freeze and preserve historical provenance.
        frame=pd.read_csv(O/'master.tsv',sep='\t',keep_default_na=False)
        if not frame.cdhit80_cluster_id.eq(frame.v2_cluster_id).all():
            frame['v1_cdhit80_cluster_id']=frame.cdhit80_cluster_id;frame['cdhit80_cluster_id']=frame.v2_cluster_id
            frame.to_csv(O/'master.tsv',sep='\t',index=False)
        assert frame.groupby('cdhit80_cluster_id').split_v2_cluster.nunique().max()==1
        del frame
        assert not F.exists();O.rename(F)
        metadata=json.loads((F/'rbp_positional.tsv.json').read_text());metadata.update(dataset=str(F/'master.tsv'),output=str(F/'rbp_positional.tsv'))
        (F/'rbp_positional.tsv.json').write_text(json.dumps(metadata,indent=2))
        digest=hashlib.sha256((F/'master.tsv').read_bytes()).hexdigest()
        (F/'FROZEN.json').write_text(json.dumps({'time':time.time(),'master_sha256':digest,'split':'split_v2_cluster','cluster':'v2_cluster_id','policy':'Immutable new benchmark; V1 is retained as historical sensitivity benchmark.'},indent=2))
        registry=json.loads((R/'data/DATA_REGISTRY.json').read_text());registry['frozen_v2']={'root':str(F),'master_sha256':digest,'split':'split_v2_cluster'};registry['new_12_class_benchmark']='frozen_v2'
        (R/'data/DATA_REGISTRY.json').write_text(json.dumps(registry,indent=2));break
    time.sleep(30)
jobs=[]
for model in ['final_no_pair','sequence_only']:
    for i,seed in enumerate([42,17,73]):
        jobs.append({'name':f'v2_{model}_seed{seed}','gpu':len(jobs)%2,'template':f'models/{model}/seed42/training_config.json','out_dir':f'models/v2/{model}/seed{seed}','dataset_version':'frozen_v2','overrides':{'seed':seed}})
for model in ['multirm','transrnam']:
    for seed in [42,17,73]:jobs.append({'name':f'v2_baseline_{model}_seed{seed}','gpu':len(jobs)%2,'template':f'models/baseline_{model}/seed42/training_config.json','out_dir':f'models/v2/baseline_{model}/seed{seed}','dataset_version':'frozen_v2','seed':seed})
(R/'audits/v2_training_plan.json').write_text(json.dumps({'jobs':jobs,'selection':'validation only','seeds_fixed_before_results':True},indent=2))
def v2lane(gpu):
    for job in [j for j in jobs if j['gpu']==gpu]:
        spec=R/'code'/(job['name']+'.json');spec.write_text(json.dumps(job,indent=2))
        script='bib_train_baseline_one.py' if 'baseline' in job['name'] else 'bib_train_one.py'
        run(job['name'],[PY,'-u',str(R/'code'/script),str(spec)],gpu)
with concurrent.futures.ThreadPoolExecutor(2) as ex:list(ex.map(v2lane,[0,1]))
record('all_scheduled_GPU_followups','completed')
