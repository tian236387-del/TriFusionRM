import json,time
from pathlib import Path
R=Path('/home/nefu1020220084/summer/StructDualRM/BIB_strengthening_20261028')
checks={'primary_seed42_bootstrap_2000':'results/primary_seed42_bootstrap_2000/COMPLETED.json','baseline_seed42_bootstrap_2000':'results/final_vs_baselines_bootstrap_2000/COMPLETED.json','m6Am_site_qc':'data/m6Am_candidates/CROWNseq_m6Am_qc_audit.json','v2_clusters':'data/candidate_v2/V2_CLUSTER_AUDIT.json','v2_features':'data/candidate_v2/FEATURES_COMPLETED.json','v2_frozen':'data/frozen_v2/FROZEN.json','near_duplicate_screen':'audits/near_sequence_audit/COMPLETED.json','external_audit':'results/external_verified/external_identity_and_overlap_audit.json'}
state={k:{'output':str(R/v),'exists':(R/v).exists()} for k,v in checks.items()}
for name in ['training_queue_status','post_training_status','followup_queue_status','followup_statistics_status']:
    path=R/f'audits/{name}.json'
    if path.exists():state[name]=json.loads(path.read_text())
(R/'EXPERIMENT_PROGRESS.json').write_text(json.dumps({'updated_unix':time.time(),'tasks':state,'manuscript_modified':False,'interpretation':'Only completed marker files establish completed tasks. Waiting and running jobs are not results.'},indent=2))
print(json.dumps(state,indent=2))
