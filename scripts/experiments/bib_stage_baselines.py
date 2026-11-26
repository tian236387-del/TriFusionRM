import csv,hashlib,json,shutil,subprocess,os
from pathlib import Path
import pandas as pd
R=Path('/home/nefu1020220084/summer/StructDualRM/BIB_strengthening_20261028');S=R.parent.parent/'m1A_m6A'
B=S/'rmbase12_work/official_reproduction_benchmark'
sources={'multirm':B/'multirm12_cdhit80_strict_split/multirm_official','transrnam':B/'multirm12_cdhit80_strict_split/transrnam_official','evormd_safe':B/'evormd_rnafm_official_core_safe_context_cdhit80_seed42'}
master=pd.read_csv(R/'data/frozen_v1/master.tsv',sep='\t',usecols=['site_id','mod_type','split_cdhit80_cluster'])
test=master[master.split_cdhit80_cluster=='test'].set_index('site_id');audit=[]
for name,source in sources.items():
    dest=R/f'models/baseline_{name}/seed42';dest.mkdir(parents=True,exist_ok=True)
    for filename in ['training_config.json','metrics.json','source_info.json','model.pt','best_model.pt']:
        if (source/filename).exists():shutil.copy2(source/filename,dest/filename)
    pred=pd.read_csv(source/'test_predictions.tsv',sep='\t');n=len(pred)
    pred=pred[pred.site_id.isin(test.index)].copy()
    assert pred.site_id.is_unique and set(pred.site_id)==set(test.index),(name,len(pred),len(test))
    assert (pred.true_mod_type.to_numpy()==test.loc[pred.site_id].mod_type.to_numpy()).all()
    pred.to_csv(dest/'test_predictions.tsv',sep='\t',index=False)
    cfg=json.loads((dest/'training_config.json').read_text());cfg.update(dataset=str(R/'data/frozen_v1/master.tsv'),embedding_path=str(R/'data/reference/embeddings_12RM.pkl'))
    (dest/'centralized_config.json').write_text(json.dumps(cfg,indent=2))
    audit.append({'model':name,'source':str(source),'training_config':str(dest/'training_config.json'),'original_prediction_rows':n,'common_test_rows':len(pred),'test_labels_identical':True,'evaluation':'Restrict archived predictions to exactly frozen master test sites. Point estimates differ from archived all-sites metrics.','task_adaptation':'MultiRM/TransRNAm core adaptations to 12-type benchmark; archived code source audit retained, not asserted identical to original publication task.','config':cfg})
(R/'audits/baseline_reproduction_audit.json').write_text(json.dumps(audit,indent=2))
spec={'output':str(R/'results/final_vs_baselines_bootstrap_2000'),'models':{'final_no_pair':str(R/'models/final_no_pair/seed42/test_predictions.tsv'),**{n:str(R/f'models/baseline_{n}/seed42/test_predictions.tsv') for n in sources}},'comparisons':[[f'final_no_pair_vs_{n}','final_no_pair',n] for n in sources]}
(R/'code/baseline_bootstrap.json').write_text(json.dumps(spec,indent=2))
subprocess.run(['/home/nefu1020220084/anaconda3/bin/python',str(R/'code/bib_bootstrap.py'),str(R/'code/baseline_bootstrap.json')],check=True,cwd=R)
(R/'audits/BASELINES_STAGED_AND_BOOTSTRAPPED.json').write_text(json.dumps({'models':list(sources)}))
