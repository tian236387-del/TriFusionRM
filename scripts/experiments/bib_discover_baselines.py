import json,os,shutil
from pathlib import Path
R=Path('/home/nefu1020220084/summer/StructDualRM/BIB_strengthening_20261028')
S=Path('/home/nefu1020220084/summer/m1A_m6A')
found=[]
for rel in ['rmbase12_work','rmbase11_full_work','StructDualRM_submission_20260804_final']:
    root=S/rel
    for directory,dirs,files in os.walk(root):
        dirs[:]=[x for x in dirs if x not in ['.git','__pycache__','node_modules','baseline_repos','official_repos','checkpoints']]
        if 'metrics.json' not in files:continue
        path=Path(directory)
        if not any(x in str(path).lower() for x in ['baseline','multirm_official','transrnam','evormd']):continue
        try:
            m=json.loads((path/'metrics.json').read_text())
            found.append({'path':str(path),'dataset':m.get('dataset'),'split':m.get('split'),'test':m.get('test'),'config_exists':(path/'training_config.json').exists(),'predictions':(path/'test_predictions.tsv').exists()})
        except Exception:pass
(R/'audits/baseline_discovery.json').write_text(json.dumps(found,indent=2))
for row in found:print(json.dumps(row),flush=True)
