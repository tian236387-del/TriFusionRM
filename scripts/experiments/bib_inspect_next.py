import json
from pathlib import Path
R=Path('/home/nefu1020220084/summer/StructDualRM/BIB_strengthening_20261028')
obj=json.loads((R/'audits/baseline_discovery.json').read_text())
print('BASELINE_TYPE',type(obj).__name__)
rows=obj if isinstance(obj,list) else obj.get('runs',[])
for x in rows:
    t=x.get('test',{})
    print(json.dumps({k:x.get(k) for k in ['path','dataset','split','config_exists','predictions']}),t.get('macro_AUPR_ovr'))
for name in ['build_rbp_positional_features.py','train_motif_struct_fusion_v4.py','run_motif_directed_perturbation.py']:
    lines=(R/'code'/name).read_text().splitlines()
    print('CODE',name)
    for i,line in enumerate(lines):
        if any(s in line for s in ['add_argument(', 'def choose','def random','distance','shuffle_structure','shuffle_rbp']):
            if name!='train_motif_struct_fusion_v4.py' or any(s in line for s in ['shuffle','pair','split','sequence','structure']):
                print(i+1,line[:220])
print('MASTER_HEADER',(R/'data/frozen_v1/master.tsv').open().readline())
print('RBP_JSON',(R/'data/frozen_v1/rbp_positional.tsv.json').read_text()[:3500])
