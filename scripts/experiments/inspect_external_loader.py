from pathlib import Path
import torch
root=Path('/home/nefu1020220084/summer')
a=torch.load(root/'external_stage3_narmbench_20261017/input/checkpoints/model_A.pt',map_location='cpu',weights_only=False)
b=torch.load(root/'StructDualRM/BIB_strengthening_20261028/models/sequence_only/seed42/model.pt',map_location='cpu',weights_only=False)
for key in ['epoch','best_epoch','best_score','variant']: print(key,a.get(key))
best=a['best_state_dict'];current=b['model_state_dict']
print('BEST_STATE_MATCH',set(best)==set(current) and all(torch.equal(best[k],current[k]) for k in best))
for base in [root/'external_stage3_narmbench_20261017',root/'StructDualRM/BIB_strengthening_20261028/code']:
    for p in base.rglob('*.py'):
        if 'site-packages' in str(p):continue
        text=p.read_text(errors='replace')
        if any(x in text for x in ['external_A_predictions','model_A.pt','best_state_dict']):
            print('FILE',p)
            for i,line in enumerate(text.splitlines()):
                if any(x in line for x in ['external_A_predictions','model_A.pt','best_state_dict','load_state_dict','torch.load']):print(i+1,line[:250])
