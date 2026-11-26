from pathlib import Path
import json
import hashlib
import torch

root=Path('/home/nefu1020220084/summer/StructDualRM/BIB_strengthening_20261028')
external=Path('/home/nefu1020220084/summer/external_stage3_narmbench_20261017/input/checkpoints')
out=root/'results/external_verified'
out.mkdir(exist_ok=True)

def state(path):
    obj=torch.load(path,map_location='cpu',weights_only=False)
    print(str(path),type(obj),list(obj)[:15],flush=True)
    if isinstance(obj,dict):
        for key in ['model_state_dict','state_dict','model']:
            if key in obj and isinstance(obj[key],dict):return obj[key]
    return obj

audit={}
for tag,archived,current in [('sequence_only',external/'model_A.pt',root/'models/sequence_only/seed42/model.pt'),('final_no_pair',external/'model_DNP.pt',root/'models/final_no_pair/seed42/model.pt')]:
    a,b=state(archived),state(current)
    missing=sorted(set(a)^set(b))
    different=[]
    for key in sorted(set(a)&set(b)):
        x,y=a[key],b[key]
        if torch.is_tensor(x) and torch.is_tensor(y):
            if x.shape!=y.shape or x.dtype!=y.dtype or not torch.equal(x,y):
                different.append({'key':key,'shape_a':list(x.shape),'shape_b':list(y.shape),'max_abs_difference':float((x.double()-y.double()).abs().max()) if x.shape==y.shape else None})
        elif x!=y:different.append({'key':key,'non_tensor_difference':True})
    audit[tag]={'archived':str(archived),'current':str(current),'archived_sha256':hashlib.sha256(archived.read_bytes()).hexdigest(),'current_sha256':hashlib.sha256(current.read_bytes()).hexdigest(),'key_count':len(a),'missing_keys':missing,'different_parameters':different,'exact_state_identity':not missing and not different}
(out/'checkpoint_state_identity_audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(audit,indent=2),flush=True)
