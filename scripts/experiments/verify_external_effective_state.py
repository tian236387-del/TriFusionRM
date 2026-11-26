from pathlib import Path
import hashlib,json
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score

r=Path('/home/nefu1020220084/summer/StructDualRM/BIB_strengthening_20261028')
e=Path('/home/nefu1020220084/summer/external_stage3_narmbench_20261017')
o=r/'results/external_verified'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((e/'outputs/external_stage3_manifest.json').read_text())
audit={'status':'EFFECTIVE_INFERENCE_STATES_VERIFIED','models':{},'prediction_checks':{}}
frames={}
for tag,archive,current,state_key,metric_tag in [('Model_A','model_A.pt','sequence_only','best_state_dict','A'),('D-NP','model_DNP.pt','final_no_pair','model_state_dict','DNP')]:
    ap=e/'input/checkpoints'/archive; cp=r/f'models/{current}/seed42/model.pt'
    a=torch.load(ap,map_location='cpu',weights_only=False);b=torch.load(cp,map_location='cpu',weights_only=False)
    x,y=a[state_key],b['model_state_dict']
    identical=set(x)==set(y) and all(torch.equal(x[k],y[k]) for k in x)
    assert identical
    assert a['config']==b['config']
    metrics=json.loads((e/f'outputs/external_{metric_tag}_metrics_unrestricted12.json').read_text())
    assert metrics['state_source']==state_key
    assert metrics['checkpoint_sha256']==sha(ap)
    assert sha(ap)==manifest['checkpoints'][tag]['sha256']
    pp=e/f'outputs/external_{metric_tag}_predictions.tsv'
    assert sha(pp)==manifest['predictions'][tag]['sha256']
    df=pd.read_csv(pp,sep='\t');frames[tag]=df
    probs=df[['p_'+c for c in metrics['classes']]].values
    assert np.isfinite(probs).all() and np.allclose(probs.sum(1),1,atol=1e-5)
    vals=[]
    for c,item in metrics['unrestricted12']['per_class'].items():
        value=average_precision_score(df.external_mod_type==c,df['p_'+item['model_class']])
        assert abs(value-item['AUPR'])<1e-10
        vals.append(value)
    audit['models'][tag]={'archived':str(ap),'current':str(cp),'loaded_state_key':state_key,'exact_tensor_identity':identical,'configuration_identity':True,'tensor_count':len(x),'archived_file_sha256':sha(ap),'current_file_sha256':sha(cp),'metric_state_source_verified':True,'prediction_sha256':sha(pp),'recomputed_macro_AUPR':float(np.mean(vals))}
    audit['prediction_checks'][tag]={'rows':len(df),'site_id_unique':bool(df.site_id.is_unique),'metrics_recomputed':True}
a,b=frames['Model_A'],frames['D-NP']
assert a[['site_id','external_mod_type','external_cluster_id']].equals(b[['site_id','external_mod_type','external_cluster_id']])
audit['matched_site_label_cluster_order']=True
audit['explanation']='The archive A file is a training-resume container; best_state_dict, explicitly selected by the inference loader and recorded in metrics, is tensor-identical to the current sequence-only inference checkpoint. Its last-epoch model_state_dict differs and was not used for external predictions.'
(o/'external_effective_checkpoint_identity_audit.json').write_text(json.dumps(audit,indent=2))
old=json.loads((o/'external_identity_and_overlap_audit.json').read_text())
old['status']=audit['status'];old['external_Model_A_matches_current_sequence_only_seed42']=True
old['comparison_policy']='Effective inference state identities verified; matched external comparisons may be reported with their actual uncertainty and direction. This verifies identity, not a superiority claim.'
old['effective_state_audit']=audit
(o/'external_identity_and_overlap_audit.json').write_text(json.dumps(old,indent=2))
print(json.dumps(audit,indent=2))
