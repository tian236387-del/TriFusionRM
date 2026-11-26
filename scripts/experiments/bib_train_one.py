"""Run the archived architecture with centralized inputs and explicit overrides."""
import argparse,json,os,sys,time
from pathlib import Path
R=Path('/home/nefu1020220084/summer/StructDualRM/BIB_strengthening_20261028')
print('IMPORTING_TRAINER',time.time(),flush=True)
sys.path.insert(0,str(R/'code'))
import torch
import train_motif_struct_fusion_v4 as trainer
parser=argparse.ArgumentParser();parser.add_argument('job');a=parser.parse_args()
job=json.loads(Path(a.job).read_text())
config=json.loads((R/job['template']).read_text())
args=trainer.build_parser().parse_args([])
unknown=[]
for key,value in config.items():
    if hasattr(args,key):setattr(args,key,value)
    else:unknown.append(key)
for key,value in job['overrides'].items():
    assert hasattr(args,key),key
    setattr(args,key,value)
args.dataset=str(R/'data/frozen_v1/master.tsv')
args.rnafold=str(R/'data/frozen_v1/rnafold.tsv')
args.rbp_positional_features=str(R/'data/frozen_v1/rbp_positional.tsv')
args.rbp_features=''
if job.get('dataset_version')=='frozen_v2':
    args.dataset=str(R/'data/frozen_v2/master.tsv')
    args.rnafold=str(R/'data/frozen_v2/rnafold.tsv')
    args.rbp_positional_features=str(R/'data/frozen_v2/rbp_positional.tsv')
    args.split='split_v2_cluster'
args.out_dir=str(R/job['out_dir'])
args.resume=False
torch.set_num_threads(4)
out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
(out/'resolved_requested_config.json').write_text(json.dumps(vars(args),indent=2))
(out/'launch_environment.json').write_text(json.dumps({'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'unknown_legacy_keys':unknown,'time':time.time()},indent=2))
assert torch.cuda.is_available(),'GPU required for planned full runs'
print('TRAINING_STARTED',job['name'],flush=True)
trainer.run(args)
(out/'TRAINING_COMPLETED.json').write_text(json.dumps({'job':job,'time':time.time()}))

