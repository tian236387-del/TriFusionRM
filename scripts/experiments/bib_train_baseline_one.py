import argparse,json,sys,time
from pathlib import Path
R=Path('/home/nefu1020220084/summer/StructDualRM/BIB_strengthening_20261028');sys.path.insert(0,str(R/'code'))
import torch
import train_rmbase11_official_reproduction_benchmark as t
p=argparse.ArgumentParser();p.add_argument('job');a=p.parse_args();job=json.loads(Path(a.job).read_text())
config=json.loads((R/job['template']).read_text())
out=R/job['out_dir'];out.mkdir(parents=True,exist_ok=True)
args=t.build_parser().parse_args(['--model',config['model'],'--out-dir',str(out)])
for k,v in config.items():
    if hasattr(args,k):setattr(args,k,v)
args.out_dir=str(out);args.seed=job['seed'];args.resume=False;args.cpu=False
version=job.get('dataset_version','frozen_v1')
args.dataset=str(R/f'data/{version}/master.tsv');args.split='split_v2_cluster' if version=='frozen_v2' else 'split_cdhit80_cluster'
args.embedding_path=str(R/'data/reference/embeddings_12RM.pkl')
torch.set_num_threads(4);assert torch.cuda.is_available()
(out/'resolved_requested_config.json').write_text(json.dumps(vars(args),indent=2))
t.run(args)
(out/'TRAINING_COMPLETED.json').write_text(json.dumps({'job':job,'time':time.time()}))
