"""Collect verified source files and real example inputs without modifying runs."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
import importlib.metadata

R=Path('/home/nefu1020220084/summer/StructDualRM/BIB_strengthening_20261028')
OUT=R/'exports/StructDualRM_final_no_pair_20261106'
OUT.mkdir(parents=True,exist_ok=True)
provenance=[]

def copy(source,relative):
    target=OUT/relative
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target)
    provenance.append({'path':relative,'source':str(source)})

def subset(source,target,ids):
    target.parent.mkdir(parents=True,exist_ok=True)
    seen=set()
    with source.open() as f,target.open('w') as out:
        header=f.readline()
        fields=header.rstrip('\r\n').split('\t')
        idx=fields.index('site_id')
        out.write(header)
        for line in f:
            sid=line.split('\t',idx+1)[idx].rstrip('\r\n')
            if sid in ids:
                assert sid not in seen,(source,sid)
                seen.add(sid)
                out.write(line)
                if seen==ids:break
    assert seen==ids,(source,len(seen),len(ids))
    print('Extracted',source.name,len(seen),flush=True)

for name in ['train_motif_struct_fusion_v4.py','train_motif_struct_fusion_v3.py','train_paired_gat_rbp_fusion.py',
             'export_motif_struct_fusion_v4_predictions.py','build_rbp_positional_features.py','run_rmbase11_rnafold.py']:
    copy(R/'code'/name,'src/'+name)
for name in ['model.pt','training_config.json','centralized_config.json','metrics.json','test_per_class_metrics.tsv']:
    copy(R/'models/final_no_pair/seed42'/name,'checkpoints/seed42/'+name)
copy(R/'models/final_no_pair/seed42/test_predictions.tsv','historical/test_predictions_seed42.tsv')
copy(R/'code/nov06_example_inference.py','examples/run_inference.py')
copy(R/'results/reviewer_followup_20261104/matched_rbp/checkpoint_and_training_identity_audit.json','audits/checkpoint_identity_audit.json')
copy(R/'audits/dataset_identity_audit.json','audits/dataset_identity_audit.json')
for source in (R/'data/frozen_v1/provenance').glob('*'):
    if source.is_file():copy(source,'preprocessing/provenance/'+source.name)

history=OUT/'historical/test_predictions_seed42.tsv'
counts={}
selected=[]
with history.open() as f:
    reader=csv.DictReader(f,delimiter='\t')
    fields=reader.fieldnames
    for row in reader:
        label=row['true_mod_type']
        if counts.get(label,0)<2:
            selected.append(row)
            counts[label]=counts.get(label,0)+1
ids={r['site_id'] for r in selected}
(OUT/'examples').mkdir(exist_ok=True)
with (OUT/'examples/historical_predictions.tsv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=fields,delimiter='\t',lineterminator='\n')
    writer.writeheader();writer.writerows(selected)
for original,target in [('master.tsv','test_sites.tsv'),('rnafold.tsv','rnafold.tsv'),('rbp_positional.tsv','rbp_positional.tsv')]:
    subset(R/'data/frozen_v1'/original,OUT/'examples'/target,ids)
copy(R/'data/frozen_v1/rbp_positional.tsv.json','examples/rbp_positional.tsv.json')
(OUT/'examples/selection.json').write_text(json.dumps({'rule':'First two archived test predictions per supported class; demonstration only, not a performance subset.','counts':counts,'source':str(history)},indent=2))
classes=['Am','Cm','Gm','Um','m1A','m5C','m5U','m6A','m6Am','m7G','Y','AtoI']
(OUT/'preprocessing/label_mapping.json').write_text(json.dumps({'index_to_label':dict(enumerate(classes)),'display_aliases':{'Y':'pseudouridine','AtoI':'I'},'base_tokens':{'unknown_or_padding':0,'A':1,'C':2,'G':3,'U_or_T':4}},indent=2))
versions={name:importlib.metadata.version(name) for name in ['torch','numpy','scipy','scikit-learn','pandas']}
(OUT/'environment.json').write_text(json.dumps({'python':sys.version,'verified_environment_packages':versions},indent=2))
(OUT/'requirements.txt').write_text('\n'.join(f'{k}=={v}' for k,v in versions.items())+'\n')
copy(R/'code/nov06_bundle_README.md','README.md')
subprocess.run([sys.executable,str(OUT/'examples/run_inference.py')],check=True)
(OUT/'source_provenance.json').write_text(json.dumps(provenance,indent=2))
manifest=[]
for path in sorted(OUT.rglob('*')):
    if path.is_file() and '__pycache__' not in str(path) and path.name!='SHA256SUMS.json':
        manifest.append({'path':str(path.relative_to(OUT)),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
(OUT/'SHA256SUMS.json').write_text(json.dumps(manifest,indent=2))
archive=OUT.with_suffix('.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for path in OUT.rglob('*'):
        if path.is_file() and '__pycache__' not in str(path):z.write(path,str(Path(OUT.name)/path.relative_to(OUT)))
print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()},indent=2),flush=True)
