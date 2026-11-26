"""Stage frozen eCLIP sources and build features for newly acquired sites only."""
import csv,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
R=Path('/home/nefu1020220084/summer/StructDualRM/BIB_strengthening_20261028');S=R.parent.parent/'m1A_m6A';O=R/'data/candidate_v2';O.mkdir(exist_ok=True)
E=R/'data/reference/eclip_top30';E.mkdir(exist_ok=True)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()
manifest=S/'rmbase11_full_work/rbp_encode_eclip_all/encode_eclip_top30_manifest.tsv'
rows=list(csv.DictReader(manifest.open(),delimiter='\t'));records=[]
for row in rows:
    p=S/row['download_path'];assert p.is_file(),p
    dst=E/p.name
    if not dst.exists():shutil.copy2(p,dst)
    row['download_path']=str(dst);records.append({'file':str(dst),'source':str(p),'sha256':sha(dst)})
with (E/'manifest.tsv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');writer.writeheader();writer.writerows(rows)
(E/'file_hashes.json').write_text(json.dumps(records,indent=2))
for name in ['run_rmbase11_rnafold.py','train_rmbase11_official_reproduction_benchmark.py']:
    shutil.copy2(S/'scripts'/name,R/'code'/name)
embedding=S/'external/MultiRM-master/Embeddings/embeddings_12RM.pkl'
if embedding.exists():shutil.copy2(embedding,R/'data/reference/embeddings_12RM.pkl')
while not (O/'new_m6Am_input.tsv').exists():time.sleep(15)
py='/home/nefu1020220084/anaconda3/envs/RNAfold/bin/python'
commands=[
 [py,str(R/'code/run_rmbase11_rnafold.py'),'--dataset',str(O/'new_m6Am_input.tsv'),'--out',str(O/'m6Am_rnafold.tsv'),'--rnafold-bin','/home/nefu1020220084/anaconda3/envs/RNAfold/bin/RNAfold','--resume'],
 [py,str(R/'code/build_rbp_positional_features.py'),'--dataset',str(O/'new_m6Am_input.tsv'),'--manifest',str(E/'manifest.tsv'),'--out',str(O/'m6Am_rbp_positional.tsv')]]
for i,cmd in enumerate(commands):
    with (R/'logs'/f'v2_feature_{i}.txt').open('w') as f:subprocess.run(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
for name,extra in [('rnafold.tsv','m6Am_rnafold.tsv'),('rbp_positional.tsv','m6Am_rbp_positional.tsv')]:
    with (O/name).open('wb') as out:
        with (R/'data/frozen_v1'/name).open('rb') as first:
            header=first.readline().strip();out.write(header+b'\n');shutil.copyfileobj(first,out)
        with (O/extra).open('rb') as second:
            assert second.readline().strip()==header,(name,'column order mismatch')
            shutil.copyfileobj(second,out)
meta=json.loads((R/'data/frozen_v1/rbp_positional.tsv.json').read_text())
meta.update(dataset=str(O/'master.tsv'),manifest=str(E/'manifest.tsv'),output=str(O/'rbp_positional.tsv'),site_count=None,feature_policy='Frozen original top30, six genomic bins; original features reused and new sites processed identically.')
(O/'rbp_positional.tsv.json').write_text(json.dumps(meta,indent=2))
(O/'FEATURES_COMPLETED.json').write_text(json.dumps({'time':time.time(),'commands':commands,'hashes':{n:sha(O/n) for n in ['rnafold.tsv','rbp_positional.tsv']},'extra_feature_rows':'Historical excluded rows may remain in feature lookup, but are not used as samples.'},indent=2))
print('FEATURES_COMPLETED',flush=True)
