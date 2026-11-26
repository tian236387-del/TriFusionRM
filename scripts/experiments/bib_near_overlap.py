"""Directional CD-HIT-est-2d cross-partition screening at fixed thresholds."""
import csv,json,subprocess,time
from pathlib import Path
R=Path('/home/nefu1020220084/summer/StructDualRM/BIB_strengthening_20261028');O=R/'audits/near_sequence_audit';O.mkdir(exist_ok=True)
binary='/home/nefu1020220084/anaconda3/envs/RNAfold/bin/cd-hit-est-2d'
rows=[]
for n in [101,201]:
    files={s:O/f'{s}_{n}.fasta' for s in ['train','valid','test']}
    handles={s:p.open('w') for s,p in files.items()}
    f=iter((R/f'data/frozen_v1/unique_{n}nt.fasta').open())
    for header in f:
        seq=next(f);split=header[1:].split('__')[0];handles[split].write(header+seq)
    for h in handles.values():h.close()
    for ref,query in [('train','valid'),('train','test'),('valid','test')]:
        out=O/f'{ref}_vs_{query}_{n}'
        cmd=[binary,'-i',str(files[ref]),'-i2',str(files[query]),'-o',str(out),'-c','.8','-n','5','-G','1','-aS','.8','-aL','.8','-r','0','-T','8','-M','16000','-d','0']
        with Path(str(out)+'.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        total=sum(line.startswith('>') for line in files[query].open())
        nonmatching=sum(line.startswith('>') for line in out.open())
        rows.append({'length':n,'reference_split':ref,'query_split':query,'query_unique_sequences':total,'query_with_detected_cross_split_match':total-nonmatching,'identity':.8,'short_coverage':.8,'long_coverage':.8,'reverse_complement':False,'search':'CD-HIT-est-2d greedy threshold screening; matches establish contamination, absence is not an exhaustive all-pairs proof'})
        with (R/'audits/cross_split_sequence_similarity.tsv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)
        print(json.dumps(rows[-1]),flush=True)
(O/'COMPLETED.json').write_text(json.dumps({'time':time.time(),'comparisons':len(rows)}))
