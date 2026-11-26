"""Reconstruct baseline eligibility from original archived configs and datasets."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

R=Path('/home/nefu1020220084/summer/StructDualRM/BIB_strengthening_20261028')
sys.path.insert(0,str(R/'code'))
import train_rmbase11_official_reproduction_benchmark as trainer
from nov04_followup_prepare import read,write,CLASSES,O


def id_hash(rows):
    digest=hashlib.sha256()
    for row in sorted(rows,key=lambda r:r['site_id']):
        digest.update((row['site_id']+'\t'+row['mod_type']+'\n').encode())
    return digest.hexdigest()


def main():
    current={r['site_id']:(r['split_cdhit80_cluster'],r['mod_type'],r['sequence_41nt'].upper().replace('U','T')) for r in read(R/'data/frozen_v1/master.tsv')}
    current_train={k for k,v in current.items() if v[0]=='train'}
    current_test={k for k,v in current.items() if v[0]=='test'}
    archived=json.loads((R/'audits/baseline_reproduction_audit.json').read_text())
    results=[]
    per_class=[]
    differences=[]
    for entry in archived:
        original_config=Path(entry['source'])/'training_config.json'
        config=json.loads(original_config.read_text())
        path=Path(config['dataset'])
        if not path.is_absolute():path=Path('/home/nefu1020220084/summer/m1A_m6A')/path
        assert path.exists(),str(path)
        rows=[{k:r.get(k,'') for k in ('site_id','mod_type','sequence_41nt',config['split'])} for r in read(path)]
        original_train=[r for r in rows if r[config['split']]=='train']
        actual_train=original_train
        downsampling={'enabled':False}
        if config['model']=='evormd_rnafm_official_core' and config.get('downsample_train_classes','').strip():
            actual_train,downsampling=trainer.downsample_training_rows(original_train,config['downsample_train_classes'].split(','),config['downsample_alpha'],config['seed'])
        used={r['site_id'] for r in actual_train}
        baseline_only=used-current_train
        current_only=current_train-used
        leakage=used&current_test
        label_errors=[]
        split_errors=[]
        sequence_errors=[]
        for row in rows:
            sid=row['site_id']
            if sid not in current:continue
            target=current[sid]
            if target[0]!=row[config['split']]:split_errors.append(sid)
            if target[1]!=row['mod_type']:label_errors.append(sid)
            if target[2]!=row['sequence_41nt'].upper().replace('U','T'):sequence_errors.append(sid)
        for difference,ids in [('baseline_train_only',baseline_only),('frozen_train_only',current_only),('baseline_train_in_frozen_test',leakage)]:
            differences.extend(dict(model=entry['model'],difference=difference,site_id=sid) for sid in sorted(ids))
        counters=Counter(r['mod_type'] for r in actual_train)
        original=Counter(r['mod_type'] for r in original_train)
        for c in CLASSES:
            per_class.append(dict(model=entry['model'],modification=c,source_training_eligible=original[c],reconstructed_training_used=counters[c],frozen_training_sites=sum(v[0]=='train' and v[1]==c for v in current.values())))
        results.append(dict(model=entry['model'],original_config=str(original_config),original_dataset=str(path),
                            training_eligible=len(original_train),training_used_reconstructed=len(actual_train),frozen_training_sites=len(current_train),
                            baseline_train_only=len(baseline_only),frozen_train_only=len(current_only),baseline_train_in_frozen_test=len(leakage),
                            shared_site_split_disagreements=len(split_errors),shared_site_label_disagreements=len(label_errors),shared_41nt_sequence_disagreements=len(sequence_errors),
                            reconstructed_training_id_label_sha256=id_hash(actual_train),downsampling=downsampling,
                            input_length=config['input_length'],loss='effective-number weighted binary cross-entropy' if config['model'] in ('multirm_official','transrnam_official') else 'softmax cross-entropy (class_weight_power='+str(config.get('class_weight_power',0))+')',
                            limitation='Eligibility reconstructed from archived source/config; an explicit serialized training-ID list was not retained. Shared-test comparison is not an equal-input/equal-training architecture experiment.'))
    out=O/'baseline_fairness'
    out.mkdir(parents=True,exist_ok=True)
    (out/'baseline_original_training_eligibility_audit.json').write_text(json.dumps(results,indent=2))
    write(out/'baseline_per_class_training_eligibility.tsv',per_class)
    if differences:write(out/'training_eligibility_differences.tsv',differences)
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':main()
