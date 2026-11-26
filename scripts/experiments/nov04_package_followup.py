"""Package current follow-up results without touching any manuscript files."""
import hashlib
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
from nov04_followup_prepare import R,O,CLASSES,read,write


def main():
    matched_status=O/'matched_rbp/postprocess_status.json'
    if matched_status.exists() and json.loads(matched_status.read_text())['state']=='COMPLETED':
        subprocess.run([sys.executable,str(R/'code/nov04_verify_matched_rbp.py')],check=True)
        subprocess.run(['/home/nefu1020220084/anaconda3/envs/RNAfold/bin/python',str(R/'code/nov04_verify_matched_checkpoint.py')],check=True)
    out=O/'long_context/cluster_bootstrap_2000'
    spec=json.loads((out/'COMPLETED.json').read_text())
    distributions=np.load(out/'bootstrap_distributions.npz')
    points=list(read(O/'long_context/all_models_per_class_metrics.tsv'))
    summaries=list(read(out/'bootstrap_macro.tsv'))
    names=list(spec['models'])
    support={r['modification']:int(r['support']) for r in points if r['model']=='final_no_pair'}
    js=[i for i,c in enumerate(CLASSES) if support[c]>0]
    for row in summaries:
        if row['scope']!='standard' or row['class_subset']!='all_evaluable':continue
        ai=names.index(row['model_a']);bi=names.index(row['model_b'])
        values=np.mean([distributions[f'standard__{j}__{ai}']-distributions[f'standard__{j}__{bi}'] for j in js],axis=0)
        assert len(values)==2000
        assert np.allclose(np.quantile(values,[.025,.975]),[float(row['ci95_low']),float(row['ci95_high'])],atol=1e-7)
        for model,field in [(row['model_a'],'AP_a'),(row['model_b'],'AP_b')]:
            observed=np.mean([float(r['AUPR']) for r in points if r['model']==model and r['AUPR']!=''])
            assert abs(observed-float(row[field]))<1e-10
    (O/'long_context/VERIFIED.json').write_text(json.dumps(dict(point_metrics_recomputed_with_sklearn=True,macro_CI_reproduced_from_saved_distributions=True,replicates=2000,
        p_formula='min(1,2*(1+min(count(delta<=0),count(delta>=0)))/(2000+1))',
        BH_family='Four macro comparisons within each scope/class subset; per-class tests corrected across four comparisons and evaluable classes within each scope',
        limitation='Test subset screened against frozen StructDualRM training and validation sets. Archived baseline training eligibility differs; this is not a full same-information architecture benchmark.'),indent=2))
    screen_logs=O/'long_context/screen_command_logs'
    screen_logs.mkdir(exist_ok=True)
    for path in (R/'audits/near_sequence_audit').glob('*.log'):
        shutil.copy2(path,screen_logs/path.name)
    shutil.copy2(R/'results/external_verified/external_effective_checkpoint_identity_audit.json',O/'external_A_centered/external_effective_checkpoint_identity_audit.json')
    baseline_audit_path=O/'baseline_fairness/baseline_original_training_eligibility_audit.json'
    baseline_audit=json.loads(baseline_audit_path.read_text())
    for row in baseline_audit:
        if row['model']=='evormd_safe':
            row['loss']='softmax cross-entropy; class_weight_power=0 gives equal class weights'
        shutil.copy2(row['original_config'],O/'baseline_fairness'/(row['model']+'_original_training_config.json'))
    baseline_audit_path.write_text(json.dumps(baseline_audit,indent=2))
    # Keep available additional benchmark evidence separate from the frozen-v1 experiment.
    v2=O/'existing_stricter_v2'
    v2.mkdir(exist_ok=True)
    for filename in ['unified_all_runs_metrics.tsv','unified_all_runs_per_class_metrics.tsv']:
        rows=[r for r in read(R/'results'/filename) if r['dataset_version']=='frozen_v2']
        write(v2/filename,rows)
    shutil.copy2(R/'data/frozen_v2/V2_CLUSTER_AUDIT.json',v2/'V2_CLUSTER_AUDIT.json')
    (v2/'SCOPE.json').write_text(json.dumps(dict(status='PRE_EXISTING_RESULTS_COLLECTED_NOT_NEW_TRAINING',
        partition='Union of 41-nt and 201-nt CD-HIT-est representative clusters; not a 41/101/201 exhaustive similarity-free partition.',
        dataset_changes='Adds independent m6Am source records and removes conflicting coordinates; differences from frozen_v1 cannot be attributed solely to sequence splitting.',
        use='Separate retrospective sensitivity benchmark; do not replace frozen-v1 checkpoint metrics or merge their macro averages.'),indent=2))
    invalid=R/'models/matched_rbp_shuffle/seed42'
    if invalid.exists():
        (invalid/'DO_NOT_USE.json').write_text(json.dumps(dict(status='ABORTED_PRELIMINARY_RUN',reason='Intergenic was a metadata placeholder and must not form a within-gene stratum.',replacement=str(R/'models/matched_rbp_shuffle_gene_validated/seed42')),indent=2))
    code=O/'reproduction_code'
    code.mkdir(exist_ok=True)
    for filename in ['nov04_followup_prepare.py','nov04_external_a_centered.py','nov04_verify_outputs.py','nov04_verify_matched_rbp.py','nov04_verify_matched_checkpoint.py','nov04_baseline_eligibility_audit.py','nov04_train_matched_rbp.py','nov04_matched_rbp_job.json','nov04_matched_rbp_postprocess.py','nov04_package_followup.py','bib_bootstrap.py','cluster_bootstrap_multimodal_parallel.py']:
        shutil.copy2(R/'code'/filename,code/filename)
    status_path=O/'matched_rbp/postprocess_status.json'
    status=json.loads(status_path.read_text()) if status_path.exists() else {'state':'NOT_STARTED'}
    (O/'followup_status.json').write_text(json.dumps(dict(time=time.time(),long_context_screening='COMPLETED_PRE_EXISTING_SCREEN_VERIFIED',
        screened_test_sensitivity='COMPLETED_2000_PAIRED_CLUSTER_BOOTSTRAP',external_A_centered='COMPLETED_2000_PAIRED_CLUSTER_BOOTSTRAP',
        baseline_eligibility_audit='COMPLETED',equal_information_equal_training_architecture_retraining='NOT_PERFORMED',matched_RBP=status,
        manuscript_and_supplement='NOT_MODIFIED'),indent=2))
    manifest=[]
    for path in sorted(O.rglob('*')):
        if not path.is_file() or path.name in ('file_manifest.tsv','reviewer_followup_20261104.zip'):continue
        digest=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b''):digest.update(chunk)
        manifest.append(dict(path=str(path.relative_to(O)),bytes=path.stat().st_size,sha256=digest.hexdigest()))
    write(O/'file_manifest.tsv',manifest)
    # The portable archive contains summary tables, audits and code; large per-site inputs remain on the server.
    with zipfile.ZipFile(O/'reviewer_followup_20261104.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in O.rglob('*'):
            if not path.is_file() or path.suffix=='.zip':continue
            relative=str(path.relative_to(O))
            if 'predictions' in relative or path.name.endswith('_master.tsv') or 'donor_mapping' in relative or path.name=='test_screen_flags.tsv' or path.name=='training_eligibility_differences.tsv':continue
            archive.write(path,relative)
    print(json.dumps(dict(directory=str(O),archive=str(O/'reviewer_followup_20261104.zip'),files=len(manifest),matched_RBP_status=status),indent=2),flush=True)


if __name__=='__main__':main()
