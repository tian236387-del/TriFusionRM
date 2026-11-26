"""Finish the matched-RBP experiment after the detached training completes."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from nov04_followup_prepare import R,O,read,write,point_metrics


def status(state,**kwargs):
    path=O/'matched_rbp/postprocess_status.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(dict(state=state,time=time.time(),**kwargs),indent=2))


def main(pid):
    output=R/'models/matched_rbp_shuffle_gene_validated/seed42'
    marker=output/'TRAINING_COMPLETED.json'
    status('WAITING_FOR_TRAINING',training_pid=pid,model_directory=str(output))
    while not marker.exists():
        try:os.kill(pid,0)
        except ProcessLookupError:
            status('TRAINING_FAILED_OR_INTERRUPTED',training_pid=pid,log=str(R/'logs/nov04_matched_rbp_gene_validated_seed42.log'))
            raise RuntimeError('Training exited without completion marker')
        time.sleep(30)
    import shutil
    audit=json.loads((output/'matched_rbp_shuffle_audit.json').read_text())
    donor=[r for r in read(O/'matched_rbp/test_donor_mapping.tsv') if r['exchangeable']=='True']
    eligible={r['site_id'] for r in donor}
    test_audit=next(r for r in audit['splits'] if r['split']=='test')
    assert len(eligible)==test_audit['exchangeable_sites']
    references={'authentic_RBP':R/'models/final_no_pair/seed42/test_predictions.tsv','matched_shuffled_RBP':output/'test_predictions.tsv'}
    # The sequence-only arm controls for model drift beyond RBP correspondence.
    references['sequence_only']=R/'models/sequence_only/seed42/test_predictions.tsv'
    for subset,selected in [('all_test',None),('exchangeable_test_only',eligible)]:
        dest=O/'matched_rbp'/subset
        paths={}
        all_metrics=[]
        for name,source in references.items():
            rows=[r for r in read(source) if selected is None or r['site_id'] in selected]
            target=dest/(name+'_test_predictions.tsv')
            write(target,rows)
            paths[name]=str(target)
            all_metrics.extend(point_metrics(target,name,subset,dest/(name+'_per_class_metrics.tsv')))
        write(dest/'all_models_per_class_metrics.tsv',all_metrics)
        dataset=R/'data/frozen_v1/master.tsv'
        if selected is not None:
            dataset=dest/'test_master.tsv'
            write(dataset,[r for r in read(R/'data/frozen_v1/master.tsv') if r['site_id'] in selected])
        spec=dict(output=str(dest/'cluster_bootstrap_2000'),dataset=str(dataset),models=paths,
                  comparisons=[['authentic_minus_gene_matched_shuffle','authentic_RBP','matched_shuffled_RBP'],
                               ['authentic_minus_sequence_only','authentic_RBP','sequence_only'],
                               ['gene_matched_shuffle_minus_sequence_only','matched_shuffled_RBP','sequence_only']])
        spec_path=dest/'bootstrap_spec.json'
        spec_path.write_text(json.dumps(spec,indent=2))
        status('BOOTSTRAPPING',subset=subset,replicates=2000)
        subprocess.run([sys.executable,str(R/'code/bib_bootstrap.py'),str(spec_path)],check=True)
    for filename in ('matched_rbp_shuffle_audit.json','training_config.json','resolved_requested_config.json','launch_environment.json','TRAINING_COMPLETED.json','test_per_class_metrics.tsv','metrics.json'):
        source=output/filename
        if source.exists():shutil.copy2(source,O/'matched_rbp'/filename)
    status('COMPLETED',replicates=2000,training_output=str(output),subsets=['all_test','exchangeable_test_only'])
    subprocess.run([sys.executable,str(R/'code/nov04_package_followup.py')],check=True)
    print('MATCHED_RBP_POSTPROCESS_COMPLETED',flush=True)


if __name__=='__main__':
    try:main(int(sys.argv[1]))
    except Exception as error:
        status('FAILED',error=repr(error))
        raise
