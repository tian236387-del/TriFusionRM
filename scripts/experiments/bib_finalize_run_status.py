import csv,hashlib,json,platform,subprocess,time
from pathlib import Path
R=Path('/home/nefu1020220084/summer/StructDualRM/BIB_strengthening_20261028')
registry=json.loads((R/'data/DATA_REGISTRY.json').read_text())
registry.update(authoritative_root=str(R/'data'),policy='All new runs use centralized versioned inputs. frozen_v1 remains historical and immutable; candidate_v2 is not an approved benchmark until QC, feature completion and freeze markers exist.',m6Am_candidate={'GEO':'GSE233655','DOI':'10.7554/eLife.104139','qc':str(R/'data/m6Am_candidates/CROWNseq_m6Am_qc_audit.json')})
(R/'data/DATA_REGISTRY.json').write_text(json.dumps(registry,indent=2))
files=[]
for p in (R/'data/m6Am_candidates/local_acquired').glob('*'):
    if not p.is_file():continue
    h=hashlib.sha256(p.read_bytes()).hexdigest()
    files.append({'path':str(p),'bytes':p.stat().st_size,'sha256':h,'status':'DOWNLOADED_LOCALLY_THEN_UPLOADED','note':'Public internet DNS unavailable on server; original source retained in GEO/source manifests.'})
(R/'data/m6Am_candidates/acquired_files_verified.json').write_text(json.dumps(files,indent=2))
commands=[('/home/nefu1020220084/anaconda3/envs/RNAfold/bin/python','RNAfold'),('/home/nefu1020220084/anaconda3/bin/python','base_statistics')]
for py,name in commands:
    result=subprocess.run([py,'-m','pip','freeze'],capture_output=True,text=True)
    (R/f'environments/{name}_pip_freeze.txt').write_text(result.stdout)
readme='''# StructDualRM BIB strengthening workspace

Use data/DATA_REGISTRY.json as the input registry. Do not mix frozen versions.

- data/frozen_v1: historical benchmark used by the existing checkpoints.
- data/m6Am_candidates: acquired source data, PCIF1-dependent QC, exclusions.
- data/candidate_v2: new benchmark under clustering/feature QC; not final results.
- data/frozen_v2: created only after support and feature checks pass.
- data/reference: hg38, frozen top-30 eCLIP peaks and pretrained embeddings.
- results/unified_all_runs_per_class_metrics.tsv: each class for available runs.
- results/primary_seed42_bootstrap_2000: final no-pair versus sequence-only and reference.
- results/final_vs_baselines_bootstrap_2000: matched test-site baseline comparisons.
- audits/*queue_status.json: actual running/waiting/failed training states.
- logs/: training and statistics logs. COMPLETED markers establish completion.

GPU continuation: initial queue -> distance-matched motif inference (3 seeds)
and missing MultiRM/TransRNAm seeds -> V2 final/sequence and baseline retraining.
Statistics watchers compute paired tests when completed prediction files exist.
No manuscript files are modified by this workspace.

Limitations still needing assessment: source/cap-position bias in m6Am;
near-duplicate screening; exact identity of the archived external Model A;
V2 generalization and EvoRMD multiseed coverage. No acceptance guarantee.
'''
(R/'README.md').write_text(readme)
subprocess.run(['/home/nefu1020220084/anaconda3/bin/python',str(R/'code/bib_progress_snapshot.py')],check=True)
