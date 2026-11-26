from pathlib import Path
import csv,json,collections
r=Path('/home/nefu1020220084/summer')
base=r/'m1A_m6A/rmbase12_work/datasets'
original=base/'rmbase_multirm12_hg38_41nt_all_sites_cdhit80_strict_splits.tsv'
final=base/'rmbase_multirm12_hg38_41_101_201nt_cdhit80_common.tsv'
central=r/'StructDualRM/BIB_strengthening_20261028/data/frozen_v1/master.tsv'
def mapping(p):
    result={};counts=collections.Counter()
    with p.open() as f:
        for row in csv.DictReader(f,delimiter='\t'):
            result[row['site_id']]=(row['cdhit80_cluster_id'],row['split_cdhit80_cluster'])
            counts[row['split_cdhit80_cluster']]+=1
    return result,dict(counts)
a,ac=mapping(original);b,bc=mapping(final);c,cc=mapping(central)
mismatches=sum(a.get(k)!=v for k,v in b.items())
report={'original_41nt_rows':len(a),'final_common_rows':len(b),'final_counts':bc,'final_vs_41nt_assignment_mismatches':mismatches,'central_master_assignments_identical':b==c,'central_counts':cc,'paths':{'original':str(original),'final':str(final),'central':str(central)}}
assert mismatches==0 and b==c
for model in ['sequence_only','final_no_pair','reference_full']:
    path=r/f'StructDualRM/BIB_strengthening_20261028/models/{model}/seed42/training_config.json'
    cfg=json.loads(path.read_text())
    report[model]={k:cfg.get(k) for k in ['dataset','split','multi_length_columns']}
out=r/'StructDualRM/BIB_strengthening_20261028/audits/split_lineage_verified_20261030.json'
out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
