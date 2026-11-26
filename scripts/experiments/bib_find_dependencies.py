import json,os
from pathlib import Path
S=Path('/home/nefu1020220084/summer/m1A_m6A');R=S.parent/'StructDualRM/BIB_strengthening_20261028'
for base in [S/'scripts',S.parent/'StructDualRM/scripts']:
    if base.exists():
        for p in base.rglob('*.py'):
            if any(x in p.name for x in ['rnafold','perturb','mutagen','official','external']):print('SCRIPT',p)
manifest=S/'rmbase11_full_work/rbp_encode_eclip_all/encode_eclip_top30_manifest.tsv'
print('MANIFEST',manifest,manifest.exists())
if manifest.exists():print(manifest.read_text()[:1700])
print('RNAFOLD_HEADER',(R/'data/frozen_v1/rnafold.tsv').open().readline())
for x in ['multirm_official','transrnam_official']:
    p=S/'rmbase12_work/official_reproduction_benchmark/multirm12_cdhit80_strict_split'/x
    print('CONFIG',x,(p/'training_config.json').read_text()[:6000])
print('CPU',os.cpu_count())
