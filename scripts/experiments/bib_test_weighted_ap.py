import json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score
import cluster_bootstrap_multimodal_parallel as m
rng=np.random.default_rng(20261028);worst=0;cases=0
for n in [5,17,200]:
    for tied in [False,True]:
        scores=rng.random(n).astype(np.float32)
        if tied:scores=np.round(scores,1)
        y=rng.integers(0,2,n);y[0]=1
        weights=rng.integers(0,5,(30,n));weights[:,0]=1
        m._WEIGHTS=weights;m._BLOCK_SIZE=7
        for indices in [np.arange(n),np.arange(0,n,2)]:
            result=m.weighted_ap(scores[indices],y[indices],indices)
            expected=[average_precision_score(y[indices],scores[indices],sample_weight=w[indices]) for w in weights]
            error=float(np.max(np.abs(result-expected)));worst=max(worst,error);cases+=len(result)
            assert error<2e-6,(n,tied,error)
out=Path('/home/nefu1020220084/summer/StructDualRM/BIB_strengthening_20261028/audits/weighted_ap_validation.json')
out.write_text(json.dumps({'status':'PASSED','weighted_comparisons':cases,'max_abs_error_vs_sklearn':worst,'cases':'ties, zero weights, whole sample and family subsets'}));print(out.read_text())
