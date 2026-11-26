"""Run the bundled final no-pair checkpoint on real archived test inputs."""
from pathlib import Path
import json
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import train_motif_struct_fusion_v4 as v4
import export_motif_struct_fusion_v4_predictions as exporter

def main():
    torch.set_num_threads(4)
    folder=ROOT/'checkpoints/seed42'
    payload=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
    args=json.loads((folder/'training_config.json').read_text())
    assert args['disable_pair_edges'] is True
    v4.configure_classes(payload['class_names'])
    cfg=exporter.build_config(payload)
    v4.configure_structure_inputs(','.join(cfg.struct_fields),','.join(cfg.node_feature_fields))
    v4.configure_pair_candidates(cfg.max_pair_candidates)
    rows=v4.read_tsv(ROOT/'examples/test_sites.tsv')
    folds=v4.load_rnafold(ROOT/'examples/rnafold.tsv')
    rbp,shape,_=exporter.load_selected_rbp_positional_features(str(ROOT/'examples/rbp_positional.tsv'),[r['site_id'] for r in rows])
    assert len(folds)==len(rows)==len(rbp)
    dataset=v4.RMBaseFusionMultiLengthPositionalDataset(
        rows,folds,cfg.input_length,rbp_by_id={},rbp_dim=0,
        struct_stats=payload['struct_stats'],rbp_stats=payload.get('rbp_stats'),
        shuffle_structure=False,shuffle_rbp=False,seed=42,
        sequence_column=args['sequence_column'],disable_pair_edges=True,
        multi_length_columns=tuple(cfg.multi_length_columns),multi_length_lengths=tuple(cfg.multi_length_lengths),
        rbp_pos_by_id=rbp,rbp_pos_shape=shape)
    assert (dataset.pair_partner_tensor == -1).all()
    model=v4.MotifStructFusionV4(cfg)
    model.load_state_dict(payload['model_state_dict'],strict=True)
    loader=DataLoader(dataset,batch_size=32,shuffle=False,collate_fn=v4.collate_batch_v4)
    predictions=exporter.predict_probs(model,loader,torch.device('cpu'))
    out=ROOT/'examples/reproduced_predictions.tsv'
    exporter.write_prediction_rows(out,predictions,append=False)
    history={r['site_id']:r for r in v4.read_tsv(ROOT/'examples/historical_predictions.tsv')}
    expected=np.array([[float(history[s]['prob_'+c]) for c in v4.CLASS_NAMES] for s in predictions['site_ids']])
    error=float(np.max(np.abs(predictions['probs']-expected)))
    report={'status':'PASSED' if error<1e-5 else 'FAILED','sites':len(rows),
            'maximum_absolute_probability_error':error,'tolerance':1e-5,
            'top1_agreement':float(np.mean(predictions['probs'].argmax(1)==expected.argmax(1))),
            'device':'cpu','torch':torch.__version__,'pair_edges_disabled':True}
    (ROOT/'examples/reproduction_check.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    assert report['status']=='PASSED',report

if __name__=='__main__':main()
