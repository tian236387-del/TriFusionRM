"""Check final matched-control checkpoint identity and training configuration."""
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

R = Path('/home/nefu1020220084/summer/StructDualRM/BIB_strengthening_20261028')
O = R/'results/reviewer_followup_20261104/matched_rbp'
sys.path.insert(0, str(R/'code'))
from train_motif_struct_fusion_v4 import MotifStructV4Config, build_parser


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def main():
    original = R/'models/final_no_pair/seed42'
    control = R/'models/matched_rbp_shuffle_gene_validated/seed42'
    checkpoints = {}
    result = {}
    for name,folder in [('authentic_RBP',original),('matched_shuffled_RBP',control)]:
        path = folder/'model.pt'
        checkpoint = torch.load(path,map_location='cpu',weights_only=False)
        state = checkpoint['model_state_dict']
        checkpoints[name] = checkpoint
        shape = [(k,list(v.shape),str(v.dtype)) for k,v in sorted(state.items())]
        metrics = json.loads((folder/'metrics.json').read_text())
        result[name] = dict(path=str(path),sha256=sha256(path),state_architecture=shape,
                            best_epoch=metrics['best_epoch'],best_validation_score=metrics['best_validation_score'],
                            row_counts=metrics['row_counts'])
    assert result['authentic_RBP']['state_architecture'] == result['matched_shuffled_RBP']['state_architecture']
    assert result['authentic_RBP']['row_counts'] == result['matched_shuffled_RBP']['row_counts']
    config_a = json.loads((original/'metrics.json').read_text())['config']
    config_b = json.loads((control/'metrics.json').read_text())['config']
    legacy_defaults = {
        'structure_encoder': 'gat', 'rbp_pos_encoder': 'attention',
        'max_pair_candidates': 1, 'confidence_gated_structure': False,
        'structure_confidence_floor': 0.1,
    }
    source_defaults = asdict(MotifStructV4Config())
    additions = set(config_b)-set(config_a)
    assert additions <= set(legacy_defaults), additions
    resolved_legacy = {}
    for key in additions:
        assert source_defaults[key] == legacy_defaults[key] == config_b[key], key
        resolved_legacy[key] = config_b[key]
    config_a_resolved = dict(config_a, **resolved_legacy)
    assert config_a_resolved==config_b,{k:(config_a_resolved.get(k),config_b.get(k)) for k in set(config_a_resolved)|set(config_b) if config_a_resolved.get(k)!=config_b.get(k)}
    for name in checkpoints:
        result[name].pop('state_architecture')
    a = json.loads((original/'training_config.json').read_text())
    b = json.loads((control/'training_config.json').read_text())
    differences = {k:dict(authentic=a.get(k),matched=b.get(k)) for k in set(a)|set(b) if a.get(k)!=b.get(k)}
    # Data locations are centralized; architecture and optimization must match.
    relocation = {'dataset','rnafold','rbp_features','rbp_positional_features','out_dir'}
    meaningful = {k:v for k,v in differences.items() if k not in relocation and k in a}
    assert not meaningful, meaningful
    parser_defaults = vars(build_parser().parse_args([]))
    added_training_defaults = {k:v for k,v in b.items() if k not in a}
    assert all(k in parser_defaults and v == parser_defaults[k] for k,v in added_training_defaults.items()), added_training_defaults
    assert b['disable_pair_edges'] is True and b['seed']==42
    assert b['shuffle_rbp'] is False and b['shuffle_rbp_bins'] is False
    assert not b['resume'] and not b.get('init_model_path')
    metrics = json.loads((control/'metrics.json').read_text())
    history = metrics['history']
    best = next(x for x in history if x['epoch']==metrics['best_epoch'])
    assert abs(best['selection_score']-metrics['best_validation_score'])<1e-10
    assert metrics['best_validation_score'] >= max(x['selection_score'] for x in history)-b['early_stopping_min_delta']-1e-10
    assert len(history) >= min(b['epochs'],metrics['best_epoch']+b['early_stopping_patience'])
    training_checkpoint = torch.load(control/'checkpoint.pt',map_location='cpu',weights_only=False)
    assert training_checkpoint['best_epoch']==metrics['best_epoch']
    final_state = checkpoints['matched_shuffled_RBP']['model_state_dict']
    assert set(final_state)==set(training_checkpoint['best_state_dict'])
    assert all(torch.equal(value,training_checkpoint['best_state_dict'][key]) for key,value in final_state.items())
    report = dict(status='VERIFIED',models=result,same_state_keys_shapes_and_dtypes=True,same_resolved_model_config=True,
                  same_row_counts=True,configuration_differences=differences,
                  legacy_model_fields_resolved_from_source_defaults=resolved_legacy,
                  added_training_fields_equal_source_parser_defaults=added_training_defaults,
                  training_from_scratch=True,validation_selected_epoch=metrics['best_epoch'],
                  completed_training_epochs=len(history),early_stopping_or_epoch_limit_reached=True,
                  final_model_equals_validation_selected_best_state=True,
                  validation_selection_metric=metrics['selection_metric'],
                  note='The whole RBP tensor permutation is applied in the dataset wrapper, not encoded by the original shuffle_rbp flag. Learned weights are expected to differ between training arms.')
    (O/'checkpoint_and_training_identity_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
