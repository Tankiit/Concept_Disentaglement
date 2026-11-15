"""Validate saved artifacts without rerunning image encoding or all inference."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def main():
    cfg=json.loads((HERE/'config.json').read_text())
    for name,sha in cfg['frozen_artifacts'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
    for name,sha in cfg['datasets'].items():
        assert hashlib.sha256((ROOT/f'cache/mnist_reach/{name}_test.npz').read_bytes()).hexdigest()==sha,name
    frozen=json.loads((ROOT/'workbook/boundary_reach_2026-10-20/config.json').read_text())
    for name,value in cfg['thresholds'].items():assert value==frozen[name]
    frame=pd.read_csv(HERE/'regions.csv')
    keys=['domain','delta','n','m','seed','partition','mode']
    assert len(frame)==4800 and not frame.duplicated(keys).any()
    assert frame.groupby(keys[:-2]+['mode']).size().eq(5).all()
    for _,g in frame.groupby(['domain','delta','n','m','seed']):
        assert g.realized_novel_fraction.nunique()==1
    for _,g in frame.groupby(['delta','n','seed']):
        assert g.realized_novel_fraction.nunique()==1  # same novelty selector across domains
    valid=frame[frame['mode'].isin(['matching','unrestricted'])]
    assert valid.status.eq('optimal').all()
    assert np.isfinite(valid[['lower','upper']]).all().all()
    assert (valid.lower>=-1e-7).all() and (valid.upper<=1+1e-7).all()
    assert (valid.lower<=valid.upper+1e-7).all()
    np.testing.assert_allclose(valid.width,valid.upper-valid.lower,atol=1e-7)
    np.testing.assert_allclose(frame[frame['mode']=='unrestricted'].upper,1.,atol=1e-7)
    # Observed coverage/clean-control results, not assertions of universal calibration.
    assert valid.covers.eq(True).all()
    assert not frame[frame.delta==0].detects.any()
    missing=frame[(frame['mode']=='letters_reference') & frame.status.eq('empty_set')]
    assert missing[['lower','upper','width']].isna().all().all()
    with np.load(HERE/'cell_populations.npz') as z:
        for domain in ('mnist','fashion','blood','letters'):
            y=z[f'{domain}_labels']
            for name,d in cfg['partitions'].items():
                cells=z[f'{domain}_{name}']
                assert cells.shape==y.shape and cells.min()>=0 and cells.max()<d
    manifest=pd.DataFrame(json.loads((HERE/'draw_manifest.json').read_text()))
    assert len(manifest)==40
    assert manifest.groupby('seed').reference_sha256.nunique().eq(1).all()
    for weights in manifest.known_weights:np.testing.assert_allclose(sum(weights),1.,atol=1e-12)
    result=dict(region_evaluations=len(frame),valid_matching_or_unrestricted=len(valid),
                matched_coverage_records=int(valid[valid['mode']=='matching'].covers.sum()),
                clean_positive_flags=int(frame[frame.delta==0].detects.sum()),
                incompatible_letter_models=len(missing),frozen_artifacts_verified=True,
                thresholds_unchanged=True,paired_draws_verified=True)
    (HERE/'validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
