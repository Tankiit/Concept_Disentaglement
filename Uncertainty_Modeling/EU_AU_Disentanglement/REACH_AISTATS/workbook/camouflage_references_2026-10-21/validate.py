"""Check paired baselines, truths, acquisition counts and target invariants."""
import json
import numpy as np
import pandas as pd
from run import HERE, PARENT, BASE


def main():
    df = pd.read_csv(HERE/'intervals.csv')
    parent = pd.read_csv(PARENT/'intervals.csv')
    prior = json.loads((BASE/'config.json').read_text())
    assert len(df) == 6600
    assert not df.duplicated(['seed', 'camouflage', 'arm', 'scale', 'cell', 'target']).any()
    w, theta = np.array(prior['weights']), np.array(prior['theta'])
    for row in df.itertuples():
        A = np.array(prior['matrices'][str(row.camouflage)])
        jt = w*A[:, row.cell]
        source = row.cell if row.cell < 10 else A[:-1, row.cell].argmax()
        truth = dict(W_cls=jt[source]/jt[:-1].sum(), W_unk=jt[-1]/jt.sum(),
                     W_label=theta@A[:-1, row.cell]/A[:-1, row.cell].sum())[row.target]
        assert abs(truth-row.truth) < 1e-12
        assert row.covers == (row.lower-1e-6 <= truth <= row.upper+1e-6)
        if row.status != 'empty_set':
            assert np.isfinite([row.lower, row.upper]).all()
            assert -1e-6 <= row.lower <= row.upper+1e-6 <= 1+2e-6
            assert abs(row.width-row.upper+row.lower) < 1e-12
    # Every 1x arm and the exact control must reproduce the prior experiment.
    keys = ['seed', 'camouflage', 'cell', 'target']
    for arm in ['known_only', 'candidate_only', 'both', 'exact']:
        new = df[(df.arm == arm) & (df.scale == (0 if arm == 'exact' else 1))]
        old = parent[(parent.a == 8192) & (parent['mode'] == ('exact_sources' if arm == 'exact' else 'estimated_sources'))]
        paired = new.merge(old, on=keys, suffixes=('_new', '_old'), validate='one_to_one')
        assert len(paired) == 660
        for col in ['lower', 'upper', 'truth']:
            np.testing.assert_allclose(paired[col+'_new'], paired[col+'_old'], atol=1e-10, rtol=0)
    label = df[(df.arm == 'candidate_only') & (df.target == 'W_label')]
    for col in ['lower', 'upper']:
        assert label.groupby(['seed', 'camouflage', 'cell'])[col].agg(lambda x: x.max()-x.min()).max() < 1e-12
    draws = pd.DataFrame(json.loads((HERE/'reference_draws.json').read_text()))
    for (_, _, arm), g in draws.groupby(['seed', 'camouflage', 'arm']):
        g = g.sort_values('scale')
        for col, base, varied in [('known_counts', 1000, arm in ['known_only', 'both']),
                                  ('candidate_counts', 6500, arm in ['candidate_only', 'both'])]:
            counts = np.array(g[col].tolist())
            assert (np.diff(counts, axis=0) >= 0).all()
            expected = base*(g.scale.to_numpy() if varied else np.ones(len(g)))
            sums = counts.sum(axis=-1)
            if sums.ndim == 2:
                expected = np.repeat(expected[:, None], 10, axis=1)
            np.testing.assert_array_equal(sums, expected)
    result = dict(records=len(df), parent_baseline_and_exact_controls_match=True,
        targets_verified=True, nested_reference_counts=True, candidate_only_label_invariance=True,
        empty_records=int((df.status == 'empty_set').sum()), missed_targets=int((~df.covers).sum()))
    (HERE/'validation.json').write_text(json.dumps(result, indent=2))
    print(result)


if __name__ == '__main__':
    main()
