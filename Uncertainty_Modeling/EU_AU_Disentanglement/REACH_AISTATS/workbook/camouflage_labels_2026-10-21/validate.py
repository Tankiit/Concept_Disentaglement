import json
import numpy as np
import pandas as pd
from run import HERE, PARENT, BASE


def main():
    df = pd.read_csv(HERE/'intervals.csv')
    old = pd.read_csv(PARENT/'intervals.csv')
    prior = json.loads((BASE/'config.json').read_text())
    assert len(df) == 9900
    assert not df.duplicated(['seed', 'camouflage', 'mode', 'budget', 'cell', 'target']).any()
    theta, weights = np.array(prior['theta']), np.array(prior['weights'])
    for row in df.itertuples():
        A = np.array(prior['matrices'][str(row.camouflage)])
        jt = weights*A[:, row.cell]
        src = row.cell if row.cell < 10 else A[:-1, row.cell].argmax()
        truth = dict(W_cls=jt[src]/jt[:-1].sum(), W_unk=jt[-1]/jt.sum(),
                     W_label=theta@A[:-1, row.cell]/A[:-1, row.cell].sum())[row.target]
        assert abs(row.truth-truth) < 1e-12
        assert row.covers == (row.lower-1e-6 <= truth <= row.upper+1e-6)
        if row.status != 'empty_set':
            assert np.isfinite([row.lower, row.upper]).all()
            assert -1e-6 <= row.lower <= row.upper+1e-6 <= 1+2e-6
            assert abs(row.width-row.upper+row.lower) < 1e-12
    keys = ['seed', 'camouflage', 'cell', 'target']
    for mode, arm, scale in [('baseline_references', 'both', 1), ('16x_references', 'both', 16), ('exact_sources', 'exact', 0)]:
        a = df[(df['mode'] == mode) & (df.budget == 512)]
        b = old[(old.arm == arm) & (old.scale == scale)]
        paired = a.merge(b, on=keys, suffixes=('_new', '_old'), validate='one_to_one')
        assert len(paired) == 660
        for col in ['lower', 'upper']:
            np.testing.assert_allclose(paired[col+'_new'], paired[col+'_old'], atol=1e-10, rtol=0)
    for col in ['lower', 'upper']:
        fixed = df[df.target != 'W_label'].groupby(['seed', 'camouflage', 'mode', 'cell', 'target'])[col]
        assert fixed.agg(lambda x: x.max()-x.min()).max() < 1e-12
    exact = df[(df['mode'] == 'exact_sources') & (df.budget == -1) & (df.target == 'W_label')]
    np.testing.assert_allclose(exact.lower, exact.truth, atol=1e-10)
    np.testing.assert_allclose(exact.upper, exact.truth, atol=1e-10)
    draws = pd.DataFrame(json.loads((HERE/'label_draws.json').read_text()))
    for _, g in draws.groupby('seed'):
        g = g.sort_values('budget')
        counts = np.array(g.positives.tolist())
        assert ((counts >= 0) & (counts <= g.budget.to_numpy()[:, None])).all()
        diff = np.diff(counts, axis=0)
        assert ((diff >= 0) & (diff <= np.diff(g.budget)[:, None])).all()
    result = dict(records=len(df), baseline_matches_parent=True, targets_verified=True,
        unchanged_class_and_candidate_endpoints=True, exact_laws_and_rates_collapse_label_interval=True,
        nested_label_counts=True, empty_records=int((df.status == 'empty_set').sum()), missed_targets=int((~df.covers).sum()))
    (HERE/'validation.json').write_text(json.dumps(result, indent=2))
    print(result)


if __name__ == '__main__':
    main()
