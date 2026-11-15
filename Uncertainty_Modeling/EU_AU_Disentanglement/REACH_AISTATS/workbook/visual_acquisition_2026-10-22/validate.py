"""Check accounting, saved evidence, truth, decisions and the extended LP."""
import json
import numpy as np
import pandas as pd
from run import HERE, bounds, ratio, solver

def main():
    manifest = pd.read_csv(HERE/'image_assignments.csv')
    purchases = pd.read_csv(HERE/'purchases.csv')
    intervals = pd.read_csv(HERE/'intervals.csv')
    cfg = json.loads((HERE/'config.json').read_text())
    evidence = json.loads((HERE/'evidence.json').read_text())
    images = pd.read_csv(HERE.parent/'candidate_audit_2026-10-22/image_cells.csv')
    assert not manifest.duplicated(['dataset', 'index']).any()
    check = manifest.merge(images, on=['dataset', 'index'], suffixes=('', '_saved'), validate='one_to_one')
    assert len(check) == len(manifest)
    assert (check.cell == check.cell_saved).all() and (check.label == check.label_saved).all()
    assert ((manifest.source == np.where(manifest.dataset=='mnist', manifest.label, 10))).all()
    assert (manifest.parity == np.where(manifest.source<10, manifest.source%2, -1)).all()
    assert len(intervals) == 16*2*11*3
    assert not intervals.duplicated(['arm', 'budget', 'method', 'cell', 'target']).any()
    assert cfg['alpha_per_family']*cfg['families']*16 <= .05+1e-12
    for saved in evidence:
        arm, budget = saved['arm'], saved['budget']
        p = purchases[(purchases.arm==arm)&(purchases.budget==budget)]
        assert len(p) == budget and p.cost.sum() == budget
        assert not p.duplicated(['dataset', 'index']).any()
        tagged = p.merge(manifest, on=['dataset', 'index'], suffixes=('', '_manifest'), validate='one_to_one')
        for field in ['source', 'cell', 'parity']:
            assert (tagged[field] == tagged[field+'_manifest']).all()
        assert (tagged.role == np.where(tagged.action=='audit', 'audit', 'reference')).all()
        for action, n in saved['allocation'].items():
            assert (p.action==action).sum() == n
        def histogram(frame):
            return np.array([[((frame.source==s)&(frame.cell==c)).sum() for c in range(11)] for s in range(11)])
        refs = pd.concat([manifest[manifest.role=='initial'], p[p.action!='audit']])
        assert np.array_equal(histogram(refs), saved['reference_counts'])
        assert np.array_equal(histogram(p[p.action=='audit']), saved['audit_counts'])
        assert np.array_equal(np.bincount(manifest[manifest.role=='log'].cell, minlength=11), saved['log_counts'])
        anchor = pd.concat([refs[refs.source==s].iloc[:int(np.array(saved['reference_counts'])[:10].sum(1).min())] for s in range(10)])
        anchor_counts = np.array([[((anchor.source%2==v)&(anchor.cell==c)).sum() for c in range(11)] for v in range(2)])
        assert np.array_equal(anchor_counts, saved['anchor_counts'])
    for arm in cfg['arms']:
        prior = set()
        for budget in cfg['budgets']:
            p = purchases[(purchases.arm==arm)&(purchases.budget==budget)]
            now = set(zip(p.dataset, p['index']))
            assert prior <= now
            prior = now
    # Census truth independently from the cached scored-image table.
    populations = [images[(images.dataset=='mnist')&(images.label==s)] for s in range(10)]
    populations.append(images[images.dataset=='fashion'])
    A = np.array([np.bincount(x.cell, minlength=11)/len(x) for x in populations])
    assert np.allclose(A, cfg['laws'])
    w = np.array(cfg['weights'])
    for row in intervals.itertuples():
        a = A[:, row.cell]
        if row.target == 'W_unk':
            truth = w[-1]*a[-1]/(w@a)
        elif row.target == 'W_cls':
            truth = w[row.source]*a[row.source]/(w[:10]@a[:10])
        else:
            truth = a[:10][np.arange(10)%2==1].sum()/a[:10].sum()
        assert abs(truth-row.truth) < 1e-10
        if row.status != 'empty_set':
            assert -1e-7 <= row.lower <= row.upper+1e-7 <= 1+1e-6
            assert row.covers == (row.lower-1e-6 <= truth <= row.upper+1e-6)
    base = intervals[intervals.budget==0].drop(columns=['arm']).sort_values(['method','cell','target'])
    for arm in cfg['arms'][1:]:
        a = intervals[(intervals.budget==0)&(intervals.arm==arm)].sort_values(['method','cell','target'])
        b = intervals[(intervals.budget==0)&(intervals.arm==cfg['arms'][0])].sort_values(['method','cell','target'])
        assert np.allclose(a[['lower','upper']], b[['lower','upper']], equal_nan=True)
    # Exact joint observations must identify membership even with unrestricted rows.
    joint = np.array([[.4,.1],[.1,.4]])
    low, high, status = solver.interval(np.zeros((2,2)), np.ones((2,2)),
                                       np.zeros(2), np.ones(2), np.zeros(2), np.ones(2),
                                       0, None, joint, joint)
    assert status=='optimal' and np.allclose([low,high], [.2,.2])
    impossible = joint.copy(); impossible[0,0] = .9
    assert solver.interval(np.zeros((2,2)), np.ones((2,2)), np.zeros(2), np.ones(2),
                           np.zeros(2), np.ones(2), 0, None, impossible, impossible)[2]=='empty_set'
    lo, hi = bounds(np.zeros((2,3)))
    assert (lo==0).all() and (hi==1).all()
    assert ratio(np.zeros(2), np.ones(2), [False,True])[:2] == (0.,1.)
    decisions = pd.read_csv(HERE/'decisions.csv')
    for _, group in decisions.groupby(['arm','budget','method','target','threshold']):
        assert np.isclose(group.probe_mass.sum(), 1.)
        assert 0 <= group.error_mass.sum() <= group.certified_mass.sum()+1e-9 <= 1+1e-9
    result = dict(records=len(intervals), settings=len(evidence), image_roles_disjoint=True,
                  purchase_costs_and_nested_prefixes_verified=True, saved_counts_verified=True,
                  truth_verified=True, baseline_equal_across_arms=True, analytic_solver_checks=2,
                  zero_count_policy_verified=True, decision_mass_normalized=True,
                  observed_nonempty_misses=int((~intervals.covers & (intervals.status!='empty_set')).sum()),
                  rejected_interval_records=int((intervals.status=='empty_set').sum()))
    (HERE/'validation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
