"""Independent checks for boundary comparison artifacts and LP targets."""
import json
import numpy as np
import pandas as pd
from run import HERE, Region


def main():
    a=np.eye(2);p=np.array([.7,.3]);r=Region(a,a,p,p)
    np.testing.assert_allclose(r.interval(0,0)[:2],[1,1])
    np.testing.assert_allclose(r.interval(0,1)[:2],[0,0])
    a=np.full((2,2),.5);p=np.array([.5,.5])
    np.testing.assert_allclose(Region(a,a,p,p).interval(0,0)[:2],[0,1])
    assert not Region(a,a,np.array([1.,0.]),np.array([1.,0.])).feasible
    a=np.array([[1.,0.],[1.,0.]]);p=np.array([1.,0.])
    assert Region(a,a,p,p).interval(1,0)[2]=='zero_mass_policy'
    df=pd.read_csv(HERE/'intervals.csv');points=pd.read_csv(HERE/'point_scores.csv')
    runs=pd.read_csv(HERE/'run_diagnostics.csv')
    assert len(df)==7500 and len(points)==10000 and len(runs)==100
    assert not df.duplicated(['seed','partition','depth','cell','target']).any()
    assert (points.eu>=-1e-9).all() and (points.eu<=points.tu+1e-9).all()
    assert (points.groupby(['partition','depth']).initially_correct.sum()==973).all()
    for (part,depth),g in points.groupby(['partition','depth']):
        for seed in range(10):
            sub=df[(df.partition==part)&(df.depth==depth)&(df.seed==seed)]
            for target in sub.target.unique():
                counts=sub[sub.target==target].set_index('cell')
                expected=g[g.initially_correct].groupby('cell').size().reindex(counts.index,fill_value=0)
                np.testing.assert_array_equal(counts.evaluated_rows,expected)
                expected=g[g.initially_correct & g.quiet_wrong].groupby('cell').size().reindex(counts.index,fill_value=0)
                np.testing.assert_array_equal(counts.quiet_wrong_rows,expected)
    for row in df.itertuples():
        if row.status=='empty_set':assert np.isnan(row.width)
        else:
            assert -1e-6<=row.lower<=row.upper+1e-6<=1+2e-6
            assert abs(row.width-row.upper+row.lower)<1e-12
        if np.isfinite(row.truth):
            assert row.retains_target==(row.lower-1e-6<=row.truth<=row.upper+1e-6)
    label=df[df.target=='W_label']
    for col in ['lower','upper','truth']:
        assert label.groupby(['seed','partition','cell'])[col].agg(lambda v:v.max()-v.min()).max()<1e-12
    out=dict(analytic_LP_cases=5,records=len(df),paired_points_verified=True,
             uncertainty_bounds_verified=True,endpoints_verified=True,
             empty_sets_preserved=True,reference_label_invariance=True)
    (HERE/'independent_validation.json').write_text(json.dumps(out,indent=2));print(out)


if __name__=='__main__':main()
