import json
import numpy as np
import pandas as pd
from run import HERE,BASE,interval,cam


def main():
    A=np.full((2,2),.5);p=np.array([.5,.5])
    np.testing.assert_allclose(interval(A,A,p,p,np.zeros(2),np.ones(2),0,None)[:2],[0,1])
    w=np.array([.7,.3])
    np.testing.assert_allclose(interval(A,A,p,p,w,w,0,None)[:2],[.3,.3])
    np.testing.assert_allclose(interval(A,A,p,p,w,w,0,0)[:2],[1,1])
    # Unconstrained identity bounds reproduce the existing joint-mass solver.
    A=np.array([[.8,.2],[.3,.7]])
    np.testing.assert_allclose(interval(A,A,np.array([.4,.3]),np.array([.7,.6]),np.zeros(2),np.ones(2),0,None)[:2],
        cam.coherent(A,A,np.array([.4,.3]),np.array([.7,.6]),0,None)[:2],atol=1e-8)
    df=pd.read_csv(HERE/'intervals.csv');prior=json.loads((BASE/'config.json').read_text())
    assert len(df)==6600 and not df.duplicated(['seed','camouflage','mode','a','cell','target']).any()
    weights=np.array(prior['weights']);theta=np.array(prior['theta'])
    for row in df.itertuples():
        a=np.array(prior['matrices'][str(row.camouflage)]);jt=weights*a[:,row.cell]
        src=row.cell if row.cell<10 else int(a[:-1,row.cell].argmax())
        truth={'W_cls':jt[src]/jt[:-1].sum(),'W_unk':jt[-1]/jt.sum(),'W_label':theta@a[:-1,row.cell]/a[:-1,row.cell].sum()}[row.target]
        assert abs(row.truth-truth)<1e-12
        assert row.covers==(row.lower-1e-6<=truth<=row.upper+1e-6)
        if row.status!='empty_set':
            assert np.isfinite([row.lower,row.upper]).all()
            assert -1e-6<=row.lower<=row.upper+1e-6<=1+2e-6
            assert abs(row.width-row.upper+row.lower)<1e-12
    label=df[df.target=='W_label']
    for col in ['lower','upper']:
        assert label.groupby(['seed','mode','cell'])[col].agg(lambda v:v.max()-v.min()).max()<1e-12
    # Adding any identity constraints can only shrink the same zero-identity set.
    baseline=df[df.a==0].set_index(['seed','camouflage','mode','cell','target'])
    for row in df[df.a>0].itertuples():
        old=baseline.loc[(row.seed,row.camouflage,row.mode,row.cell,row.target)]
        if row.status!='empty_set':
            assert row.lower>=old.lower-1e-6 and row.upper<=old.upper+1e-6
    draws=pd.DataFrame(json.loads((HERE/'identity_draws.json').read_text()))
    for _,g in draws.groupby('seed'):
        g=g.sort_values('a');counts=np.array(g.counts.tolist())
        assert (np.diff(counts,axis=0)>=0).all()
        np.testing.assert_array_equal(counts.sum(1),g.a)
    out=dict(records=len(df),analytic_cases=4,targets_verified=True,label_invariance=True,
        identity_intervals_nested_inside_zero_identity_baseline=True,nested_identity_counts=True,
        empty_records=int((df.status=='empty_set').sum()),missed_targets=int((~df.covers).sum()))
    (HERE/'validation.json').write_text(json.dumps(out,indent=2));print(out)


if __name__=='__main__':main()
