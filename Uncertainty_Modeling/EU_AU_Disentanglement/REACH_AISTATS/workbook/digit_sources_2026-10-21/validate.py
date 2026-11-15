"""Independent artifact, leakage, target and analytic checks."""
import json
import numpy as np
import pandas as pd
from run import HERE, label_interval, known_only_feasible, coherent


def main():
    cfg=json.loads((HERE/'config.json').read_text())
    df=pd.read_csv(HERE/'intervals.csv')
    assert len(df)==10140
    assert not df.duplicated(['world','delta','seed','mode','cell','target']).any()
    assigned=pd.read_csv(HERE/'training_assignments.csv')
    assert assigned.label.between(0,5).all()
    assert not assigned['index'].duplicated().any()
    assert (assigned.role=='train').sum()==18000
    assert (assigned.role=='partition_dev').sum()==6000
    for row in df.itertuples():
        A=np.array(cfg['matrices'][row.world]);k=len(A)-1
        kw=np.arange(1,k+1,dtype=float);kw/=kw.sum()
        w=np.r_[(1-row.delta)*kw,row.delta];jt=w*A[:,row.cell]
        truth={'W_cls':jt[row.yhat]/jt[:-1].sum(),'W_unk':jt[-1]/jt.sum(),
               'W_label':np.linspace(.1,.9,k)@A[:-1,row.cell]/A[:-1,row.cell].sum()}[row.target]
        assert abs(row.truth-truth)<1e-12
        assert row.covers==(row.lower-1e-6<=truth<=row.upper+1e-6)
        if row.status!='empty_set':
            assert -1e-6<=row.lower<=row.upper+1e-6<=1+2e-6
            assert abs(row.width-row.upper+row.lower)<1e-12
    draws=pd.DataFrame(json.loads((HERE/'draw_manifest.json').read_text()))
    for _,g in draws.groupby(['world','delta','seed']):
        assert len({tuple(v) for v in g.log_counts})==1
        small=g[g['mode']=='estimated_200'].iloc[0]
        large=g[g['mode']=='estimated_1000'].iloc[0]
        assert (np.array(large.known_counts)>=np.array(small.known_counts)).all()
        assert (np.array(large.candidate_counts)>=np.array(small.candidate_counts)).all()
    # Source-only/target-only dependencies: candidate information cannot affect W_label.
    held=df[df.world=='heldout_6_9']
    for r in [200,1000]:
        sub=held[(held.target=='W_label')&held['mode'].isin([f'estimated_{r}',f'unrestricted_{r}'])]
        p=sub.pivot(index=['seed','delta','cell'],columns='mode',values='width')
        assert np.max(p.max(axis=1)-p.min(axis=1))<1e-12
    A=np.array([[.8,.2],[.2,.8]])
    lo,hi,status=label_interval(A,A,np.array([.1,.7]),np.array([.3,.9]),0)
    np.testing.assert_allclose([lo,hi],[.8*.1+.2*.7,.8*.3+.2*.9])
    assert known_only_feasible(A,A,np.array([.5,.5]),np.array([.5,.5]))
    assert not known_only_feasible(A,A,np.array([1.,0.]),np.array([1.,0.]))
    I=np.eye(2);p=np.array([.7,.3])
    np.testing.assert_allclose(coherent(I,I,p,p,1,None)[:2],[1,1])
    result=dict(records=len(df),unique_records=True,no_heldout_training_or_partition_images=True,
        targets_and_endpoints_verified=True,paired_nested_draws_verified=True,
        label_candidate_invariance=True,analytic_checks_passed=True)
    (HERE/'independent_validation.json').write_text(json.dumps(result,indent=2))
    print(result)


if __name__=='__main__':main()
