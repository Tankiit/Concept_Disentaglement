import json
import numpy as np
import pandas as pd
from run import HERE,PARENT,CASES


def main():
    df=pd.read_csv(HERE/'intervals.csv');rf=pd.read_csv(HERE/'compatibility.csv')
    old=pd.read_csv(PARENT/'intervals.csv');cfg=json.loads((PARENT/'config.json').read_text())
    assert len(df)==13860 and len(rf)==420
    keys=['seed','case','identities','delta']
    assert not df.duplicated(keys+['cell','target']).any()
    A=np.array(cfg['laws']['fashion']);w=np.array(cfg['weights']);theta=np.array(cfg['theta'])
    for row in df.itertuples():
        jt=w*A[:,row.cell];src=row.cell if row.cell<10 else A[:-1,row.cell].argmax()
        truth=dict(W_cls=jt[src]/jt[:-1].sum(),W_unk=jt[-1]/jt.sum(),W_label=theta@A[:-1,row.cell]/A[:-1,row.cell].sum())[row.target]
        assert abs(row.truth-truth)<1e-12
        assert row.covers==(row.lower-1e-6<=truth<=row.upper+1e-6)
        if row.status=='empty_set':assert np.isnan([row.lower,row.upper,row.width]).all()
        else:
            assert np.isfinite([row.lower,row.upper]).all()
            assert -1e-6<=row.lower<=row.upper+1e-6<=1+2e-6
            assert abs(row.width-row.upper+row.lower)<1e-12
            assert abs(row.truth_distance-max(row.lower-truth,truth-row.upper,0))<1e-12
    for case,bad,t in CASES:
        a=df[(df.case==case)&(df.delta==0)]
        b=old[(old.bad==bad)&(old.t==t)&(old['mode']=='estimated')]
        m=a.merge(b,on=['seed','identities','cell','target'],suffixes=('_a','_b'),validate='one_to_one')
        assert len(m)==660
        for col in ['lower','upper','truth']:
            np.testing.assert_allclose(m[col+'_a'],m[col+'_b'],atol=1e-9,rtol=0,equal_nan=True)
    # Expanded candidate boxes nest, unlike merely increasing sample budgets.
    for _,g in df.groupby(['seed','case','identities','cell','target']):
        g=g.sort_values('delta');v=g[g.status!='empty_set']
        if len(v):
            assert (g[g.delta>=v.delta.min()].status!='empty_set').all()
            assert (np.diff(v.lower)<=1e-6).all()
            assert (np.diff(v.upper)>=-1e-6).all()
    label=df[df.target=='W_label']
    for col in ['lower','upper']:
        assert label.groupby(['seed','cell'])[col].agg(lambda x:x.max()-x.min()).max()<1e-12
    m=df.merge(rf,on=keys,validate='many_to_one');m=m[m.target!='W_label']
    assert ((m.status=='empty_set')==m.rejected).all()
    oracle=pd.read_csv(HERE/'oracle_diagnostics.csv')
    m=df.merge(oracle,on=['seed','case'],validate='many_to_one')
    valid_assumption=m[m.delta>=m.true_linf_error-1e-12]
    out=dict(records=len(df),runs=len(rf),parent_nominal_endpoints_match=True,targets_verified=True,
        intervals_nested_under_expansion=True,label_invariance=True,rejection_consistency=True,
        records_at_budget_covering_actual_linf_error=len(valid_assumption),
        misses_at_budget_covering_actual_linf_error=int((~valid_assumption.covers).sum()),
        empty_at_budget_covering_actual_linf_error=int((valid_assumption.status=='empty_set').sum()))
    (HERE/'validation.json').write_text(json.dumps(out,indent=2));print(out)


if __name__=='__main__':main()
