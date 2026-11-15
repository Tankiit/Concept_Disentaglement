"""Validate paired experiment artifacts independently of their aggregation."""
import json
import numpy as np
import pandas as pd
from repeat import OUT, BASE, SEEDS, BUDGETS
from run import coherent


def main():
    df = pd.read_csv(OUT/'intervals.csv')
    assert len(df)==SEEDS*len(BUDGETS)*11*2*2
    assert not df.duplicated(['seed','n','cell','target','method']).any()
    z = np.load(BASE/'single_run_inputs.npz')
    yhat = pd.read_csv(BASE/'single_run.csv').set_index('cell').yhat
    for row in df.itertuples():
        jt = z['true_weights']*z['truth_conditionals'][:,row.cell]
        expected = jt[-1]/jt.sum() if row.target=='novelty' else jt[int(yhat[row.cell])]/jt[:-1].sum()
        assert abs(row.truth-expected)<1e-12
        assert row.covers == (row.lower-1e-6<=expected<=row.upper+1e-6)
        if row.empty:
            assert np.isnan(row.width)
        else:
            assert -1e-6<=row.lower<=row.upper+1e-6<=1+2e-6
            assert abs(row.width-(row.upper-row.lower))<1e-12
    draws = json.loads((OUT/'draws.json').read_text())
    for seed in range(SEEDS):
        a,b = [v for v in draws if v['seed']==seed]
        for key in ['known_counts','novel_counts','log_counts']:
            assert (np.array(b[key])>=np.array(a[key])).all()
        for item in [a,b]:
            assert (np.array(item['known_counts']).sum(1)==item['m']).all()
            assert sum(item['novel_counts'])==item['q']
            assert sum(item['log_counts'])==item['n']
    a=np.eye(2); p=np.array([.7,.3])
    assert np.allclose(coherent(a,a,p,p,1,None)[:2],[1,1])
    assert np.allclose(coherent(a,a,p,p,0,0)[:2],[1,1])
    a=np.full((2,2),.5); p=np.array([.5,.5])
    assert np.allclose(coherent(a,a,p,p,0,None)[:2],[0,1])
    p=np.array([1.,0.]); a=np.array([[1.,0.],[1.,0.]])
    assert coherent(a,a,p,p,1,None)[2]=='zero_mass_policy'
    a=np.full((2,2),.5)
    assert coherent(a,a,p,p,0,None)[2]=='empty_set'
    result=dict(rows=len(df),unique_records=True,targets_verified=True,
                endpoints_verified=True,nested_counts_verified=True,analytic_LP_cases=5)
    (OUT/'validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':
    main()
