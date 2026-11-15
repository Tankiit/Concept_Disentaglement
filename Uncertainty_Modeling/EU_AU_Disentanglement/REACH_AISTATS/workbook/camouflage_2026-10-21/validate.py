"""Independent checks, including the exact confounded fiber endpoints."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent


def main():
    c=json.loads((HERE/'config.json').read_text());df=pd.read_csv(HERE/'intervals.csv')
    floor=pd.read_csv(HERE/'exact_floors.csv');geo=pd.read_csv(HERE/'geometry.csv')
    assert len(df)==6600
    assert not df.duplicated(['seed','camouflage','n','mode','cell','target']).any()
    w=np.array(c['weights']);theta=np.array(c['theta']);beta=np.array(c['camouflage_weights'])
    A=np.array(c['matrices']['1.0']);known=A[:-1]
    np.testing.assert_allclose(A[-1],beta@known)
    s=w[:-1]+w[-1]*beta;tmax=np.min(s/beta)
    endpoint_weights=[np.r_[s,0.],np.r_[s-tmax*beta,tmax]]
    for v in endpoint_weights:np.testing.assert_allclose(v@A,w@A,atol=1e-12)
    for row in floor[floor.camouflage==1].itertuples():
        k=row.cell if row.cell<10 else int(known[:,row.cell].argmax())
        vals=[]
        for v in endpoint_weights:
            joint=v*A[:,row.cell]
            vals.append(joint[-1]/joint.sum() if row.target=='W_unk' else joint[k]/joint[:-1].sum())
        np.testing.assert_allclose([row.lower,row.upper],[min(vals),max(vals)],atol=1e-7)
    assert (geo[geo.camouflage<1]['rank']==11).all()
    assert geo[geo.camouflage==1]['rank'].iloc[0]==10
    assert floor[floor.camouflage<1].width.abs().max()<1e-6
    for row in df.itertuples():
        a=np.array(c['matrices'][str(row.camouflage)]);jt=w*a[:,row.cell]
        k=row.cell if row.cell<10 else int(a[:-1,row.cell].argmax())
        target={'W_cls':jt[k]/jt[:-1].sum(),'W_unk':jt[-1]/jt.sum(),
                'W_label':theta@a[:-1,row.cell]/a[:-1,row.cell].sum()}[row.target]
        assert abs(row.truth-target)<1e-12
        assert row.covers==(row.lower-1e-6<=target<=row.upper+1e-6)
        if row.status!='empty_set':
            assert -1e-6<=row.lower<=row.upper+1e-6<=1+2e-6
            assert abs(row.width-row.upper+row.lower)<1e-12
    draws=pd.DataFrame(json.loads((HERE/'draws.json').read_text()))
    for _,g in draws.groupby('seed'):
        assert len({str(v) for v in g.known_counts})==1
    for _,g in draws.groupby(['seed','camouflage']):
        g=g.sort_values('n');small,large=g.iloc[0],g.iloc[1]
        assert (np.array(large.log_counts)>=np.array(small.log_counts)).all()
        assert small.candidate_counts==large.candidate_counts
        assert sum(small.log_counts)==1000 and sum(large.log_counts)==16000
    result=dict(records=len(df),targets_verified=True,endpoints_verified=True,
        exact_floors_match_analytic_confounded_fiber=True,identified_floors_zero=True,
        known_references_fixed=True,nested_logs=True,confounded_candidate_weight_range=[0,float(tmax)])
    (HERE/'independent_validation.json').write_text(json.dumps(result,indent=2));print(result)


if __name__=='__main__':main()
