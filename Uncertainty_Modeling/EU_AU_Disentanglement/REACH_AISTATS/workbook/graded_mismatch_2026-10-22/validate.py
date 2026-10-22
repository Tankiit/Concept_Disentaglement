import json
import numpy as np
import pandas as pd
from run import HERE,PARENT


def main():
    cfg=json.loads((HERE/'config.json').read_text());df=pd.read_csv(HERE/'intervals.csv')
    runs=pd.read_csv(HERE/'compatibility.csv');old=pd.read_csv(PARENT/'intervals.csv')
    assert len(df)==18480 and len(runs)==560
    keys=['seed','bad','t','mode','identities']
    assert not df.duplicated(keys+['cell','target']).any()
    A=np.array(cfg['laws']['fashion']);w=np.array(cfg['weights']);theta=np.array(cfg['theta'])
    for row in df.itertuples():
        jt=w*A[:,row.cell];src=row.cell if row.cell<10 else A[:-1,row.cell].argmax()
        truth=dict(W_cls=jt[src]/jt[:-1].sum(),W_unk=jt[-1]/jt.sum(),W_label=theta@A[:-1,row.cell]/A[:-1,row.cell].sum())[row.target]
        assert abs(row.truth-truth)<1e-12
        assert row.covers==(row.lower-1e-6<=truth<=row.upper+1e-6)
        if row.status=='empty_set':assert np.isnan([row.lower,row.upper,row.width,row.truth_distance]).all()
        else:
            assert np.isfinite([row.lower,row.upper]).all()
            assert -1e-6<=row.lower<=row.upper+1e-6<=1+2e-6
            assert abs(row.width-row.upper+row.lower)<1e-12
            assert abs(row.truth_distance-max(row.lower-truth,truth-row.upper,0))<1e-12
        assert row.narrow_wrong==(row.status!='empty_set' and row.width<=.1 and not row.covers)
    # Endpoints must reproduce both matched and fully mismatched parent runs.
    join=['seed','mode','identities','cell','target']
    for bad in cfg['bad_laws']:
        for t in [0,1]:
            new=df[(df.bad==bad)&(df.t==t)]
            parent=old[(old.deployed=='fashion')&(old.declared==('fashion' if t==0 else bad))]
            m=new.merge(parent,on=join,validate='one_to_one',suffixes=('_new','_old'))
            assert len(m)==1320
            for col in ['lower','upper','truth']:
                np.testing.assert_allclose(m[col+'_new'],m[col+'_old'],atol=1e-9,rtol=0,equal_nan=True)
    labels=df[df.target=='W_label']
    for col in ['lower','upper']:
        assert labels.groupby(['seed','mode','cell'])[col].agg(lambda x:x.max()-x.min()).max()<1e-12
    merged=df.merge(runs,on=keys,validate='many_to_one')
    m=merged[merged.target!='W_label']
    assert ((m.status=='empty_set')==m.rejected).all()
    # Additional identity constraints cannot restore an empty zero-identity set.
    pivot=runs.pivot(index=keys[:-1],columns='identities',values='rejected')
    assert (~pivot[0]|pivot[8192]).all()
    pop=pd.read_csv(HERE/'population.csv')
    np.testing.assert_allclose(pop.fixed_identity_tv,.2*pop.candidate_tv,atol=1e-12)
    assert (pop.min_marginal_tv<=pop.fixed_identity_tv+1e-8).all()
    draws=pd.DataFrame(json.loads((HERE/'draws.json').read_text()))
    for _,g in draws.groupby('seed'):
        for col in ['known_counts','log_counts','identity_counts','target_positives']:
            assert len({json.dumps(x) for x in g[col]})==1
    result=dict(records=len(df),runs=len(runs),parent_endpoints_reproduced=True,truths_verified=True,
        fixed_evidence_verified=True,label_invariance=True,rejection_consistency=True,population_residual_identity=True,
        matched_rejections=int(runs[runs.t==0].rejected.sum()),
        matched_nonempty_misses=int((~df[(df.t==0)&(df.status!='empty_set')].covers).sum()))
    (HERE/'validation.json').write_text(json.dumps(result,indent=2));print(result)


if __name__=='__main__':main()
