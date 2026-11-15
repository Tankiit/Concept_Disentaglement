import json
import numpy as np
import pandas as pd
from run import HERE,helper


def main():
    cfg=json.loads((HERE/'config.json').read_text())
    df=pd.read_csv(HERE/'intervals.csv');runs=pd.read_csv(HERE/'compatibility.csv')
    assert len(df)==7920 and len(runs)==240
    keys=['seed','deployed','declared','mode','identities']
    assert not df.duplicated(keys+['cell','target']).any()
    w=np.array(cfg['weights']);theta=np.array(cfg['theta'])
    for row in df.itertuples():
        A=np.array(cfg['laws'][row.deployed]);jt=w*A[:,row.cell]
        src=row.cell if row.cell<10 else A[:-1,row.cell].argmax()
        truth=dict(W_cls=jt[src]/jt[:-1].sum(),W_unk=jt[-1]/jt.sum(),W_label=theta@A[:-1,row.cell]/A[:-1,row.cell].sum())[row.target]
        assert abs(truth-row.truth)<1e-12
        assert row.covers==(row.lower-1e-6<=truth<=row.upper+1e-6)
        assert row.narrow_wrong==(row.status!='empty_set' and row.width<=.1 and not row.covers)
        if row.status=='empty_set':assert np.isnan([row.lower,row.upper,row.width]).all()
        else:
            assert np.isfinite([row.lower,row.upper]).all()
            assert -1e-6<=row.lower<=row.upper+1e-6<=1+2e-6
            assert abs(row.width-row.upper+row.lower)<1e-12
    joined=df.merge(runs,on=keys,validate='many_to_one')
    coupled=joined[joined.target!='W_label']
    assert ((coupled.status=='empty_set')==coupled.rejected).all()
    label=df[df.target=='W_label']
    for col in ['lower','upper']:
        assert label.groupby(['seed','mode','cell'])[col].agg(lambda x:x.max()-x.min()).max()<1e-12
    # Exact contradictory identity and input evidence must reject.
    A=np.array([[1.,0.],[0.,1.]])
    r=helper.interval(A,A,np.array([1.,0.]),np.array([1.,0.]),np.array([0.,1.]),np.array([0.,1.]),0,None)
    assert r[2]=='empty_set'
    # Adding identities cannot restore feasibility of a rejected zero-identity set.
    zero=runs[runs.identities==0].set_index(keys[:-1]).rejected
    more=runs[runs.identities==8192].set_index(keys[:-1]).rejected
    assert (~zero | more).all()
    for d in json.loads((HERE/'draws.json').read_text()):
        assert np.array(d['known_counts']).shape==(10,11)
        assert (np.array(d['known_counts']).sum(1)==1000).all()
        for key,total in [('candidate_counts',6500),('log_counts',16000),('identity_counts',8192)]:assert sum(d[key])==total
    result=dict(records=len(df),runs=len(runs),truths_verified=True,label_invariance=True,
        rejection_consistency=True,analytic_contradiction_rejected=True,
        matched_rejections=int(runs[runs.deployed==runs.declared].rejected.sum()),
        matched_nonempty_misses=int((~df[(df.matched)&(df.status!='empty_set')].covers).sum()),
        mismatched_rejections=int(runs[runs.deployed!=runs.declared].rejected.sum()),
        mismatched_nonempty_misses=int((~df[(~df.matched)&(df.status!='empty_set')].covers).sum()))
    (HERE/'validation.json').write_text(json.dumps(result,indent=2));print(result)


if __name__=='__main__':main()
