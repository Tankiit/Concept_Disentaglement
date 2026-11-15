import json
import numpy as np
import pandas as pd
from run import HERE,bounds


def main():
    df=pd.read_csv(HERE/'intervals.csv');runs=pd.read_csv(HERE/'compatibility.csv')
    cfg=json.loads((HERE/'config.json').read_text());assign=pd.read_csv(HERE/'image_assignments.csv')
    source=pd.read_csv(HERE/'image_cells.csv');audit=pd.read_csv(HERE/'audit_bounds.csv')
    assert len(df)==15840 and len(runs)==480
    keys=['seed','nominal','budget','identities','method']
    assert not df.duplicated(keys+['cell','target']).any()
    assert not assign.duplicated(['seed','dataset','index']).any()
    join=assign.merge(source,on=['dataset','index'],suffixes=('_saved','_source'),validate='many_to_one')
    assert len(join)==len(assign)
    assert (join.cell_saved==join.cell_source).all() and (join.label_saved==join.label_source).all()
    for seed,g in assign.groupby('seed'):
        assert len(g[g.role=='deployment'])==4000
        assert len(g[g.role=='candidate_audit'])==2048
        assert len(g[g.role=='known_reference'])==2000
        assert len(g[g.role=='nominal_reference'])==1024
        assert len(g[g.role=='probe'])==5000
    A=np.array(cfg['laws']);w=np.array(cfg['weights']);theta=np.linspace(.1,.9,10)
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
    k=['seed','nominal','budget','identities','cell','target']
    direct=df[df.method=='direct'];tv=df[df.method=='audit_tv']
    pairs=direct.merge(tv,on=k,suffixes=('_direct','_tv'),validate='one_to_one')
    finite=pairs[pairs.status_direct!='empty_set']
    assert (finite.status_tv!='empty_set').all()
    assert (finite.lower_direct>=finite.lower_tv-1e-6).all()
    assert (finite.upper_direct<=finite.upper_tv+1e-6).all()
    for col in ['lower','upper']:
        assert direct.groupby(['seed','budget','identities','cell','target'])[col].agg(lambda x:x.max()-x.min()).max()<1e-12
    counts=json.loads((HERE/'counts.json').read_text())
    for d in counts:
        rl,rh=bounds(np.array(d['nominal_counts']),.008);al,ah=bounds(np.array(d['audit_counts']),.01)
        U=min(1,.5*np.maximum(ah-rl,rh-al).sum())
        r=audit[(audit.seed==d['seed'])&(audit.nominal==d['nominal'])&(audit.budget==d['budget'])].iloc[0]
        assert abs(U-r.tv_upper)<1e-12
        g=assign[assign.seed==d['seed']]
        sampled=g[g.role=='candidate_audit'].iloc[:d['budget']]
        np.testing.assert_array_equal(np.bincount(sampled.cell,minlength=11),d['audit_counts'])
        assert sum(d['log_counts'])==4000 and sum(d['nominal_counts'])==512
    m=df.merge(runs,on=keys,validate='many_to_one');m=m[m.target!='W_label']
    assert ((m.status=='empty_set')==m.rejected).all()
    out=dict(records=len(df),runs=len(runs),image_roles_disjoint_within_seed=True,image_scores_verified=True,
        audit_bounds_recomputed=True,direct_contained_in_audit_tv=True,truths_verified=True,
        audit_upper_bound_misses=int((audit.tv_upper+1e-6<audit.true_tv).sum()),
        direct_target_misses=int((~direct.covers).sum()),audit_tv_target_misses=int((~tv.covers).sum()))
    (HERE/'validation.json').write_text(json.dumps(out,indent=2));print(out)


if __name__=='__main__':main()
