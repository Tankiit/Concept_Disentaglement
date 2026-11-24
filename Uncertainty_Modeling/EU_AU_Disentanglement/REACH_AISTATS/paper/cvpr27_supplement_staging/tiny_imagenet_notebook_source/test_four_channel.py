import numpy as np
from pathlib import Path
import tempfile
from tiny_four_channel import (
    joint_constraints,sharp_ratio,anchored_bounds,intervals_for_state,
    run_experiment, true_targets
)

def test_exact_confounding_non_identification():
    A0=np.array([.9,.1,.0,.0]);A1=np.array([.1,.9,.0,.0]);A2=(A0+A1)/2
    A=np.stack([A0,A1,A2]);w=np.array([.45,.25,.30]);p=w@A
    G,h,E,f=joint_constraints(A,A,p,p)
    n=np.zeros(12);d=np.zeros(12);n[8]=1;d[0::4]=1
    lo,hi,st=sharp_ratio(G,h,E,f,n,d)
    assert st=='ok'
    assert lo < .00001 and hi > .68, (lo,hi)
    wL=np.array([.45,.25,.30]);wU=np.array([.45,.25,.30]);
    G,h,E,f=joint_constraints(A,A,p,p,wL,wU)
    lo,hi,st=sharp_ratio(G,h,E,f,n,d)
    t=true_targets(A,np.ones_like(A)*.5,w,0)['candidate_membership']
    assert np.isclose(lo,hi,atol=1e-6) and np.isclose(lo,t,atol=1e-6)

def test_anchored_corners():
    rowlo=np.array([[.2,.8],[.1,.9],[.5,.5]])
    label=np.array([.3,.7]); l,u,s=anchored_bounds(rowlo,rowlo,label,label,0)
    assert s=='ok' and np.isclose(l, (.2*.3+.1*.7)/.3) and np.isclose(u,l)

def test_invariance():
    A=np.array([[.55,.45],[.25,.75],[.50,.50]])
    w=np.array([.45,.25,.30]);p=w@A
    st=intervals_for_state(np.array([1000*p[0],1000*p[1]]),
       np.array([[120,80],[80,120]]),np.array([20,10,10]),
       np.array([[40,60],[70,30]]),A[2],0)
    st2=intervals_for_state(np.array([5000*p[0],5000*p[1]]),
       np.array([[120,80],[80,120]]),np.array([30,30,30]),
       np.array([[40,60],[70,30]]),A[2],0)
    assert np.allclose(st['anchored_label'][:2],st2['anchored_label'][:2])
    st3=intervals_for_state(np.array([1000*p[0],1000*p[1]]),
       np.array([[120,80],[80,120]]),np.array([20,10,10]),
       np.array([[10,90],[70,30]]),A[2],0)
    assert np.allclose(st['known_membership'][:2],st3['known_membership'][:2]);
    assert np.allclose(st['candidate_membership'][:2],st3['candidate_membership'][:2]);

def test_end_to_end(tmp_path):
    rng=np.random.default_rng(42)
    arr={}
    for name,n in [('fit',6000),('eval',3000)]:
        src=rng.choice(3,size=n,p=[.45,.25,.3]);cl=rng.choice(4,size=n)
        feat=np.zeros((n,8),dtype='float32')
        feat[:,:4]=np.eye(4)[cl]*4+rng.normal(0,.4,size=(n,4))
        feat[:,4:]=rng.normal(0,.4,size=(n,4))
        err=(rng.random(n)<.1+.2*(src==2)+.1*(cl==2)).astype(np.int8)
        arr[f'{name}_features']=feat;arr[f'{name}_source']=src;arr[f'{name}_error']=err
    cache=tmp_path/'data.npz';np.savez_compressed(cache,**arr)
    summary,intervals=run_experiment(cache,tmp_path/'res',cells=4,repeats=2,budgets=(0,100,1000))
    assert (tmp_path/'res/four_channel_acquisition.png').exists()
    assert len(intervals)==2*2*4*3*3
    assert (intervals[intervals.budget==0].groupby(['regime','repeat','target']).width.nunique()<=1).all()
    assert (intervals[intervals.regime=='confounded'].target=='candidate_membership').any()
if __name__=='__main__':
    import pytest
    raise SystemExit(pytest.main([__file__,'-q']))