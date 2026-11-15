import json
from pathlib import Path
import numpy as np
from solver import interval


def main():
    A=np.array([[1.,0.],[.5,.5]]);w=np.array([.5,.5])
    for delta in [0,.1,.5,1.]:
        lower=max(0,.5-delta);upper=min(1,.5+delta)
        got=interval(A,A,np.zeros(2),np.ones(2),w,w,0,None,delta)
        np.testing.assert_allclose(got[:2],[lower/(1+lower),upper/(1+upper)],atol=1e-8)
    # Zero nominal candidate weight is handled without division by that weight.
    got=interval(A,A,np.array([1.,0.]),np.array([1.,0.]),np.array([1.,0.]),np.array([1.,0.]),0,None,.5)
    np.testing.assert_allclose(got[:2],[0,0],atol=1e-8)
    # A disjoint input law cannot be explained by an exact candidate with zero TV.
    B=np.array([[1.,0.],[1.,0.]])
    assert interval(B,B,np.array([0.,1.]),np.array([0.,1.]),w,w,1,None,0)[2]=='empty_set'
    out=dict(analytic_checks=6,two_cell_fractional_extrema=True,zero_candidate_weight=True,contradiction_rejected=True)
    (Path(__file__).resolve().parent/'solver_checks.json').write_text(json.dumps(out,indent=2));print(out)


if __name__=='__main__':main()
