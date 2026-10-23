"""Existing joint-mass LP with optional audited joint-mass bounds."""
import numpy as np
from scipy.optimize import linprog

def interval(lo,hi,pl,ph,wl,wh,cell,source,jl=None,jh=None,diagnostics=None):
    k,d=lo.shape;size=k*d;rows=[]
    for s in range(k):
        total=np.zeros(size);total[s*d:(s+1)*d]=1
        for c in range(d):
            unit=np.zeros(size);unit[s*d+c]=1
            rows.extend([np.r_[unit-hi[s,c]*total,0.],np.r_[lo[s,c]*total-unit,0.]])
        rows.extend([np.r_[total,-wh[s]],np.r_[-total,wl[s]]])
    for c in range(d):
        col=np.zeros(size);col[c::d]=1
        rows.extend([np.r_[col,-ph[c]],np.r_[-col,pl[c]]])
    if jl is not None:
        for i in range(size):
            unit=np.zeros(size);unit[i]=1
            rows.extend([np.r_[unit,-jh.flat[i]],np.r_[-unit,jl.flat[i]]])
    matrix=np.array(rows);den=np.zeros(size);den[cell::d]=1
    if source is not None:den[(k-1)*d+cell]=0
    num=np.zeros(size);num[(k-1 if source is None else source)*d+cell]=1
    mass=linprog(den,A_ub=matrix[:,:-1],b_ub=-matrix[:,-1],A_eq=np.ones((1,size)),b_eq=[1.],bounds=(0,None))
    if diagnostics is not None:
        diagnostics['minimum_mass'] = float(mass.fun) if mass.success else np.nan
    if mass.status==2:return np.nan,np.nan,'empty_set'
    assert mass.success,mass.message
    if mass.fun<=1e-10:return 0.,1.,'zero_mass_policy'
    eq=np.stack([np.r_[np.ones(size),-1],np.r_[den,0.]])
    values=[]
    for sign in [1.,-1.]:
        fit=linprog(np.r_[sign*num,0],A_ub=matrix,b_ub=np.zeros(len(matrix)),A_eq=eq,b_eq=[0,1],bounds=(0,None))
        assert fit.success,fit.message
        assert np.max(matrix@fit.x)<1e-6
        assert np.max(np.abs(eq@fit.x-[0,1]))<1e-6
        values.append(float(np.clip(sign*fit.fun,0,1)))
    return *values,'optimal'

