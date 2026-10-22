"""TV sensitivity around any normalized nominal row in its confidence box."""
import numpy as np
from scipy.optimize import linprog

RETRIES=[]


def solve(cost,**kwargs):
    result=linprog(cost,**kwargs)
    if result.status not in (0,2):
        initial=result.message
        result=linprog(cost,**kwargs,method='highs-ds',options={'presolve':False})
        RETRIES.append(dict(initial=initial,status=int(result.status),message=result.message))
    if result.status not in (0,2):raise RuntimeError(result.message)
    return result


def interval(lo,hi,pl,ph,wl,wh,cell,source,delta):
    k,d=lo.shape;n=k*d+2*d
    # z = deployed joint masses; v = nominal candidate row times candidate weight;
    # e = absolute z_candidate-v, bounded in total by 2 delta candidate weight.
    rows=[];rhs=[];totals=[]
    for s in range(k):
        total=np.zeros(n);total[s*d:(s+1)*d]=1;totals.append(total)
        for c in range(d):
            unit=np.zeros(n);unit[(s*d+c) if s<k-1 else k*d+c]=1
            rows.extend([unit-hi[s,c]*total,lo[s,c]*total-unit]);rhs.extend([0,0])
        rows.extend([total,-total]);rhs.extend([wh[s],-wl[s]])
    for c in range(d):
        col=np.zeros(n);col[c:k*d:d]=1
        rows.extend([col,-col]);rhs.extend([ph[c],-pl[c]])
        diff=np.zeros(n);diff[(k-1)*d+c]=1;diff[k*d+c]=-1
        e=np.zeros(n);e[k*d+d+c]=1
        rows.extend([diff-e,-diff-e]);rhs.extend([0,0])
    tv=np.zeros(n);tv[k*d+d:]=1;tv-=2*delta*totals[-1]
    rows.append(tv);rhs.append(0)
    norm=np.zeros(n);norm[:k*d]=1
    nominal=np.zeros(n);nominal[k*d:k*d+d]=1;nominal-=totals[-1]
    eq=np.stack([norm,nominal]);beq=np.array([1.,0.])
    matrix=np.array(rows);b=np.array(rhs)
    den=np.zeros(n);den[cell:k*d:d]=1
    if source is not None:den[(k-1)*d+cell]=0
    num=np.zeros(n);num[((k-1) if source is None else source)*d+cell]=1
    mass=solve(den,A_ub=matrix,b_ub=b,A_eq=eq,b_eq=beq,bounds=(0,None))
    if mass.status==2:return np.nan,np.nan,'empty_set'
    if mass.fun<=1e-10:return 0.,1.,'zero_mass_policy'
    cc=np.c_[matrix,-b];ceq=np.vstack([np.c_[eq,-beq],np.r_[den,0.]])
    crhs=np.array([0.,0.,1.]);values=[]
    for sign in [1.,-1.]:
        fit=solve(np.r_[sign*num,0.],A_ub=cc,b_ub=np.zeros(len(b)),A_eq=ceq,b_eq=crhs,bounds=(0,None))
        assert fit.success,fit.message
        assert np.max(cc@fit.x)<1e-6
        assert np.max(np.abs(ceq@fit.x-crhs))<1e-6
        values.append(float(np.clip(sign*fit.fun,0,1)))
    return *values,'optimal'
