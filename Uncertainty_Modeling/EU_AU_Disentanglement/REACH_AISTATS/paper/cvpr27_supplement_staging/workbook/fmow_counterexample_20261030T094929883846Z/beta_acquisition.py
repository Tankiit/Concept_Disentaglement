"""Independent Beta source-mixture posterior for real-record acquisition studies.

T=(1-W)*Theta0+W*Theta1. Source identity and stratified target label
are different observation channels. Exact one-step squared-error Bayes values
are derived in the companion notebook. Entropy is integrated by quadrature.
"""
import numpy as np
from scipy.special import roots_jacobi, xlogy


def beta_moments(a,b):
    a,b=np.asarray(a),np.asarray(b)
    if np.any(a<=0) or np.any(b<=0):raise ValueError('Positive Beta shapes required')
    m=a/(a+b)
    return m,m*(1-m)/(a+b+1)


def moments(shapes):
    s=np.asarray(shapes,dtype=float)
    if s.shape[-2:]!=(3,2):raise ValueError('Expected (...,3,2) shapes for W,Theta0,Theta1')
    m,v=beta_moments(s[...,0],s[...,1])
    w,t0,t1=m[...,0],m[...,1],m[...,2]
    vw,v0,v1=v[...,0],v[...,1],v[...,2]
    mean=(1-w)*t0+w*t1
    variance=((1-w)**2+vw)*v0+(w*w+vw)*v1+vw*(t1-t0)**2
    return mean,variance


def action_values(shapes):
    s=np.asarray(shapes,dtype=float)
    m,v=beta_moments(s[...,0],s[...,1])
    totals=s.sum(-1)
    identity=(m[...,2]-m[...,1])**2*v[...,0]/(totals[...,0]+1)
    label0=(1-m[...,0])**2*v[...,1]/(totals[...,1]+1)
    label1=m[...,0]**2*v[...,2]/(totals[...,2]+1)
    return identity,.5*(label0+label1)


def entropy(p):return -xlogy(p,p)-xlogy(1-p,1-p)


def beta_quadrature(a,b,order):
    x,w=roots_jacobi(order,b-1,a-1)
    w=w/w.sum()
    return (x+1)/2,w


def epistemic_mi(shapes,order=24):
    s=np.asarray(shapes,dtype=float)
    if s.shape!=(3,2):raise ValueError('One posterior state at a time')
    grids=[beta_quadrature(a,b,order) for a,b in s]
    w,t0,t1=[g[0] for g in grids]
    probabilities=(1-w[:,None,None])*t0[None,:,None]+w[:,None,None]*t1[None,None,:]
    expected=np.einsum('i,j,k,ijk->',grids[0][1],grids[1][1],grids[2][1],entropy(probabilities))
    mean,_=moments(s)
    return max(0.,float(entropy(mean)-expected))


def update(shapes,channel,outcome,source=None):
    s=np.array(shapes,dtype=float,copy=True)
    index=0 if channel=='source_id' else 1+int(source) if channel=='reference_label' else -1
    if index<0 or index>2 or outcome not in (0,1):raise ValueError('Invalid observation')
    s[index,0 if outcome else 1]+=1
    return s


def enumerate_one_step(shapes,channel):
    s=np.asarray(shapes,dtype=float)
    mean,var=moments(s)
    m,_=beta_moments(s[:,0],s[:,1])
    expected=0.
    groups=[(None,1.,m[0])] if channel=='source_id' else [(0,.5,m[1]),(1,.5,m[2])]
    for source,weight,rate in groups:
        for y,prob in [(0,1-rate),(1,rate)]:
            _,after=moments(update(s,channel,y,source))
            expected+=weight*prob*after
    return float(var-expected)
