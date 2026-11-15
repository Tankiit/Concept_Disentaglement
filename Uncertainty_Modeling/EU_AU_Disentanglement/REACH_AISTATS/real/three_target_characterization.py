"""Exact finite-hypothesis characterization; no training, fitted data or sampling.

Four hypotheses (H,U) share a fixed three-cell source law. H controls the
deployment label law; U controls hidden source/label structure. Oracle-U
conditioning changes information, not the model family or prior definition.
"""
from fractions import Fraction as F
from itertools import product
import json
import numpy as np
from scipy.special import xlogy
from real.source_acquisition import polytope, ratio_interval

FAMILIES = ('known_membership', 'candidate_membership', 'reference_target', 'aligned_reference_control')
TARGETS = ('known_membership', 'reference_target', 'candidate_membership')
CELL_LAW = np.array([.5, .3, .2])


def family(name):
    if name not in FAMILIES: raise ValueError(name)
    rows=[]
    for h,u in product(range(2), repeat=2):
        p=F(2+h,5);d=F(2*u-1,10)
        if name=='known_membership':
            r=F(1+2*u,4);w=[F(4,5)*r,F(4,5)*(1-r),F(1,5)];theta=[p]*3
        elif name=='candidate_membership':
            c=F(1+2*u,5);w=[(1-c)/2,(1-c)/2,c];theta=[p]*3
        elif name=='reference_target':
            w=[F(3,5),F(1,5),F(1,5)];theta=[p+d,p-3*d,p]
        else:
            w=[F(2,5),F(2,5),F(1,5)];theta=[p+d,p-d,p]
        deployment=sum(a*b for a,b in zip(w,theta))
        assert deployment==p  # Rational arithmetic, not a numerical near-match.
        assert all(0<t<1 for t in theta) and sum(w)==1 and all(t>0 for t in w)
        target=[w[0]/(w[0]+w[1]),(theta[0]+theta[1])/2,w[2]]
        rows.append(dict(h=h,u=u,weights=np.array(w,float),theta=np.array(theta,float),
                         targets=np.array(target,float),p=float(p),p_exact=str(p),
                         targets_exact=[str(t) for t in target]))
    return rows


def state_weights(models,state='unresolved'):
    if state=='unresolved':return np.full(len(models),1/len(models))
    if state not in ('oracle_U0','oracle_U1'):raise ValueError(state)
    weights=np.array([m['u']==int(state[-1]) for m in models],float)
    return weights/weights.sum()


def entropy(p):
    p=np.asarray(p)
    return -xlogy(p,p)-xlogy(1-p,1-p)


def predictive_signature(models,weights):
    """Full second-order law over joint distributions P(C,Y), not just scalar MI."""
    atoms={}
    for m,q in zip(models,weights):
        joint=np.column_stack([CELL_LAW*(1-m['p']),CELL_LAW*m['p']]).ravel()
        key=tuple(np.round(joint,14))
        atoms[key]=atoms.get(key,0.)+float(q)
    return tuple((key,round(mass,14)) for key,mass in sorted(atoms.items()) if mass>0)


def predictive_mi(models,weights):
    p=np.array([m['p'] for m in models])
    return float(entropy(weights@p)-weights@entropy(p))


def target_statistics(models,weights):
    values=np.array([m['targets'] for m in models]);active=weights>0
    mean=weights@values
    var=weights@((values-mean)**2)
    return dict(mean=mean,variance=var,lower=values[active].min(0),upper=values[active].max(0),
                width=np.ptp(values[active],axis=0))


def likelihood(models,channel):
    """Columns are outcomes; each row sums to one. Every finite outcome is possible."""
    if channel=='unlabeled_input':return np.tile(CELL_LAW,(len(models),1))
    if channel=='source_identity':return np.array([m['weights'] for m in models])
    if channel=='deployment_label':return np.array([[1-m['p'],m['p']] for m in models])
    if channel=='reference_label':
        return np.array([[.5*(1-m['theta'][0]),.5*m['theta'][0],
                          .5*(1-m['theta'][1]),.5*m['theta'][1]] for m in models])
    raise ValueError(channel)


def oracle_partition(models,channel):
    if channel=='input_law':return [tuple(CELL_LAW)]*len(models)
    if channel=='source_weights':return [tuple(m['weights']) for m in models]
    if channel=='deployment_label_law':return [m['p'] for m in models]
    if channel=='reference_label_laws':return [tuple(m['theta'][:2]) for m in models]
    raise ValueError(channel)


def expected_update(models,weights,channel,oracle=False):
    """Exact finite sum over outcomes; no Monte Carlo noise or fitted target truth."""
    if oracle:
        keys=oracle_partition(models,channel)
        L=np.array([[key==outcome for outcome in dict.fromkeys(keys)] for key in keys],float)
    else:L=likelihood(models,channel)
    before=target_statistics(models,weights)
    after_var=np.zeros(3);after_width=np.zeros(3);expected_mean=np.zeros(3);mean_second=np.zeros(3)
    after_mi=0.;details=[]
    for outcome in range(L.shape[1]):
        mass=float(weights@L[:,outcome])
        if mass==0:continue
        post=weights*L[:,outcome]/mass
        stat=target_statistics(models,post)
        after_var+=mass*stat['variance'];after_width+=mass*stat['width'];expected_mean+=mass*stat['mean']
        mean_second+=mass*stat['mean']**2;after_mi+=mass*predictive_mi(models,post)
        details.append(dict(outcome=outcome,probability=mass,posterior=post.tolist(),
                            lower=stat['lower'].tolist(),upper=stat['upper'].tolist()))
    assert np.allclose(expected_mean,before['mean'],atol=1e-13)
    assert np.allclose(before['variance']-after_var,mean_second-before['mean']**2,atol=1e-13)
    return dict(expected_variance=after_var,variance_reduction=before['variance']-after_var,
                expected_width=after_width,width_reduction=before['width']-after_width,
                expected_predictive_mi=after_mi,mi_reduction=predictive_mi(models,weights)-after_mi,
                outcomes=details)


def membership_lp_certificate(name,state):
    """Independent LP certificate on a continuous source-weight extension.

    Candidate family has known w0=w1. Known-family candidate mass is 1/5.
    The bounds are not obtained from zero-denominator fallback.
    """
    models=family(name);q=state_weights(models,state);active=q>0
    w=np.array([m['weights'] for m in models])[active]
    row=np.tile(CELL_LAW,(3,1))
    G,b,E,f=polytope(row,row,(CELL_LAW,CELL_LAW),(w.min(0),w.max(0)))
    if name=='candidate_membership':
        E=np.vstack([E,np.r_[np.ones(3),-np.ones(3),np.zeros(3)]]);f=np.r_[f,0.]
    numerator=np.zeros((3,3));denominator=np.zeros((3,3))
    if name=='known_membership':numerator[0,0]=1;denominator[:2,0]=1
    elif name=='candidate_membership':numerator[2,0]=1;denominator[:,0]=1
    else:raise ValueError(name)
    return ratio_interval((G,b,E,f),numerator,denominator)


def identifiable_source_controls():
    """Exact known full-rank rows plus input marginal identify the source weights."""
    rows=np.array([[.7,.2,.1],[.2,.7,.1],[.1,.2,.7]])
    result=[]
    for name in ('known_membership','candidate_membership'):
        models=family(name)
        for u in (0,1):
            m=next(m for m in models if m['u']==u)
            w=m['weights'];marginal=w@rows
            P=polytope(rows,rows,(marginal,marginal))
            for target in ('known_membership','candidate_membership'):
                num=np.zeros_like(rows);den=np.zeros_like(rows)
                if target=='known_membership':num[0,0]=1;den[:2,0]=1
                else:num[2,0]=1;den[:,0]=1
                truth=float((num*w[:,None]*rows).sum()/(den*w[:,None]*rows).sum())
                result.append(dict(family=name,u=u,target=target,truth=truth,
                                   marginal=json.dumps(marginal.tolist()),affine_rank=int(np.linalg.matrix_rank((rows[:2]-rows[2]).T)),
                                   **ratio_interval(P,num,den)))
    return result
