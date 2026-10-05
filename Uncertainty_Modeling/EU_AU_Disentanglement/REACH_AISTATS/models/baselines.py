"""EMQ on oracle/estimated discrete conditionals; known-only and augmented fits explicit."""
import numpy as np

def em_prevalence(conditionals, log, max_iter=500, tol=1e-9):
    a=np.asarray(conditionals,dtype=float)
    return em_from_counts(a,np.bincount(log,minlength=a.shape[1]),max_iter,tol)


def em_from_counts(conditionals, counts, max_iter=500, tol=1e-9):
    a=np.asarray(conditionals,dtype=float)
    counts=np.asarray(counts,dtype=float)
    if np.any((counts>0)&(a.sum(0)==0)):
        return None  # closed-set misspecification; do not silently discard novel observations
    w=np.ones(len(a))/len(a)
    for _ in range(max_iter):
        den=w@a
        resp=np.divide(w[:,None]*a,den[None,:],out=np.zeros_like(a),where=den[None,:]>0)
        new=(resp@counts)/counts.sum()
        if np.max(np.abs(new-w)) < tol:
            return new
        w=new
    return w

def bootstrap_em(conditionals, log, n_boot=50, alpha=.05, seed=0):
    rng=np.random.default_rng(seed)
    samples=[]
    for _ in range(n_boot):
        w=em_prevalence(conditionals,rng.choice(log,len(log),replace=True))
        if w is not None:
            samples.append(w)
    if not samples:
        return dict(status='model_misspecified',lower=None,upper=None)
    samples=np.asarray(samples)
    lo,hi=np.quantile(samples,[alpha/2,1-alpha/2],axis=0)
    probe_intervals={}
    # Match REACH's W_cls estimand, preserving dependence between bootstrap weights.
    for cell in range(np.asarray(conditionals).shape[1]):
        known_mass=samples[:,:2]@np.asarray(conditionals)[:2,cell]
        valid=known_mass>1e-12
        if not valid.all():
            probe_intervals[str(cell)]=dict(lower=None,upper=None,status='undefined_known_probe')
        else:
            values=samples[:,0]*np.asarray(conditionals)[0,cell]/known_mass
            left,right=np.quantile(values,[alpha/2,1-alpha/2])
            probe_intervals[str(cell)]=dict(lower=float(left),upper=float(right),status='bootstrap')
    return dict(status='percentile_bootstrap_no_finite_sample_guarantee',lower=lo.tolist(),upper=hi.tolist(),
                known_posterior_intervals=probe_intervals)


def feature_match(conditionals, empirical):
    """Simple simplex least-squares feature matching; no claim to reproduce full robust DFM."""
    from scipy.optimize import minimize
    a=np.asarray(conditionals,dtype=float)
    empirical=np.asarray(empirical,dtype=float)
    result=minimize(lambda w: np.sum((w@a-empirical)**2),np.ones(len(a))/len(a),
                    method='SLSQP',bounds=[(0,1)]*len(a),
                    constraints=[dict(type='eq',fun=lambda w:w.sum()-1)],
                    options=dict(ftol=1e-12,maxiter=1000))
    if not result.success:
        raise RuntimeError('Feature matching solver failed: '+result.message)
    return dict(weights=result.x.tolist(),residual=float(np.linalg.norm(result.x@a-empirical)),
                status='point_estimate_no_confidence_interval')


def em_batch(conditionals, counts, max_iter=2000, tol=1e-9):
    """EMQ on a batch of count vectors (B x cells) at once; rows that are closed-set
    misspecified (mass on a cell no source supports) are dropped, as in em_from_counts."""
    a=np.asarray(conditionals,dtype=float)
    counts=np.atleast_2d(np.asarray(counts,dtype=float))
    ok=~np.any((counts>0)&(a.sum(0)==0)[None,:],axis=1)
    counts=counts[ok]
    if not len(counts):
        return np.empty((0,len(a)))
    totals=counts.sum(1,keepdims=True)
    w=np.full((len(counts),len(a)),1./len(a))
    for _ in range(max_iter):
        den=w@a
        ratio=np.divide(counts,den,out=np.zeros_like(counts),where=den>0)
        new=w*(ratio@a.T)/totals
        if np.max(np.abs(new-w)) < tol:
            return new
        w=new
    return w


def _probe_posteriors(weights, conditionals, cell, theta=None):
    """Per-sample P(Z=0 | cell, known), P(Z=novel | cell), and optionally P(Y=1 | cell)."""
    a=np.asarray(conditionals,dtype=float)[:,cell]
    joint=weights*a[None,:]
    known=joint[:,:2].sum(1); total=joint.sum(1)
    with np.errstate(invalid='ignore',divide='ignore'):
        cls=np.where(known>0,joint[:,0]/known,np.nan)
        unk=np.where(total>0,joint[:,-1]/total,np.nan)
        y=None if theta is None else np.where(total>0,(joint*theta).sum(1)/total,np.nan)
    return cls,unk,y


def emq_probe_intervals(conditionals, counts, cells, n_boot=200, alpha=.05, seed=0):
    """EMQ (Saerens et al. 2002) with a multinomial bootstrap of the log counts.

    Same probe estimands as W_cls and W_unk, percentile intervals, no finite-sample guarantee.
    The novel row is the oracle candidate (a favourable reference, not an open-world detector).
    """
    rng=np.random.default_rng(seed)
    counts=np.asarray(counts); n=int(counts.sum())
    samples=em_batch(conditionals,rng.multinomial(n,counts/n,size=n_boot))
    out={}
    for cell in cells:
        cls,unk,_=_probe_posteriors(samples,conditionals,cell)
        row={}
        for name,values in (('EMQ_cls',cls),('EMQ_unk',unk)):
            if len(values)==0 or not np.all(np.isfinite(values)):
                row.update({f'{name}_lower':np.nan,f'{name}_upper':np.nan,name:np.nan})
            else:
                lo,hi=np.quantile(values,[alpha/2,1-alpha/2])
                row.update({f'{name}_lower':float(lo),f'{name}_upper':float(hi),name:float(hi-lo)})
        out[cell]=row
    return out


def _entropy(p):
    p=np.clip(p,1e-12,1-1e-12)
    return -(p*np.log(p)+(1-p)*np.log(1-p))


def ensemble_epistemic(conditionals, counts, labels, eta, cells, members=20, seed=0):
    """Bootstrap ensemble over BOTH acquisitions; one scalar epistemic score per probe.

    Member j refits EMQ weights on a multinomial resample of the log, re-estimates each known
    source's label rate from a resample of its labels (debiased for the known flip rate), and
    draws the unidentified novel label rate uniformly. Reports mutual information
    H(mean p) - mean H(p) and the member spread of p = P(Y=1 | cell). Neither score says which
    acquisition produced it: that is the point of the comparison.
    """
    rng=np.random.default_rng(seed)
    counts=np.asarray(counts); n=int(counts.sum())
    labels=np.asarray(labels,dtype=float); m=labels.shape[1]
    weights=em_batch(conditionals,rng.multinomial(n,counts/n,size=members))
    noisy=rng.binomial(m,labels.mean(1),size=(len(weights),labels.shape[0]))/m
    clean=np.clip((noisy-eta)/(1-2*eta),0,1)
    thetas=np.column_stack([clean,rng.random(len(weights))])
    out={}
    for cell in cells:
        a=np.asarray(conditionals,dtype=float)[:,cell]
        joint=weights*a[None,:]; total=joint.sum(1)
        if len(total)==0 or np.any(total<=0):
            out[cell]=dict(Ens_MI=np.nan,Ens_std=np.nan,Ens_mean=np.nan)
            continue
        p=(joint*thetas).sum(1)/total
        out[cell]=dict(Ens_MI=float(_entropy(p.mean())-_entropy(p).mean()),
                       Ens_std=float(p.std(ddof=1)),Ens_mean=float(p.mean()))
    return out
