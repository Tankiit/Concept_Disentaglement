"""Exact two-hypothesis example: identical predictive beliefs, reversed action value.

All likelihoods are declared. No neural score is frozen after informative data.
"""
import numpy as np
from scipy.special import expit, logit, xlogy
from scipy.stats import binom

P_LOW=.2
P_HIGH=.8
WORLDS=('mixture_unknown','label_unknown')
ACTIONS=('inputs','source_id','reference_label')


def entropy(p):
    p=np.asarray(p,dtype=float)
    return -xlogy(p,p)-xlogy(1-p,1-p)


def informative(world,action):
    if world not in WORLDS or action not in ACTIONS:
        raise ValueError('Unknown world or acquisition')
    return (world,action) in [('mixture_unknown','source_id'),('label_unknown','reference_label')]


def observation_rate(world,action,p):
    if informative(world,action):return p
    if action=='source_id':return .5
    return 0.  # constant input, or Y=0 in reference source 0


def posterior_q(q,n,successes,is_informative=True):
    """Posterior weight of p=.8 from n Bernoulli observations; stable log odds."""
    if not 0<q<1:raise ValueError('Use interior prior weights')
    if not is_informative:return np.zeros_like(np.asarray(successes),dtype=float)+q
    k=np.asarray(successes)
    if np.any(k<0) or np.any(k>n):raise ValueError('Invalid success counts')
    llr=k*np.log(P_HIGH/P_LOW)+(n-k)*np.log((1-P_HIGH)/(1-P_LOW))
    return expit(logit(q)+llr)


def predictive_eu(q):
    """I(Y;H), nats, under the specified two-member predictive distribution."""
    q=np.asarray(q)
    mean=P_LOW+(P_HIGH-P_LOW)*q
    return np.maximum(0.,entropy(mean)-(1-q)*entropy(P_LOW)-q*entropy(P_HIGH))


def target_variance(q):
    q=np.asarray(q)
    return (P_HIGH-P_LOW)**2*q*(1-q)


def eu_bounds(q_lower,q_upper):
    """Concavity gives minimum at endpoints, maximum at q nearest 1/2."""
    lo,hi=np.asarray(q_lower),np.asarray(q_upper)
    if np.any(lo<0) or np.any(hi>1) or np.any(lo>hi):raise ValueError('Invalid weight interval')
    lower=np.minimum(predictive_eu(lo),predictive_eu(hi))
    upper=predictive_eu(np.clip(.5,lo,hi))
    return lower,upper


def exact_expected(world,action,n,q=.5,q_range=(.25,.75)):
    """Enumerate the full count distribution, averaging over the declared prior.

    Expected EU endpoints use the nominal q count law, not a robust expectation.
    """
    yes=informative(world,action)
    k=np.arange(n+1)
    r0=observation_rate(world,action,P_LOW)
    r1=observation_rate(world,action,P_HIGH)
    mass=(1-q)*binom.pmf(k,n,r0)+q*binom.pmf(k,n,r1)
    assert np.isclose(mass.sum(),1.)
    post=posterior_q(q,n,k,yes)
    lo,hi=eu_bounds(posterior_q(q_range[0],n,k,yes),posterior_q(q_range[1],n,k,yes))
    risk=float(mass@target_variance(post))
    return dict(expected_variance=risk,variance_reduction=float(target_variance(q)-risk),
                expected_eu=float(mass@predictive_eu(post)),
                expected_eu_lower=float(mass@lo),expected_eu_upper=float(mass@hi))


def confidence_set(n,k,is_informative,alpha=.05):
    """Fixed-n Hoeffding acceptance of p=.2/.8; not a Bayesian credible set.

    NaN endpoints mark rejected/empty sets, never zero uncertainty.
    """
    k=np.asarray(k)
    if n==0 or not is_informative:
        shape=k.shape
        return np.full(shape,P_LOW),np.full(shape,P_HIGH),np.zeros(shape,dtype=bool)
    radius=np.sqrt(np.log(2/alpha)/(2*n))
    low_ok=np.abs(k/n-P_LOW)<=radius
    high_ok=np.abs(k/n-P_HIGH)<=radius
    empty=~(low_ok|high_ok)
    lower=np.where(low_ok,P_LOW,np.where(high_ok,P_HIGH,np.nan))
    upper=np.where(high_ok,P_HIGH,np.where(low_ok,P_LOW,np.nan))
    return lower,upper,empty
