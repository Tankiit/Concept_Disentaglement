"""Finite-cell acquisition experiment; probability bounds, not entropy decomposition.

Joint variables z[s,c]=P(S=s,X=c) retain reference-row uncertainty.
Hoeffding boxes and Charnes--Cooper LPs are explained in the companion notebook.
"""
import numpy as np
from scipy.optimize import linprog


def frequency_box(counts, alpha):
    counts = np.asarray(counts, dtype=float)
    n = counts.sum()
    if n == 0:
        return np.zeros_like(counts), np.ones_like(counts)
    radius = np.sqrt(np.log(2 * counts.size / alpha) / (2 * n))
    return np.maximum(0, counts / n - radius), np.minimum(1, counts / n + radius)


def polytope(row_lo, row_hi, marginal=None, weights=None):
    """Nonnegative joint masses; row-normalized conditional boxes are linear."""
    lo, hi = np.asarray(row_lo), np.asarray(row_hi)
    s, d = lo.shape
    eye = np.eye(s*d).reshape(s, d, s*d)
    row = eye.sum(1)
    col = eye.sum(0)
    G, h = [], []
    for j in range(s):
        for c in range(d):
            G.extend([eye[j,c] - hi[j,c]*row[j], lo[j,c]*row[j] - eye[j,c]])
            h.extend([0., 0.])
    for bounds, vectors in [(marginal, col), (weights, row)]:
        if bounds is not None:
            lower, upper = bounds
            for v, a, b in zip(vectors, lower, upper):
                G.extend([v, -v]); h.extend([b, -a])
    return np.asarray(G), np.asarray(h), np.ones((1,s*d)), np.ones(1)


def ratio_interval(poly, numerator, denominator):
    """Sharp range within the declared polytope; undefined denominator -> [0,1].

    Empty sets are reported, never silently interpreted as certainty.
    Numerical residuals include transformed LP constraints and nonnegativity.
    """
    G,h,E,f = poly
    c, d = np.asarray(numerator).ravel(), np.asarray(denominator).ravel()
    base = linprog(d, A_ub=G, b_ub=h, A_eq=E, b_eq=f, bounds=(0,None), method='highs')
    if base.status == 2:
        return dict(lower=np.nan, upper=np.nan, width=np.nan, status='empty', residual=np.nan)
    if not base.success:
        raise RuntimeError(base.message)
    if base.fun <= 1e-9:
        return dict(lower=0., upper=1., width=1., status='undefined_denominator_policy', residual=0.)
    # y=z/(d.z), t=1/(d.z): G y <= h t, E y = f t, d.y=1.
    AG = np.column_stack([G,-h])
    AE = np.vstack([np.column_stack([E,-f]), np.r_[d,0.]])
    be = np.r_[np.zeros(len(f)),1.]
    endpoints, residuals = [], []
    for sign in [1.,-1.]:
        out = linprog(sign*np.r_[c,0.], A_ub=AG, b_ub=np.zeros(len(h)),
                      A_eq=AE,b_eq=be,bounds=(0,None),method='highs')
        if not out.success:
            raise RuntimeError(out.message)
        endpoints.append(float(c @ out.x[:-1]))
        residuals.append(max(0., float(np.max(AG@out.x)),
                             float(np.max(np.abs(AE@out.x-be))),float(-out.x.min())))
    lower, upper = np.clip(endpoints,0,1)
    if lower > upper+1e-7:
        raise AssertionError('Inverted bounds')
    return dict(lower=lower, upper=upper, width=max(0.,upper-lower),status='ok',residual=max(residuals))


def membership_bounds(lo, hi, marginal, weights, cell):
    s,d = lo.shape
    P = polytope(lo,hi,marginal,weights)
    known = np.zeros((s,d)); known[:2,cell]=1
    first = np.zeros((s,d)); first[0,cell]=1
    novel = np.zeros((s,d)); novel[-1,cell]=1
    total = np.zeros((s,d)); total[:,cell]=1
    return {'known_membership': ratio_interval(P,first,known),
            'candidate_membership': ratio_interval(P,novel,total)}


def anchored_label_bounds(lo,hi,theta_lo,theta_hi,cell,anchor=np.array([.5,.5])):
    P = polytope(lo[:2],hi[:2],weights=(anchor,anchor))
    denom = np.zeros_like(lo[:2]); denom[:,cell]=1
    low = ratio_interval(P,denom*np.asarray(theta_lo)[:,None],denom)
    high = ratio_interval(P,denom*np.asarray(theta_hi)[:,None],denom)
    if low['status']=='empty' or high['status']=='empty':
        return dict(lower=np.nan,upper=np.nan,width=np.nan,status='empty',residual=np.nan)
    return dict(lower=low['lower'],upper=high['upper'],width=high['upper']-low['lower'],
                status=low['status'],residual=max(low['residual'],high['residual']))


def categorical_stream(prob, sizes, rng):
    """Nested acquisition counts without saving large redundant raw streams."""
    counts=np.zeros(len(prob),dtype=int); previous=0; result={}
    for n in sorted(set(sizes)):
        counts=counts+rng.multinomial(n-previous,prob)
        result[n]=counts.copy(); previous=n
    return result
