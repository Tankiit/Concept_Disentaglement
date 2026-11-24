"""Independent rational enumeration of the informative-cell construction.

No dependence on the archived constant-input implementation. Probabilities,
posterior risks and regret calculations use fractions, including repeated draws.
"""
from fractions import Fraction as F
from itertools import product
from math import comb

P = (F(1, 5), F(4, 5))
WORLDS = ('M', 'L')
ACTIONS = ('identity', 'source0_label', 'cell')
BUDGETS = (0, 1, 2, 5, 10, 20, 50, 100, 200)


def joint(world, p):
    """P(S,Y,C | p), derived from the two generative factorizations."""
    if world not in WORLDS:
        raise ValueError(world)
    out = {}
    for s, y, c in product((0, 1), repeat=3):
        ps = (p if s else 1-p) if world == 'M' else F(1, 2)
        py = F(y == s) if world == 'M' else (p if y else 1-p)
        out[s, y, c] = ps * py * (F(3, 4) if c == y else F(1, 4))
    return out


def predictive_joint(world, p):
    j = joint(world, p)
    return {(c, y): sum(j[s, y, c] for s in (0, 1))
            for c, y in product((0, 1), repeat=2)}


def rates(world, action):
    """Marginalize deployment fields; condition on S=0 for stratified labels."""
    result = []
    for p in P:
        j = joint(world, p)
        if action == 'identity':
            r = sum(v for (s, y, c), v in j.items() if s == 1)
        elif action == 'cell':
            r = sum(v for (s, y, c), v in j.items() if c == 1)
        elif action == 'source0_label':
            r = (sum(v for (s, y, c), v in j.items() if s == 0 and y == 1)
                 / sum(v for (s, y, c), v in j.items() if s == 0))
        else:
            raise ValueError(action)
        result.append(r)
    return tuple(result)


def variance(q):
    return (P[1]-P[0])**2 * q*(1-q)


def count_outcomes(channel, n, q=F(1, 2)):
    if n < 0 or not 0 <= q <= 1:
        raise ValueError('Invalid budget or prior')
    for k in range(n+1):
        weights = tuple(prior * comb(n, k)*r**k*(1-r)**(n-k)
                        for prior, r in zip((1-q, q), channel))
        mass = sum(weights)
        if mass:  # Impossible events have no posterior and contribute no risk.
            yield k, mass, weights[1]/mass


def risk(channel, n, q=F(1, 2)):
    return sum(mass*variance(post) for _, mass, post in count_outcomes(channel, n, q))


def sequence_risk(channel, n, q=F(1, 2)):
    """Independent ordered-sequence calculation for small-budget cross-checks."""
    total = F(0)
    for seq in product((0, 1), repeat=n):
        weights = []
        for prior, r in zip((1-q, q), channel):
            w = prior
            for outcome in seq:
                w *= r if outcome else 1-r
            weights.append(w)
        mass = sum(weights)
        if mass:
            mean = sum(w*p for w, p in zip(weights, P))/mass
            total += sum(w*(p-mean)**2 for w, p in zip(weights, P))
    return total


def value(world, action):
    return variance(F(1, 2)) - risk(rates(world, action), 1)
