"""Real-shift audit on frozen CLIP ViT-B/32 CIFAR-10 features (one family, one backbone).

Three arms share one engine (cells from real/head.py, the MMD ball from models/ball.py):

label_shift  Deployment resamples the clean CIFAR-10 test pool at a skewed class prevalence pi.
             The frozen-conditionals assumption holds, so W_cls should contract in n and the
             truth P(Y=yhat | cell, known) should stay covered. m = labelled reference images per
             class: it sets the estimated conditionals and their radius, so on real data m
             cross-loads on W_cls (README: report cross-effects separately). W_label is the
             fixed-anchor width over per-entry conditional intervals, exact in n by construction.
held_out     Two classes are withheld from the head and appear in deployment with mass delta.
             q_oracle: the novel row is estimated from a disjoint pool of the withheld classes
             (a favourable reference: tests contraction when the family is right).
             q_uniform: the declared uniform candidate (tests whether a wrong q is flagged).
             s = distance from q to the known convex hull is reported; it is the separation
             the novelty claim needs.
corruption   Deployment is CIFAR-10-C at severities 1..5 with the same pi. The conditionals move,
             so the declared family is wrong; the incompatibility rate should rise with severity,
             and coverage is reported, not assumed.

Labels of deployment images are used only to define truth after fitting. Alpha is split three
ways (log, references, label-channel entries), as in train/cache.py.
"""
import argparse, json, os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from models.ball import MMDBall, embedding_radius
from models.baselines import em_batch, feature_match
import test.inference as inf
import test.slopes as sl
from real.head import LDAHead, conditionals, load, CORRUPTIONS
import real.expectations as RE

ALPHA = .05
BASE = dict(n=4000, m=2000, eta=0.)
GRIDS = dict(n=(500, 1000, 2000, 4000, 8000, 16000), m=(125, 250, 500, 1000, 2000, 4000),
             eta=(.025, .05, .1, .2, .3))
PI_RATIO = 10.        # max/min class prevalence in deployment
DELTA = .1            # novel mass in the held-out arm
NOVEL = (8, 9)        # ship, truck
PLACEBO_OFFSET = 10**6
SET_KIND = 'l2'  # exact MMD ball; 'box' is the coordinate outer LP (near-vacuous at 10 classes)
# Mixture sources. 'classes': one source per CIFAR-10 class. 'super4': four superclasses whose
# within-group composition is held fixed, so deployment is still pure label shift over sources.
GROUPINGS = dict(classes=tuple((c,) for c in range(10)),
                 super4=((0, 8), (1, 9), (3, 5), (2, 4, 6, 7)),
                 # CIFAR-100 limit arm: trees and vehicles_1 as two sources (fixed composition).
                 c100_2=((47, 52, 56, 59, 96), (8, 13, 48, 58, 90)))
# Coverage is judged at the cone solver's accuracy: Clarabel returns bounds of order 1e-9 where
# the exact extremum is 0, so a 1e-9 test misreads solver error as miscoverage.
SOLVER_TOL = 1e-6
_CTX = {}


def skewed_prevalence(k, ratio=PI_RATIO):
    p = ratio**(-np.arange(k)/(k-1))
    return p/p.sum()


def hull_distance(q, known):
    """s: Euclidean distance from q to the convex hull of the known rows."""
    k = len(known)
    res = minimize(lambda w: np.sum((w@known-q)**2), np.ones(k)/k, method='SLSQP',
                   bounds=[(0, 1)]*k, constraints=[dict(type='eq', fun=lambda w: w.sum()-1)])
    return float(np.sqrt(max(res.fun, 0.)))


def feature_name(dataset, split, backbone='clip'):
    """CLIP files keep their original names; other backbones are prefixed."""
    return f'{dataset}_{split}' if backbone == 'clip' else f'{backbone}_{dataset}_{split}'


def build_context(arm, novel=NOVEL, seed=0, corruption=None, severity=None, set_kind=SET_KIND,
                  grouping='classes', prevalence='skew', dirichlet_alpha=1., delta=DELTA,
                  dataset='cifar10', backbone='clip', classes=None, label=None):
    """Fit the frozen head once and reduce every pool to integer cells.

    classes: the class ids in play (default: all ten CIFAR-10 classes). In the held-out arm the
    novel ids are removed from them to form the known set. label: the arm name written to rows.
    """
    train = load(feature_name(dataset, 'train', backbone))
    test = load(feature_name(dataset, 'test', backbone))
    classes = tuple(range(10)) if classes is None else tuple(classes)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(train['y']))
    dev, ref = perm[:10000], perm[10000:]
    known = np.array([c for c in classes if arm != 'held_out' or c not in novel])
    head = LDAHead.fit(train['X'][dev], train['y'][dev], known)
    ref_cells, ref_y = head.cell(train['X'][ref]), train['y'][ref]
    if arm == 'corruption':
        # CIFAR-10.1 is a natural-shift pool with no severities: all of it is deployment.
        pool = load('cifar10_1_test') if corruption == 'cifar10_1' else load(f'cifar10c_{corruption}')
        mask = pool['severity'] == severity if 'severity' in pool else np.ones(len(pool['y']), bool)
        dep_X, dep_y = pool['X'][mask], pool['y'][mask]
    else:
        dep_X, dep_y = test['X'], test['y']
    dep_cells = head.cell(dep_X)
    if grouping == 'classes':
        groups = [(int(c),) for c in known]
    else:
        groups = [tuple(c for c in g if c in known) for g in GROUPINGS[grouping]]
        groups = [g for g in groups if g]
    # Within a group, classes enter in equal proportion in references and in deployment.
    pool = lambda cells, y, g: [cells[y == c] for c in g]
    ctx = dict(arm=arm, label=label or arm, known=known, groups=groups, grouping=grouping, d=head.cells, set_kind=set_kind,
               ref_cells={g: pool(ref_cells, ref_y, g) for g in groups},
               dep_cells={g: pool(dep_cells, dep_y, g) for g in groups},
               cell_group=np.array([next(i for i, g in enumerate(groups) if c in g) for c in known]+[-1]))
    if arm == 'held_out':
        ctx['ref_cells']['novel'] = pool(ref_cells, ref_y, novel)
        ctx['dep_cells']['novel'] = pool(dep_cells, dep_y, novel)
    ctx.update(prevalence=prevalence, dirichlet_alpha=dirichlet_alpha,
               delta=float(delta) if arm == 'held_out' else 0.)
    if arm == 'held_out':
        ctx['novel'] = np.asarray(novel)
    # Truth: conditionals of the pools deployment is actually drawn from.
    keys = groups+(['novel'] if arm == 'held_out' else [])
    ctx['truth_conditionals'] = np.vstack([_group_hist(ctx['dep_cells'][g], head.cells) for g in keys])
    ctx['head_accuracy'] = float(np.mean(known[np.minimum(dep_cells, len(known)-1)][np.isin(dep_y, known)]
                                         == dep_y[np.isin(dep_y, known)]))
    return ctx


def _group_hist(pools, d):
    """P(cell | group) with classes in equal proportion inside the group."""
    return np.mean([np.bincount(p, minlength=d)/len(p) for p in pools], axis=0)


def _draw_from(pools, count, rng):
    """count draws from a group: class uniformly within the group, image uniformly within class."""
    which = rng.integers(0, len(pools), count)
    out = np.empty(count, dtype=int)
    for j, p in enumerate(pools):
        hit = which == j
        out[hit] = p[rng.integers(0, len(p), hit.sum())]
    return out


def true_weights(ctx, seed):
    """Deployment source weights. 'skew': fixed geometric prevalence; 'dirichlet': a fresh
    Dirichlet(alpha) prevalence per seed, shared by every budget of that seed (so the grid stays
    paired). Novel mass delta is appended in the held-out arm."""
    k = len(ctx['groups'])
    if ctx['prevalence'] == 'dirichlet':
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 505]))
        pi = rng.dirichlet(np.full(k, ctx['dirichlet_alpha']))
    else:
        pi = skewed_prevalence(k)
    return np.r_[(1-ctx['delta'])*pi, ctx['delta']] if 'novel' in ctx else pi


def _draw_log(ctx, n, seed, weights):
    """Deployment cells: sources by the true weights, then images within the source pool."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 101]))
    keys = ctx['groups']+(['novel'] if 'novel' in ctx else [])
    labels = rng.choice(len(keys), n, p=weights)
    out = np.empty(n, dtype=int)
    for g, key in enumerate(keys):
        hit = labels == g
        out[hit] = _draw_from(ctx['dep_cells'][key], int(hit.sum()), rng)
    return out


def _bernstein(p, m, a):
    """Two-sided Bernstein bound on |Bin(m, p)/m - p|, failure probability a."""
    log = np.log(2/a)
    return np.sqrt(2*p*(1-p)*log/m)+2*log/(3*m)


def _reference(ctx, m, seed, eta=0.):
    """Estimated source rows from m labelled reference images per source, with their radii.

    Returns (rows on the simplex, entry centres, L2 radius per row-set, per-entry half-width),
    with failure budgets ALPHA/(3(k+1)) per row and ALPHA/(3kd) per entry, as for clean labels.

    eta = 0: per-source histograms (prefix of one seeded stream per source).
    eta > 0: each reference label is flipped with known probability eta to a uniformly chosen
    other source. T_l = (1/m) sum_i 1{noisy label = l} e_cell(i), and A_hat = M^{-1} T with
    M = (1-eta) I + eta/(k-1) (11^T - I). The error of T_l splits into
      (i) cell sampling given which images landed in l: McDiarmid over the n_l landed images;
     (ii) composition: sum_j |c_jl/m - M_jl| ||A_j|| <= sum_j Bernstein(M_jl), c_jl ~ Bin(m, M_jl).
    (ii) vanishes like sqrt(eta), so the radius is continuous at eta = 0. M^{-1} amplifies a row
    error by at most kappa = ||M^{-1}||_inf. Rows are Euclidean-projected onto the simplex, which
    cannot move them further from the true rows. Flip draws are shared across eta and m.
    """
    k, d, top = len(ctx['groups']), ctx['d'], max(GRIDS['m'])
    row_a, entry_a = ALPHA/(3*(k+1)), ALPHA/(3*k*d)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 303]))
    streams = [_draw_without_replacement(ctx['ref_cells'][g], top, rng)[:m] for g in ctx['groups']]
    m = min(map(len, streams))
    if eta == 0:
        rows = np.vstack([np.bincount(s, minlength=d)/len(s) for s in streams])
        return rows, rows, embedding_radius(m, row_a), np.sqrt(np.log(2/entry_a)/(2*m))
    flip = np.random.default_rng(np.random.SeedSequence([int(seed), 606]))
    u, shift = flip.random((k, top)), flip.integers(1, k, (k, top))
    T = np.zeros((k, d)); landed = np.zeros(k)
    for j, s in enumerate(streams):
        noisy = np.where(u[j, :len(s)] < eta, (j+shift[j, :len(s)]) % k, j)
        np.add.at(T, (noisy, s), 1./len(s))
        landed += np.bincount(noisy, minlength=k)
    M = (1-eta)*np.eye(k)+eta/(k-1)*(np.ones((k, k))-np.eye(k))
    Minv = np.linalg.inv(M)
    kappa = float(np.abs(Minv).sum(1).max())
    # Composition term per row l: one source with p = 1-eta, k-1 with p = eta/(k-1).
    comp = lambda a: _bernstein(1-eta, m, a/k)+(k-1)*_bernstein(eta/(k-1), m, a/k)
    nl = landed.max()
    l2 = (1+np.sqrt(2*np.log(2/row_a)))*np.sqrt(nl)/m+comp(row_a/2)
    entry = np.sqrt(nl*np.log(4/entry_a)/2)/m+comp(entry_a/2)
    raw = Minv@T
    return np.vstack([_simplex_projection(r) for r in raw]), raw, kappa*l2, kappa*entry


def _simplex_projection(v):
    """Euclidean projection onto the probability simplex (Duchi et al. 2008)."""
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = np.nonzero(u*np.arange(1, len(v)+1) > css-1)[0][-1]
    return np.maximum(v-(css[rho]-1)/(rho+1), 0)


def _draw_without_replacement(pools, count, rng):
    """Balanced round-robin over the group's classes, each class pool permuted once."""
    perms = [p[rng.permutation(len(p))] for p in pools]
    take = min(-(-count//len(pools)), min(map(len, pools)))
    order = np.stack([q[:take] for q in perms], axis=1).ravel()
    return order[:count]


def _entropy(p):
    p = np.clip(p, 1e-12, 1)
    return -(p*np.log(p)).sum(-1)


def run_cell(ctx, n, m, seed, q_mode='oracle', n_boot=200, members=20, eta=0.):
    k, d = len(ctx['groups']), ctx['d']
    truth_w = true_weights(ctx, seed)
    log = _draw_log(ctx, max(GRIDS['n']+(n,)), seed, truth_w)[:n]
    a_known, centre, ref_radius, r = _reference(ctx, m, seed, eta)
    radii = [ref_radius]
    q_r = 0.
    if 'novel' in ctx:
        if q_mode == 'oracle':
            # Disjoint from deployment: the withheld classes' TRAIN reference images.
            pools = ctx['ref_cells']['novel']
            q = _group_hist(pools, d)
            q_r = embedding_radius(min(map(len, pools))*len(pools), ALPHA/(3*(k+1)))
            radii.append(q_r)
        else:
            q = np.ones(d)/d  # declared exactly: no estimation error to carry
        a = np.vstack([a_known, q])
    else:
        a = a_known
    ball = MMDBall.from_log(a, log, ALPHA/3, max(radii), _CTX.get('set_kind', SET_KIND))
    counts = np.bincount(log, minlength=d)
    emp = counts/n
    # Per-entry intervals for the label channel (union over k x d entries), centred on the
    # unprojected debiased entries when labels are noisy.
    lo, hi = np.clip(centre-r, 0, 1), np.clip(centre+r, 0, 1)
    # Baselines: EMQ + multinomial bootstrap at the estimated conditionals; bootstrap ensemble
    # over BOTH log and references.
    rng = np.random.default_rng(seed+7)
    emq_w = em_batch(a, rng.multinomial(n, emp, size=n_boot))
    ens_a = np.stack([np.vstack([rng.multinomial(m, row)/m for row in a_known]
                                + ([a[-1:]] if 'novel' in ctx else [])) for _ in range(members)])
    fits = [(j, em_batch(ens_a[j], rng.multinomial(n, emp)[None])) for j in range(members)]
    kept = [j for j, w in fits if len(w)]  # members whose resampled log is closed-set compatible
    ens_w = np.vstack([w for _, w in fits if len(w)]) if kept else np.empty((0, len(a)))
    ens_a = ens_a[kept]
    # Two-stage bootstrap baseline: resample the references AND the log, refit EMQ (no guarantee;
    # unlike log-only EMQ it does not ignore reference error by construction).
    two_a, two_w = [], []
    for _ in range(n_boot):
        a_b = np.vstack([rng.multinomial(m, row)/m for row in a_known] + ([a[-1:]] if 'novel' in ctx else []))
        w_b = em_batch(a_b, rng.multinomial(n, emp)[None])
        if len(w_b):
            two_a.append(a_b); two_w.append(w_b[0])
    two_a, two_w = np.array(two_a), np.array(two_w)
    # Constraint-region baseline after Li, Han & Ma (2026): per-cell Hoeffding intervals on the log
    # frequencies and per-entry intervals on the rows define a polytope of weights (no ball).
    t_cell = np.sqrt(np.log(2*d/(ALPHA/3))/(2*n))
    rows_lo = np.vstack([lo] + ([np.clip(a[-1:]-q_r, 0, 1)] if 'novel' in ctx else []))
    rows_hi = np.vstack([hi] + ([np.clip(a[-1:]+q_r, 0, 1)] if 'novel' in ctx else []))
    mac_ub = np.vstack([rows_lo.T, -rows_hi.T])
    mac_b = np.r_[emp+t_cell, -(emp-t_cell)]
    gap = min(np.linalg.norm(.4*(a_known[i]-a_known[j])) for i in range(k) for j in range(i+1, k))
    full = feature_match(a, emp)
    truth_a = ctx['truth_conditionals']
    shared = dict(arm=ctx.get('label', ctx['arm']), q_mode=q_mode if 'novel' in ctx else 'none', seed=int(seed),
                  n=int(n), m=int(m), eta=float(eta), delta=ctx['delta'],
                  prevalence=ctx['prevalence'], eps_n=ball.eps_n, radius=ball.radius,
                  ball_resolution=gap/ball.radius,
                  feasible=bool(full['residual'] <= ball.radius) if ball.set_kind == 'l2' else ball.feasible(),
                  weights_in_outer_set=ball.outer_covers(truth_w),
                  declared_family_residual=full['residual'],
                  incompatible=bool(full['residual'] > ball.radius))
    rows = []
    for cell in range(d):
        # The outlier cell is the LAST cell, not index k: with grouped sources k < d-1.
        region = 'outlier_cell' if cell == d-1 else 'on_support'
        g = ctx['cell_group'][cell]
        yhat = int(g) if g >= 0 else int(np.argmax(a_known[:, cell]))
        col = a[:, cell]
        known_den = col.copy()
        if 'novel' in ctx:
            known_den[-1] = 0
        num = np.zeros_like(col); num[yhat] = col[yhat]
        cls = ball.ratio_interval(num, num, known_den)
        joint_t = truth_w*truth_a[:, cell]
        known_t = joint_t[:k].sum()
        row = dict(shared, cell=cell, region=region, yhat=yhat,
                   true_cls=float(joint_t[yhat]/known_t) if known_t > 0 else np.nan,
                   cls_lower=cls['lower'], cls_upper=cls['upper'], W_cls=cls['upper']-cls['lower'],
                   cls_status=cls['status'])
        if 'novel' in ctx:
            num = np.zeros_like(col); num[-1] = col[-1]
            unk = ball.ratio_interval(num, num, col)
            row.update(unk_lower=unk['lower'], unk_upper=unk['upper'], W_unk=unk['upper']-unk['lower'],
                       true_unk=float(joint_t[-1]/joint_t.sum()) if joint_t.sum() > 0 else np.nan)
            # Joint novelty interval: also ranges over the novel entry (|q_hat(x) - q(x)| <=
            # ||q_hat - q||_2 <= q_r on the same event, so no extra failure budget) and over the
            # known entries' label-channel intervals. Plug-in W_unk stays as the attribution.
            q_lo, q_hi = max(col[-1]-q_r, 0.), min(col[-1]+q_r, 1.)
            n_lo = np.zeros_like(col); n_lo[-1] = q_lo
            n_hi = np.zeros_like(col); n_hi[-1] = q_hi
            d_lo = np.r_[hi[:, cell], q_lo]; d_hi = np.r_[lo[:, cell], q_hi]
            u_lo = ball.ratio_interval(n_lo, n_lo, d_lo) if q_lo > 0 else dict(lower=0.)
            u_hi = ball.ratio_interval(n_hi, n_hi, d_hi) if q_hi > 0 else dict(upper=0.)
            row.update(unk_joint_lower=u_lo['lower'], unk_joint_upper=u_hi['upper'],
                       W_unk_joint=u_hi['upper']-u_lo['lower'])
        # Fixed uniform anchor weights; interval over the label-channel conditional entries.
        others = np.arange(k) != yhat
        lab_lo = lo[yhat, cell]/max(lo[yhat, cell]+hi[others, cell].sum(), 1e-12)
        lab_hi = hi[yhat, cell]/max(hi[yhat, cell]+lo[others, cell].sum(), 1e-12)
        row['W_label'] = float(lab_hi-lab_lo)
        # Joint interval: weights over the ball AND known conditional entries over their label-
        # channel intervals. For fixed weights the extremes sit at interval endpoints, so each
        # side is one linear-fractional program. This, not W_cls alone, carries the coverage
        # claim once conditionals are estimated: W_cls plugs in the estimates.
        den_lo = np.zeros_like(col); den_hi = np.zeros_like(col)
        den_lo[:k] = np.where(others, hi[:, cell], lo[yhat, cell])
        den_hi[:k] = np.where(others, lo[:, cell], hi[yhat, cell])
        low = np.zeros_like(col); low[yhat] = lo[yhat, cell]
        high = np.zeros_like(col); high[yhat] = hi[yhat, cell]
        j_lo, j_hi = ball.ratio_interval(low, low, den_lo), ball.ratio_interval(high, high, den_hi)
        row.update(joint_lower=j_lo['lower'], joint_upper=j_hi['upper'],
                   W_joint=j_hi['upper']-j_lo['lower'])
        jw = emq_w*col[None]
        p = np.divide(jw[:, yhat], jw[:, :k].sum(1), out=np.full(len(jw), np.nan), where=jw[:, :k].sum(1) > 0)
        if len(p) and np.all(np.isfinite(p)):
            row['EMQ_cls_lower'], row['EMQ_cls_upper'] = np.quantile(p, [ALPHA/2, 1-ALPHA/2])
            row['EMQ_cls'] = row['EMQ_cls_upper']-row['EMQ_cls_lower']
        if len(two_w):
            jw2 = two_w*two_a[:, :, cell]
            kn = jw2[:, :k].sum(1)
            p2 = np.divide(jw2[:, yhat], kn, out=np.full(len(jw2), np.nan), where=kn > 0)
            if np.all(np.isfinite(p2)):
                row['EMQ2_lower'], row['EMQ2_upper'] = np.quantile(p2, [ALPHA/2, 1-ALPHA/2])
                row['EMQ2'] = row['EMQ2_upper']-row['EMQ2_lower']
        m_lo = lp_ratio(low, den_lo, mac_ub, mac_b, 'min')
        m_hi = lp_ratio(high, den_hi, mac_ub, mac_b, 'max')
        row.update(mac_lower=m_lo, mac_upper=m_hi, MAC=m_hi-m_lo)
        ej = ens_w*ens_a[:, :, cell]
        tot = ej.sum(1, keepdims=True)
        if len(ej) > 1 and np.all(tot > 0):
            probs = ej/tot
            row['Ens_MI'] = float(_entropy(probs.mean(0))-_entropy(probs).mean())
            row['Ens_std'] = float(probs[:, yhat].std(ddof=1))
        rows.append(row)
    return mark_coverage(pd.DataFrame(rows)).to_dict('records')


COVERED = (('cls', 'true_cls'), ('joint', 'true_cls'), ('unk', 'true_unk'), ('unk_joint', 'true_unk'),
           ('EMQ_cls', 'true_cls'), ('EMQ2', 'true_cls'), ('mac', 'true_cls'))


def lp_ratio(num, den, a_ub, b_ub, sense):
    """Extremum of num.w / den.w over {w in simplex : a_ub^T w <= b_ub} (Charnes-Cooper LP).
    Empty region -> NaN (abstain); a feasible w with zero denominator -> 0 or 1 (vacuous policy)."""
    from scipy.optimize import linprog
    kk = len(num)
    a_rows = a_ub  # constraint rows over weights, shape (constraints, kk)
    mass = linprog(den, A_ub=a_rows, b_ub=b_ub, A_eq=np.ones((1, kk)), b_eq=[1], bounds=(0, None), method='highs')
    if not mass.success:
        return np.nan
    if mass.fun <= 1e-10:
        return 0. if sense == 'min' else 1.
    aub = np.column_stack([a_rows, -b_ub])
    aeq = np.vstack([np.r_[np.ones(kk), -1.], np.r_[den, 0.]])
    sign = 1. if sense == 'min' else -1.
    res = linprog(np.r_[sign*np.asarray(num), 0.], A_ub=aub, b_ub=np.zeros(len(b_ub)), A_eq=aeq,
                  b_eq=[0, 1], bounds=(0, None), method='highs')
    return float(np.clip(sign*res.fun, 0, 1)) if res.success else np.nan


def mark_coverage(df, tol=SOLVER_TOL):
    """(Re)compute *_covers from stored interval endpoints and truths."""
    for name, truth in COVERED:
        if f'{name}_lower' not in df or truth not in df:
            continue
        lo, hi, t = df[f'{name}_lower'], df[f'{name}_upper'], df[truth]
        ok = np.isfinite(lo.astype(float)) & np.isfinite(t.astype(float))
        df[f'{name}_covers'] = np.where(ok, (lo-tol <= t) & (t <= hi+tol), np.nan)
        df[f'{name}_covers'] = df[f'{name}_covers'].astype(object).where(ok, None)
    return df


def summarise_coverage(df, summary):
    """All coverage tables, from rows. Used by main and by the post-hoc resummarise."""
    df = mark_coverage(df)
    sweeps = df[df.axis.isin(['n', 'm', 'eta', 'placebo'])]
    if len(sweeps):
        summary['n_axis_coverage'] = coverage(sweeps[sweeps.axis == 'n'], ['arm', 'q_mode', 'delta', 'n', 'region'])
        if (sweeps.axis == 'eta').any():
            summary['eta_axis_coverage'] = coverage(sweeps[sweeps.axis == 'eta'], ['arm', 'eta', 'region'])
    sev = df[df.axis == 'severity']
    if len(sev):
        summary['severity'] = coverage(sev, ['severity', 'region'])
        summary['severity_by_corruption'] = coverage(sev[sev.region == 'on_support'], ['corruption', 'severity'])
    summary['coverage_tolerance'] = SOLVER_TOL
    return summary


def _init(ctx):
    _CTX.clear(); _CTX.update(ctx)


def _run(task):
    axis, n, m, eta, seed, q_mode, extra = task
    return [dict(row, axis=axis, **extra) for row in run_cell(_CTX, n, m, seed, q_mode, eta=eta)]


def _pool_run(ctx, tasks, workers):
    with ProcessPoolExecutor(workers or os.cpu_count(), initializer=_init, initargs=(ctx,)) as pool:
        return [row for chunk in pool.map(_run, tasks, chunksize=8) for row in chunk]


def sweep_tasks(axes, seeds, q_mode='oracle'):
    tasks = []
    for axis in axes:
        for g in GRIDS[axis]:
            b = dict(BASE, **{axis: g})
            tasks += [(axis, b['n'], b['m'], b['eta'], s, q_mode, {}) for s in seeds]
    placebo = range(PLACEBO_OFFSET+seeds.start, PLACEBO_OFFSET+seeds.start+len(seeds)*max(map(len, GRIDS.values())))
    tasks += [('placebo', BASE['n'], BASE['m'], BASE['eta'], s, q_mode, {}) for s in placebo]
    return tasks


def real_matrix(df, components):
    tested = sum(1 for v in RE.EXPECT.values() if v not in (None, 'exact'))
    alpha = RE.FAMILY_ALPHA/max(tested, 1)
    out = []
    for (arm, q_mode, delta), adf in df.groupby(['arm', 'q_mode', 'delta']):
        placebo = adf[adf.axis == 'placebo']
        for axis in [a for a in GRIDS if a in adf.axis.unique()]:
            grid = GRIDS[axis]
            sweep = adf[adf.axis == axis]
            for comp in [c for c in components if c in adf]:
                for region in ('on_support', 'outlier_cell'):
                    # Average the probe widths within each seed (cells are not replicates).
                    per = sweep[sweep.region == region].groupby(['seed', axis])[comp].mean().unstack(axis)
                    widths = per.reindex(columns=list(grid)).to_numpy(dtype=float)
                    if not np.isfinite(widths).any():
                        continue
                    res = sl.signed_slope(grid, widths)
                    pool = placebo[placebo.region == region].groupby('seed')[comp].mean().to_numpy()
                    ok = res['status'] == 'ok' and np.isfinite(pool).sum() >= len(grid)*res['seeds']
                    null = sl.placebo_slopes(grid, pool, res['seeds']) if ok else np.array([])
                    dec = sl.decide(res, null, alpha)
                    expected = RE.EXPECT.get((arm, q_mode, round(float(delta), 3), axis, comp, region))
                    out.append(dict(arm=arm, q_mode=q_mode, delta=float(delta), axis=axis, component=comp,
                                    region=region, **res, **dec, expected=expected,
                                    agrees=sl.agrees(expected, dec['decision']),
                                    width_at_min=float(np.nanmean(widths[:, 0])),
                                    width_at_max=float(np.nanmean(widths[:, -1]))))
    return pd.DataFrame(out)


def coverage(df, keys):
    rows = []
    for key, block in df.groupby(keys):
        key = key if isinstance(key, tuple) else (key,)
        # One event per independent run: a run is (seed, corruption) when corruptions are pooled.
        run_keys = ['seed'] + [k for k in keys if k != 'region'] + (
            ['corruption'] if 'corruption' in block and 'corruption' not in keys else [])
        seeds = block.drop_duplicates(run_keys)
        row = dict(zip(keys, key), runs=int(len(seeds)),
                   weight_outer_coverage=float(seeds.weights_in_outer_set.mean()),
                   incompatible_rate=float(seeds.incompatible.mean()),
                   min_ball_resolution=float(seeds.ball_resolution.min()))
        for name in ('cls', 'joint', 'unk', 'unk_joint', 'EMQ_cls', 'EMQ2', 'mac'):
            col = block.get(f'{name}_covers')
            if col is not None and col.notna().any():
                answered = col.notna()
                row[f'{name}_coverage'] = float(col.dropna().astype(bool).mean())
                # Abstention (empty ball / region) is reported, never silently dropped.
                row[f'{name}_answer_rate'] = float(answered.mean())
                row[f'{name}_coverage_unconditional'] = float((col.fillna(False).astype(bool)).mean())
        for comp in ('W_cls', 'W_joint', 'W_label', 'W_unk', 'W_unk_joint', 'EMQ_cls', 'EMQ2', 'MAC'):
            if comp in block:
                row[f'mean_{comp}'] = float(block[comp].mean())
        rows.append(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='results/audit')
    parser.add_argument('--arms', nargs='+', default=['label_shift', 'held_out', 'corruption'])
    parser.add_argument('--seeds', type=int, default=50)
    parser.add_argument('--corruption-seeds', type=int, default=20)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--set', choices=('l2', 'box'), default=SET_KIND)
    parser.add_argument('--grouping', choices=tuple(GROUPINGS), default='classes')
    parser.add_argument('--prevalence', choices=('skew', 'dirichlet'), default='dirichlet')
    parser.add_argument('--dirichlet-alpha', type=float, default=1.)
    parser.add_argument('--deltas', type=float, nargs='+', default=[DELTA, 0.])
    parser.add_argument('--axes', nargs='+', default=['n', 'm', 'eta'])
    parser.add_argument('--dataset', choices=('cifar10', 'cifar100'), default='cifar10')
    parser.add_argument('--backbone', choices=('clip', 'dinov2'), default='clip')
    parser.add_argument('--classes', type=int, nargs='+', default=None,
                        help='class ids in play (default: the ten CIFAR-10 classes)')
    parser.add_argument('--novel', type=int, nargs='+', default=list(NOVEL))
    parser.add_argument('--label', default=None, help='arm name written to rows (held-out arm)')
    args = parser.parse_args(argv)
    shift = dict(set_kind=args.set, grouping=args.grouping, prevalence=args.prevalence,
                 dirichlet_alpha=args.dirichlet_alpha, dataset=args.dataset, backbone=args.backbone,
                 classes=args.classes)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    seeds = range(args.seeds)
    frames, meta = [], {}
    if 'label_shift' in args.arms:
        ctx = build_context('label_shift', **shift)
        meta['label_shift'] = dict(head_accuracy=ctx['head_accuracy'],
                                   example_weights=true_weights(ctx, 0).tolist())
        frames.append(pd.DataFrame(_pool_run(ctx, sweep_tasks(tuple(args.axes), seeds), args.workers)))
    if 'held_out' in args.arms:
        for delta in args.deltas:
            ctx = build_context('held_out', delta=delta, novel=tuple(args.novel), label=args.label, **shift)
            truth = ctx['truth_conditionals']
            meta[f'held_out_delta{delta:g}'] = dict(
                head_accuracy=ctx['head_accuracy'], novel=list(args.novel), delta=delta,
                known=[int(c) for c in ctx['known']],
                separation_s_true_q=hull_distance(truth[-1], truth[:-1]),
                separation_s_uniform_q=hull_distance(np.ones(ctx['d'])/ctx['d'], truth[:-1]))
            for q_mode in ('oracle', 'uniform'):
                frames.append(pd.DataFrame(_pool_run(ctx, sweep_tasks(('n',), seeds, q_mode), args.workers)))
    if 'corruption' in args.arms:
        meta['corruption'] = {}
        cseeds = range(args.corruption_seeds)
        clean = build_context('label_shift', **shift)
        rows = _pool_run(clean, [('severity', BASE['n'], BASE['m'], 0., s, 'oracle',
                                  dict(corruption='none', severity=0)) for s in cseeds], args.workers)
        for corruption in CORRUPTIONS:
            for severity in range(1, 6):
                ctx = build_context('corruption', corruption=corruption, severity=severity, **shift)
                meta['corruption'][f'{corruption}_{severity}'] = ctx['head_accuracy']
                rows += _pool_run(ctx, [('severity', BASE['n'], BASE['m'], 0., s, 'oracle',
                                         dict(corruption=corruption, severity=severity)) for s in cseeds],
                                  args.workers)
            print('done', corruption, flush=True)
        frames.append(pd.DataFrame(rows))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out/'rows.csv.gz', index=False)
    sweeps = df[df.axis.isin(['n', 'm', 'eta', 'placebo'])]
    import hashlib
    summary = dict(config=vars(args), base=BASE, grids=GRIDS, meta=meta, set_kind=args.set,
                   groups=[list(map(int, g)) for g in GROUPINGS[args.grouping]],
                   expectations_sha256=hashlib.sha256(Path(RE.__file__).read_bytes()).hexdigest())
    if len(sweeps):
        gate = sweeps[(sweeps.axis == 'n') & (sweeps.n == min(GRIDS['n']))]
        summary['ball_resolution_gate'] = dict(n=min(GRIDS['n']), min_ball_resolution=float(gate.ball_resolution.min()),
                                               passed=bool(gate.ball_resolution.min() > 1))
        matrix = real_matrix(sweeps, ('W_cls', 'W_label', 'W_joint', 'W_unk', 'W_unk_joint', 'EMQ_cls',
                                      'Ens_MI', 'Ens_std'))
        matrix.to_csv(out/'response_matrix.csv', index=False)
        tested = matrix[matrix.expected.notna() & (matrix.expected != 'exact')]
        summary.update(predicted_cells=int(len(tested)), agreeing=int(tested.agrees.sum()),
                       disagreeing=tested[~tested.agrees.astype(bool)][
                           ['arm', 'q_mode', 'delta', 'axis', 'component', 'region', 'expected',
                            'decision', 'slope', 'lower', 'upper', 'tau']].to_dict('records'))
    summarise_coverage(df, summary)
    (out/'summary.json').write_text(json.dumps(summary, indent=2, default=float))
    print('audit written to', out, '| gate:', summary.get('ball_resolution_gate'))


if __name__ == '__main__':
    main()
