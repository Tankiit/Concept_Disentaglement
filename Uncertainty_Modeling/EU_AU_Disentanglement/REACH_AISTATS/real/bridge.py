"""Real-data bridge: oracle-removal attribution and per-component dose-responses on CIFAR
(protocol: real/bridge_prereg.py).

    python -m real.bridge --part H_c10_clip --out results/bridge/H_c10_clip
    python -m real.bridge --analyse results/bridge

Parts: L_c10_<enc>, L_c100_<enc>, H_c10_<enc>, H_c100_<enc>, S_clip; enc in clip, dinov2, resnet50.
One row per (setting, seed, n, m, probe cell).
"""
import argparse, hashlib, json, os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from models.ball import MMDBall, embedding_radius, _programs
from models.baselines import em_batch, feature_match
import real.audit as A
import real.bridge_prereg as PR

ALPHA = A.ALPHA
TOL = A.SOLVER_TOL
_PROGS = {}
_CTXS = {}


# ---------------------------------------------------------------- tight joint set, with oracles
def _program(k, d):
    """Tight joint set (real/exchange.py) plus one optional linear pin fix.w = val (unseen oracle)."""
    if (k, d) not in _PROGS:
        import cvxpy as cp
        lo, hi = cp.Parameter((k, d), nonneg=True), cp.Parameter((k, d), nonneg=True)
        emp, eps = cp.Parameter(d, nonneg=True), cp.Parameter(nonneg=True)
        num, den = cp.Parameter(k, nonneg=True), cp.Parameter(k, nonneg=True)
        fix, val = cp.Parameter(k, nonneg=True), cp.Parameter(nonneg=True)
        u, t, sl = cp.Variable(k, nonneg=True), cp.Variable(nonneg=True), cp.Variable(d, nonneg=True)
        cons = [sl >= lo.T@u - t*emp, sl >= t*emp - hi.T@u, cp.norm(sl, 2) <= eps*t,
                cp.sum(u) == t, den@u == 1, fix@u == val*t]
        w, s2 = cp.Variable(k, nonneg=True), cp.Variable(d, nonneg=True)
        mcons = [s2 >= lo.T@w - emp, s2 >= emp - hi.T@w, cp.norm(s2, 2) <= eps, cp.sum(w) == 1, fix@w == val]
        _PROGS[(k, d)] = dict(lo=lo, hi=hi, emp=emp, eps=eps, num=num, den=den, fix=fix, val=val,
                              low=cp.Problem(cp.Minimize(num@u), cons),
                              high=cp.Problem(cp.Maximize(num@u), cons),
                              mass=cp.Problem(cp.Minimize(den@w), mcons))
    return _PROGS[(k, d)]


def tight(lo, hi, emp, eps, cell, yhat, pin=None):
    """[lower, upper] of w_yhat A_yhat(cell) / sum_j w_j A_j(cell) over ALL sources, over the tight joint
    set; pin = (index, value) fixes one weight. NaN = empty set (abstain)."""
    import cvxpy as cp
    k, d = lo.shape
    pr = _program(k, d)
    pr['lo'].value, pr['hi'].value = np.clip(lo, 0, 1), np.clip(hi, 0, 1)
    pr['emp'].value, pr['eps'].value = np.clip(emp, 0, None), max(eps, 1e-9)
    fix = np.zeros(k)
    if pin is not None:
        fix[pin[0]] = 1.
    pr['fix'].value, pr['val'].value = fix, (pin[1] if pin is not None else 0.)
    others = np.ones(k, bool); others[yhat] = False
    out = []
    for side in ('low', 'high'):
        a_num = lo[yhat, cell] if side == 'low' else hi[yhat, cell]
        if a_num <= 0:
            out.append(0. if side == 'low' else 1.); continue
        den = np.where(others, (hi if side == 'low' else lo)[:, cell], 0.); den[yhat] = a_num
        pr['den'].value = np.clip(den, 0, None)
        try:
            pr['mass'].solve(solver=cp.CLARABEL)
        except cp.SolverError:
            return np.nan, np.nan
        if pr['mass'].status not in ('optimal', 'optimal_inaccurate'):
            return np.nan, np.nan
        if pr['mass'].value <= 1e-10:
            out.append(0. if side == 'low' else 1.); continue
        num = np.zeros(k); num[yhat] = a_num
        pr['num'].value = num
        try:
            pr[side].solve(solver=cp.CLARABEL)
        except cp.SolverError:
            return np.nan, np.nan
        if pr[side].status not in ('optimal', 'optimal_inaccurate'):
            return np.nan, np.nan
        out.append(float(np.clip(pr[side].value, 0, 1)))
    return out[0], out[1]


def min_mass(ball, den):
    """min_{w in ball} den.w (Prop. B' constant)."""
    import cvxpy as cp
    k, d = ball.conditionals.shape
    pr = _programs(k, d)
    pr['at'].value, pr['emp'].value, pr['rad'].value, pr['den'].value = \
        ball.conditionals.T, ball.empirical, ball.radius, np.asarray(den, float)
    try:
        pr['mass'].solve(solver=cp.CLARABEL)
    except cp.SolverError:
        return np.nan
    return float(pr['mass'].value) if pr['mass'].status in ('optimal', 'optimal_inaccurate') else np.nan


def sigma(rows):
    """Smallest singular value of the embedding map on the simplex tangent space."""
    k = len(rows)
    if k < 2:
        return np.nan
    basis = np.linalg.qr(np.eye(k)-1/k)[0][:, :k-1]
    return float(np.linalg.svd(basis.T@rows, compute_uv=False).min())


# ---------------------------------------------------------------- settings
def _flat(groups):
    return tuple(sorted(c for g in groups for c in g))  # LDAHead.fit needs sorted class ids


def settings(part):
    kind, data, enc = part.split('_')  # e.g. H_c10_clip, S_c10_clip
    base = dict(set_kind='l2', prevalence='dirichlet', backbone=enc)
    out = {}
    if kind == 'L' and data == 'c10':
        for g in ('super4', 'classes'):
            out[f'L/c10/{enc}/{g}'] = dict(arm='label_shift', grouping=g, **base)
    elif kind == 'L' and data == 'c100':
        out[f'L/c100/{enc}/c100_2'] = dict(arm='label_shift', grouping='c100_2', dataset='cifar100',
                                           classes=_flat(A.GROUPINGS['c100_2']), **base)
    elif kind == 'H' and data == 'c10':
        for c in range(10):
            out[f'H/c10/{enc}/{c}'] = dict(arm='held_out', grouping='classes', novel=(c,), delta=PR.DELTA, **base)
    elif kind == 'H' and data == 'c100':
        for name, nov in PR.C100_NOVEL.items():
            out[f'H/c100/{enc}/{name}'] = dict(arm='held_out', grouping='c100_2', dataset='cifar100', novel=nov,
                                               classes=tuple(sorted(_flat(A.GROUPINGS['c100_2'])+nov)), delta=PR.DELTA, **base)
    elif kind == 'S':
        for corr in A.CORRUPTIONS:
            for sev in range(1, 6):
                out[f'S/c10/{enc}/{corr}/{sev}'] = dict(arm='corruption', corruption=corr, severity=sev,
                                                        grouping='super4', **base)
        out[f'S/c10/{enc}/cifar10_1/0'] = dict(arm='corruption', corruption='cifar10_1', severity=0,
                                               grouping='super4', **base)
    else:
        raise ValueError(part)
    return out


def context(name, spec):
    ctx = A.build_context(**spec)
    ctx['name'] = name
    if spec['arm'] == 'corruption':
        clean = A.build_context('label_shift', **{k: v for k, v in spec.items()
                                                  if k not in ('arm', 'corruption', 'severity')})
        ctx['clean_conditionals'] = clean['truth_conditionals']
        ctx['row_drift'] = float(np.linalg.norm(ctx['truth_conditionals']-clean['truth_conditionals'], axis=1).max())
    if spec['arm'] == 'held_out':
        t = ctx['truth_conditionals']
        ctx['s_true'] = A.hull_distance(t[-1], t[:-1])
    ctx['sigma_true'] = sigma(ctx['truth_conditionals'][:len(ctx['groups'])])
    return ctx


# ---------------------------------------------------------------- one episode
def episode(ctx, seed, n, m, members=20):
    k, d = len(ctx['groups']), ctx['d']
    novel = 'novel' in ctx
    kk = k+int(novel)
    truth_w = A.true_weights(ctx, seed)
    nmax = max(max(PR.BUDGETS_N), max(PR.SHIFT_N))
    log = A._draw_log(ctx, nmax, seed, truth_w)[:n]
    a_known, centre, ref_radius, r = A._reference(ctx, m, seed)
    emp = np.bincount(log, minlength=d)/n
    eps = embedding_radius(n, ALPHA/3)
    q_r = 0.
    lo, hi = np.clip(centre-r, 0, 1), np.clip(centre+r, 0, 1)
    if novel:
        pools = ctx['ref_cells']['novel']
        q = A._group_hist(pools, d)
        q_r = embedding_radius(min(map(len, pools))*len(pools), ALPHA/(3*(k+1)))
        a_hat = np.vstack([a_known, q])
        lo, hi = np.vstack([lo, np.clip(q-q_r, 0, 1)]), np.vstack([hi, np.clip(q+q_r, 0, 1)])
    else:
        a_hat = a_known
    truth_a = ctx['truth_conditionals']
    p_star = truth_w@truth_a
    clean_a = ctx.get('clean_conditionals', truth_a)
    ball = MMDBall(a_hat, emp, eps, max(ref_radius, q_r), 'l2')
    fit = feature_match(a_hat, emp)
    sig = sigma(a_known)
    # Ensemble MI (as in the audit): bootstrap the references and the log, refit EMQ.
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), n, m, 77]))
    ens_w, ens_a = [], []
    for _ in range(members):
        a_j = np.vstack([rng.multinomial(m, row)/m for row in a_known] + ([a_hat[-1:]] if novel else []))
        w_j = em_batch(a_j, rng.multinomial(n, emp)[None])
        if len(w_j):
            ens_w.append(w_j[0]); ens_a.append(a_j)
    ens_w, ens_a = np.array(ens_w), np.array(ens_a)
    shared = dict(setting=ctx['name'], arm=ctx['arm'], seed=int(seed), n=int(n), m=int(m), K=k,
                  eps=eps, r_entry=float(r), ref_radius=float(ref_radius), q_r=float(q_r), sigma_hat=sig,
                  sigma_true=ctx['sigma_true'], s_true=ctx.get('s_true', np.nan),
                  row_drift=ctx.get('row_drift', np.nan),
                  incompatible=bool(fit['residual'] > ball.radius), residual=float(fit['residual']),
                  radius=float(ball.radius))
    rows = []
    for cell in range(d):
        yhat = int(np.argmax(a_known[:, cell]))
        jt = truth_w*truth_a[:, cell]
        target = float(jt[yhat]/jt.sum()) if jt.sum() > 0 else np.nan
        row = dict(shared, cell=cell, outlier=cell == d-1, yhat=yhat, target=target)
        # Total certified interval and its oracle removals.
        tl, tu = tight(lo, hi, emp, eps, cell, yhat)
        row.update(I_lo=tl, I_hi=tu, W=tu-tl,
                   I_covers=bool(tl-TOL <= target <= tu+TOL) if np.isfinite(tl) and np.isfinite(target) else None)
        variants = dict(log=(lo, hi, p_star, 0., None), rows=(clean_a, clean_a, emp, eps, None))
        if novel:
            # Infinite labels pin every row, the unseen row included.
            variants['rows'] = (truth_a, truth_a, emp, eps, None)
            variants['unseen'] = (lo, hi, emp, eps, (k, truth_w[-1]))
        if ctx['arm'] == 'corruption':
            variants['shift'] = (truth_a, truth_a, emp, eps, None)
        for key, (l, h, e, ep, pin) in variants.items():
            vl, vu = tight(l, h, e, ep, cell, yhat, pin)
            row[f'W_minus_{key}'] = vu-vl
            row[f'D_{key}'] = row['W']-(vu-vl)
        # REACH widths (published plug-in definitions, real/audit.py).
        col = a_hat[:, cell]
        known_den = col.copy()
        if novel:
            known_den[-1] = 0
        num = np.zeros(kk); num[yhat] = col[yhat]
        cls = ball.ratio_interval(num, num, known_den)
        row.update(W_cls=cls['upper']-cls['lower'], cls_status=cls['status'])
        mm = min_mass(ball, known_den)
        if np.isfinite(sig) and sig > 0 and mm > 0:
            row['bprime'] = float(2*np.sqrt(k)*a_known[:, cell].max()/mm*2*ball.radius/sig)
        if novel:
            un = np.zeros(kk); un[-1] = col[-1]
            unk = ball.ratio_interval(un, un, col)
            row['W_unk'] = unk['upper']-unk['lower']
        others = np.arange(k) != yhat
        lab_lo = lo[yhat, cell]/max(lo[yhat, cell]+hi[:k][others, cell].sum(), 1e-12)
        lab_hi = hi[yhat, cell]/max(hi[yhat, cell]+lo[:k][others, cell].sum(), 1e-12)
        tk = truth_a[:k, cell]
        anch_true = float(tk[yhat]/tk.sum()) if tk.sum() > 0 else np.nan
        hk = a_known[:, cell]
        anch_hat = float(hk[yhat]/hk.sum()) if hk.sum() > 0 else np.nan
        row.update(W_label=float(lab_hi-lab_lo), lab_covers=bool(lab_lo-TOL <= anch_true <= lab_hi+TOL)
                   if np.isfinite(anch_true) else None, anchor_err=abs(anch_hat-anch_true))
        if len(ens_w) > 1:
            ej = ens_w*ens_a[:, :, cell]
            tot = ej.sum(1, keepdims=True)
            if np.all(tot > 0):
                p = np.clip(ej/tot, 1e-12, 1)
                row['Ens_MI'] = float(A._entropy(p.mean(0))-A._entropy(p).mean())
        rows.append(row)
    return rows


def _init(ctxs):
    if ctxs is not None:
        _CTXS.clear(); _CTXS.update(ctxs)


def _run(task):
    name, seed, n, m = task
    return episode(_CTXS[name], seed, n, m)


def run_part(part, out, seeds=None, workers=None, smoke=False):
    specs = settings(part)
    if smoke:
        specs = dict(list(specs.items())[:1])
    ctxs = {name: context(name, spec) for name, spec in specs.items()}
    shift = part.startswith('S')
    ns, ms = (PR.SHIFT_N, PR.SHIFT_M) if shift else (PR.BUDGETS_N, PR.BUDGETS_M)
    seeds = seeds or (PR.SHIFT_SEEDS if shift else PR.SEEDS)
    if smoke:
        ns, ms, seeds = ns[:1]+ns[-1:], ms[-1:], 2
    tasks = [(name, s, n, m) for name in ctxs for s in range(seeds) for n in ns for m in ms]
    _init(ctxs)
    import multiprocessing as mp
    fork = mp.get_start_method() == 'fork'
    with ProcessPoolExecutor(workers or os.cpu_count(), initializer=_init,
                             initargs=(None if fork else ctxs,)) as pool:
        rows = [r for chunk in pool.map(_run, tasks, chunksize=4) for r in chunk]
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'rows.csv.gz', index=False)
    meta = {name: dict(head_accuracy=c['head_accuracy'], sigma_true=c['sigma_true'],
                       s_true=c.get('s_true'), row_drift=c.get('row_drift'), K=len(c['groups']), d=c['d'])
            for name, c in ctxs.items()}
    (out/'meta.json').write_text(json.dumps(dict(
        part=part, prereg_sha256=hashlib.sha256(Path(PR.__file__).read_bytes()).hexdigest(),
        settings=meta, tasks=len(tasks)), indent=2, default=float))
    print('wrote', out, len(rows), 'rows', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part')
    parser.add_argument('--out')
    parser.add_argument('--seeds', type=int, default=None)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--analyse', default=None, help='folder holding part sub-folders')
    args = parser.parse_args(argv)
    if args.analyse:
        from real.bridge_analysis import analyse_all
        analyse_all(args.analyse)
        return
    run_part(args.part, args.out, args.seeds, args.workers, args.smoke)


if __name__ == '__main__':
    main()
