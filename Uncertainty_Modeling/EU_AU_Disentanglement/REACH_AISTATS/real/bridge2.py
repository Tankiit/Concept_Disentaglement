"""Bridge v2: oracle-free pinned-estimate attribution (REACH-pin), real and synthetic
(protocol: real/bridge2_prereg.py).

    python -m real.bridge2 --part H_c10_clip --out results/bridge2/H_c10_clip
    python -m real.bridge2 --part syn_K3_p1 --out results/bridge2/syn_K3_p1
    python -m real.bridge2 --analyse results/bridge2
"""
import argparse, hashlib, json, os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from models.ball import MMDBall, embedding_radius
from models.baselines import feature_match
import real.audit as A
import real.bridge as B
import real.bridge2_prereg as PR
import real.bridge2_amendment as AM

ALPHA, TOL = A.ALPHA, A.SOLVER_TOL
_NEAR = {}
_STATE = {}
_LOGS = {}


# ---------------------------------------------------------------- pinned estimates
def nearest_box(lo, hi, emp):
    """w0 = argmin_w dist(emp, box(w)) over the simplex, and x0 = the point of box(w0) nearest emp."""
    import cvxpy as cp
    k, d = lo.shape
    if (k, d) not in _NEAR:
        L, H, e = cp.Parameter((k, d), nonneg=True), cp.Parameter((k, d), nonneg=True), cp.Parameter(d, nonneg=True)
        w, sl = cp.Variable(k, nonneg=True), cp.Variable(d, nonneg=True)
        prob = cp.Problem(cp.Minimize(cp.norm(sl, 2)), [sl >= L.T@w - e, sl >= e - H.T@w, cp.sum(w) == 1])
        _NEAR[(k, d)] = (L, H, e, w, prob)
    L, H, e, w, prob = _NEAR[(k, d)]
    L.value, H.value, e.value = np.clip(lo, 0, 1), np.clip(hi, 0, 1), np.clip(emp, 0, None)
    try:
        prob.solve(solver=cp.CLARABEL)
    except cp.SolverError:
        return None, None
    if w.value is None:
        return None, None
    w0 = np.clip(w.value, 0, None); w0 /= w0.sum()
    x0 = np.clip(emp, w0@lo, w0@hi)
    return w0, x0


# ---------------------------------------------------------------- inputs: real
def real_inputs(ctx, seed, n, m):
    k, d = len(ctx['groups']), ctx['d']
    novel = 'novel' in ctx
    truth_w = A.true_weights(ctx, seed)
    key = (ctx['name'], seed)
    if key not in _LOGS:
        _LOGS.clear()
        _LOGS[key] = A._draw_log(ctx, PR.LOG_MAX, seed, truth_w)
    log = _LOGS[key][:n]
    a_known, centre, ref_radius, r = A._reference(ctx, m, seed)
    q, q_r = None, 0.
    if novel:
        pools = ctx['ref_cells']['novel']
        q = A._group_hist(pools, d)
        q_r = embedding_radius(min(map(len, pools))*len(pools), ALPHA/(3*(k+1)))
    return dict(k=k, d=d, emp=np.bincount(log, minlength=d)/n, a_known=a_known, centre=centre, r=float(r),
                ref_radius=float(ref_radius), q=q, q_r=float(q_r), truth_w=truth_w,
                truth_a=ctx['truth_conditionals'], n=n, m=m)


# ---------------------------------------------------------------- inputs: synthetic categorical worlds
def worlds(K, per):
    """Classifier worlds (real/bridge2_amendment.py): confusion-like rows plus an outlier cell."""
    d = K*per+1
    out = []
    for acc in AM.SYN_ACC:
        for lam in AM.SYN_LAM:
            for draw in range(AM.SYN_DRAWS):
                rng = np.random.default_rng(np.random.SeedSequence([K, per, int(acc*100), int(lam*10), draw, 2028]))
                rows = np.zeros((K, d))
                for k in range(K):
                    own = np.zeros(d); own[k*per:(k+1)*per] = rng.dirichlet(np.ones(per))
                    noise = rng.dirichlet(np.ones(d)); noise[-1] *= .2; noise /= noise.sum()
                    rows[k] = acc*own+(1-acc)*noise
                q = lam*(.5*np.eye(d)[-1]+.5*rng.dirichlet(np.ones(d)))+(1-lam)*(rng.dirichlet(np.ones(K))@rows)
                probes = np.arange(d) if d <= AM.SYN_PROBES else np.sort(rng.choice(d, AM.SYN_PROBES, replace=False))
                cfg = f'syn/K{K}/p{per}/a{acc:g}/l{lam:g}'
                out.append(dict(name=f'{cfg}/{draw}', config=cfg, wid=len(out), K=K, d=d, acc=acc, lam=lam,
                                rows=rows, q=q, probes=probes, s_true=A.hull_distance(q, rows),
                                sigma_true=B.sigma(rows), sigma_all=B.sigma(np.vstack([rows, q]))))
    return out


def synth_inputs(wd, seed, n, m):
    K, d = wd['K'], wd['d']
    ss = lambda tag: np.random.default_rng(np.random.SeedSequence([int(seed), tag, wd['wid'], K, d]))
    pi = ss(505).dirichlet(np.ones(K))
    truth_w = np.r_[(1-PR.DELTA)*pi, PR.DELTA]
    truth_a = np.vstack([wd['rows'], wd['q']])
    log = ss(101).choice(d, max(AM.SYN_N), p=truth_w@truth_a)[:n]
    rr = ss(303)
    refs = [rr.choice(d, max(AM.SYN_M), p=row)[:m] for row in wd['rows']]
    a_known = np.vstack([np.bincount(x, minlength=d)/m for x in refs])
    qd = ss(404).choice(d, PR.SYN_Q_DRAWS, p=wd['q'])
    return dict(k=K, d=d, emp=np.bincount(log, minlength=d)/n, a_known=a_known, centre=a_known,
                r=float(np.sqrt(np.log(2/(ALPHA/(3*K*d)))/(2*m))),
                ref_radius=embedding_radius(m, ALPHA/(3*(K+1))), q=np.bincount(qd, minlength=d)/PR.SYN_Q_DRAWS,
                q_r=embedding_radius(PR.SYN_Q_DRAWS, ALPHA/(3*(K+1))), truth_w=truth_w, truth_a=truth_a, n=n, m=m)


# ---------------------------------------------------------------- evaluation
def boxes(inp):
    k = inp['k']
    lo, hi = np.clip(inp['centre']-inp['r'], 0, 1), np.clip(inp['centre']+inp['r'], 0, 1)
    if inp['q'] is not None:
        q, q_r = inp['q'], inp['q_r']
        lo, hi = np.vstack([lo, np.clip(q-q_r, 0, 1)]), np.vstack([hi, np.clip(q+q_r, 0, 1)])
        a_hat = np.vstack([inp['a_known'], q])
    else:
        a_hat = inp['a_known']
    return lo, hi, a_hat


def total_width(inp, cells, yhats):
    lo, hi, _ = boxes(inp)
    eps = embedding_radius(inp['n'], ALPHA/3)
    out = []
    for c, y in zip(cells, yhats):
        l, u = B.tight(lo, hi, inp['emp'], eps, c, y)
        out.append(u-l)
    return np.array(out)


def evaluate(inp, cells, extra=None):
    k, d = inp['k'], inp['d']
    novel = inp['q'] is not None
    kk = k+int(novel)
    lo, hi, a_hat = boxes(inp)
    emp, truth_w, truth_a = inp['emp'], inp['truth_w'], inp['truth_a']
    eps = embedding_radius(inp['n'], ALPHA/3)
    p_star = truth_w@truth_a
    ball = MMDBall(a_hat, emp, eps, max(inp['ref_radius'], inp['q_r']), 'l2')
    fit = feature_match(a_hat, emp)
    w0, x0 = nearest_box(lo, hi, emp)
    shared = dict(extra or {}, n=int(inp['n']), m=int(inp['m']), K=k, eps=eps, r_entry=inp['r'],
                  radius=float(ball.radius), incompatible=bool(fit['residual'] > ball.radius),
                  residual=float(fit['residual']))
    rows = []
    for cell in cells:
        yhat = int(np.argmax(inp['a_known'][:, cell]))
        jt = truth_w*truth_a[:, cell]
        target = float(jt[yhat]/jt.sum()) if jt.sum() > 0 else np.nan
        row = dict(shared, cell=int(cell), yhat=yhat, target=target)
        tl, tu = B.tight(lo, hi, emp, eps, cell, yhat)
        row.update(W=tu-tl, I_covers=bool(tl-TOL <= target <= tu+TOL)
                   if np.isfinite(tl) and np.isfinite(target) else None)
        oracle = dict(log=(lo, hi, p_star, 0., None), rows=(truth_a, truth_a, emp, eps, None))
        pinned = dict(rows=(a_hat, a_hat, emp, max(eps, fit['residual']+1e-9), None))
        if w0 is not None:
            pinned['log'] = (lo, hi, x0, 0., None)
        if novel:
            oracle['unseen'] = (lo, hi, emp, eps, (k, truth_w[-1]))
            if w0 is not None:
                pinned['unseen'] = (lo, hi, emp, eps, (k, w0[-1]))
        for tag, variants in (('D', oracle), ('P', pinned)):
            for key, (l, h, e, ep, pin) in variants.items():
                vl, vu = B.tight(l, h, e, ep, cell, yhat, pin)
                row[f'{tag}_{key}'] = row['W']-(vu-vl)
        # v1 REACH widths (published plug-in definitions).
        col = a_hat[:, cell]
        known_den = col.copy()
        if novel:
            known_den[-1] = 0
        num = np.zeros(kk); num[yhat] = col[yhat]
        cls = ball.ratio_interval(num, num, known_den)
        row['W_cls'] = cls['upper']-cls['lower']
        if novel:
            un = np.zeros(kk); un[-1] = col[-1]
            unk = ball.ratio_interval(un, un, col)
            row['W_unk'] = unk['upper']-unk['lower']
        others = np.arange(k) != yhat
        lab_lo = lo[yhat, cell]/max(lo[yhat, cell]+hi[:k][others, cell].sum(), 1e-12)
        lab_hi = hi[yhat, cell]/max(hi[yhat, cell]+lo[:k][others, cell].sum(), 1e-12)
        row['W_label'] = float(lab_hi-lab_lo)
        rows.append(row)
    return rows


# ---------------------------------------------------------------- tasks
def _init(state):
    if state is not None:
        _STATE.clear(); _STATE.update(state)


def _run(task):
    kind, name, seed, n, m = task
    if kind == 'real':
        ctx = _STATE[name]
        inp = real_inputs(ctx, seed, n, m)
        cells = list(range(ctx['d']))
        extra = dict(setting=name, arm=ctx['arm'], seed=int(seed), s_true=ctx.get('s_true', np.nan),
                     sigma_true=ctx['sigma_true'])
        rows = evaluate(inp, cells, extra)
        if ctx['arm'] == 'label_shift' and name.startswith('L/c10') and m in PR.FINITE_M:
            yh = [r['yhat'] for r in rows]
            f = PR.FINITE_FACTOR
            w_log = total_width(real_inputs(ctx, seed, n*f, m), cells, yh)
            w_lab = total_width(real_inputs(ctx, seed, n, m*f), cells, yh)
            for r, a, b in zip(rows, w_log, w_lab):
                r.update(F_log=r['W']-a, F_rows=r['W']-b)
        return rows
    wd = _STATE[name]
    inp = synth_inputs(wd, seed, n, m)
    extra = dict(setting=wd['config'], world=wd['name'], arm='synthetic', seed=int(seed), s_true=wd['s_true'],
                 sigma_true=wd['sigma_true'], sigma_all=wd['sigma_all'], acc=wd['acc'], lam=wd['lam'])
    return evaluate(inp, wd['probes'], extra)


def run_part(part, out, workers=None, smoke=False):
    if part.startswith('syn'):
        _, K, per = part.split('_')  # e.g. syn_K3_p1
        state = {w['name']: w for w in worlds(int(K[1:]), int(per[1:]))}
        seeds = PR.SYN_CAL_SEEDS+PR.SYN_EVAL_SEEDS
        ns, ms = AM.SYN_N, AM.SYN_M
        tasks = [('syn', nm, s, n, m) for nm in state for s in seeds for n in ns for m in ms]
    else:
        specs = B.settings(part)
        state = {}
        for nm, spec in specs.items():
            state[nm] = B.context(nm, spec); state[nm]['name'] = nm
        seeds = PR.REAL_CAL_SEEDS+PR.REAL_EVAL_SEEDS
        tasks = [('real', nm, s, n, m) for nm in state for s in seeds for n in PR_REAL_N for m in PR_REAL_M]
    if smoke:
        tasks = tasks[::max(1, len(tasks)//24)][:24]
    _init(state)
    import multiprocessing as mp
    fork = mp.get_start_method() == 'fork'
    with ProcessPoolExecutor(workers or os.cpu_count(), initializer=_init, initargs=(None if fork else state,)) as pool:
        rows = [r for chunk in pool.map(_run, tasks, chunksize=2) for r in chunk]
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'rows.csv.gz', index=False)
    (out/'meta.json').write_text(json.dumps(dict(
        part=part, tasks=len(tasks), prereg_sha256=hashlib.sha256(Path(PR.__file__).read_bytes()).hexdigest()),
        indent=2))
    print('wrote', out, len(rows), 'rows', flush=True)


import real.bridge_prereg as _P1
PR_REAL_N, PR_REAL_M = _P1.BUDGETS_N, _P1.BUDGETS_M


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part'); parser.add_argument('--out')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--analyse', default=None)
    args = parser.parse_args(argv)
    if args.analyse:
        from real.bridge2_analysis import analyse_all
        analyse_all(args.analyse)
        return
    run_part(args.part, args.out, args.workers, args.smoke)


if __name__ == '__main__':
    main()
