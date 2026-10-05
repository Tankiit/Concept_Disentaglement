"""Exchange-rate and certification-acquisition test (protocol: real/exchange_prereg.py).

    python -m real.exchange --setting synthetic --out results/exchange_synth
    python -m real.exchange --setting real --backbone resnet50 --out results/exchange_resnet50
"""
import argparse, hashlib, json, os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from models.ball import MMDBall, embedding_radius
from models.baselines import em_batch, feature_match
import real.exchange_prereg as PR

ALPHA = .05
_PROGS = {}
_CTX = {}


# ---------------------------------------------------------------- tight joint interval
def _program(k, d):
    """Charnes-Cooper cone program for the tight joint interval, compiled once per shape.

    Feasible weights: dist(p_hat, box(w)) <= eps, box(w)_c = [sum_j w_j lo_jc, sum_j w_j hi_jc].
    Homogenised with u = t w and slack s' = t s.
    """
    if (k, d) not in _PROGS:
        import cvxpy as cp
        lo, hi = cp.Parameter((k, d), nonneg=True), cp.Parameter((k, d), nonneg=True)
        emp, eps = cp.Parameter(d, nonneg=True), cp.Parameter(nonneg=True)
        num, den = cp.Parameter(k, nonneg=True), cp.Parameter(k, nonneg=True)
        u, t, sl = cp.Variable(k, nonneg=True), cp.Variable(nonneg=True), cp.Variable(d, nonneg=True)
        cons = [sl >= lo.T@u - t*emp, sl >= t*emp - hi.T@u, cp.norm(sl, 2) <= eps*t,
                cp.sum(u) == t, den@u == 1]
        w, s2 = cp.Variable(k, nonneg=True), cp.Variable(d, nonneg=True)
        mass_cons = [s2 >= lo.T@w - emp, s2 >= emp - hi.T@w, cp.norm(s2, 2) <= eps, cp.sum(w) == 1]
        _PROGS[(k, d)] = dict(lo=lo, hi=hi, emp=emp, eps=eps, num=num, den=den,
                              low=cp.Problem(cp.Minimize(num@u), cons),
                              high=cp.Problem(cp.Maximize(num@u), cons),
                              mass=cp.Problem(cp.Minimize(den@w), mass_cons))
    return _PROGS[(k, d)]


def tight_interval(lo, hi, emp, eps, cell, yhat, known):
    """[lower, upper] of P(Y=yhat | cell, known) over the tight joint set; NaN = empty (abstain)."""
    import cvxpy as cp
    k, d = lo.shape
    pr = _program(k, d)
    pr['lo'].value, pr['hi'].value = np.clip(lo, 0, 1), np.clip(hi, 0, 1)
    pr['emp'].value, pr['eps'].value = emp, eps
    others = np.zeros(k, bool); others[:known] = True; others[yhat] = False
    out = []
    for side in ('low', 'high'):
        a_num = lo[yhat, cell] if side == 'low' else hi[yhat, cell]
        den = np.zeros(k)
        den[yhat] = a_num
        den[others] = (hi if side == 'low' else lo)[others, cell]
        num = np.zeros(k); num[yhat] = a_num
        if a_num <= 0:
            out.append(0. if side == 'low' else 1.); continue
        pr['den'].value = np.clip(den, 0, None)
        try:
            pr['mass'].solve(solver=cp.CLARABEL)
        except cp.SolverError:
            return np.nan, np.nan
        if pr['mass'].status in ('infeasible', 'infeasible_inaccurate'):
            return np.nan, np.nan
        if pr['mass'].value is None or pr['mass'].value <= 1e-10:
            out.append(0. if side == 'low' else 1.); continue
        pr['num'].value = num
        try:
            pr[side].solve(solver=cp.CLARABEL)
        except cp.SolverError:
            return np.nan, np.nan
        if pr[side].status not in ('optimal', 'optimal_inaccurate'):
            return np.nan, np.nan
        out.append(float(np.clip(pr[side].value, 0, 1)))
    return out[0], out[1]


def mean_tight(lo, hi, emp, eps, cells, yhats, known):
    vals = [tight_interval(lo, hi, emp, eps, c, y, known) for c, y in zip(cells, yhats)]
    return vals


# ---------------------------------------------------------------- settings
def entry_half(m, k, d):
    return np.sqrt(np.log(2/(ALPHA/(3*k*d)))/(2*m))


def synthetic_context():
    from data.worlds import make_world
    w = make_world('separated')
    return dict(kind='synthetic', K=2, d=4, cells=[0, 1], q=w.conditionals[-1], rows=w.conditionals[:2],
                weights=w.weights, marginal=w.marginal, bases_m=PR.BASES_M_SYNTH)


def real_context(backbone):
    import real.audit as A
    ctx = A.build_context('label_shift', grouping='super4', prevalence='dirichlet', backbone=backbone)
    return dict(kind='real', K=len(ctx['groups']), d=ctx['d'], audit=ctx, cells=list(range(ctx['d']-1)),
                bases_m=PR.BASES_M_REAL, head_accuracy=ctx['head_accuracy'])


def draws(c, seed, n_max, m_max):
    """Prefix-paired log cells and per-source reference streams; truth weights and rows."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 9090]))
    K, d = c['K'], c['d']
    if c['kind'] == 'synthetic':
        log = rng.choice(d, n_max, p=c['marginal'])
        refs = [rng.choice(d, m_max, p=c['rows'][k]) for k in range(K)]
        return log, refs, c['weights'], np.vstack([c['rows'], c['q']])
    import real.audit as A
    ctx = c['audit']
    w = A.true_weights(ctx, seed)
    log = A._draw_log(ctx, n_max, seed, w)
    rr = np.random.default_rng(np.random.SeedSequence([int(seed), 303]))
    refs = [A._draw_without_replacement(ctx['ref_cells'][g], m_max, rr) for g in ctx['groups']]
    return log, refs, w, ctx['truth_conditionals']


def state(c, log, refs, n, m):
    K, d = c['K'], c['d']
    emp = np.bincount(log[:n], minlength=d)/n
    a_known = np.vstack([np.bincount(r[:m], minlength=d)/len(r[:m]) for r in refs])
    e = entry_half(m, K, d)
    lo, hi = np.clip(a_known-e, 0, 1), np.clip(a_known+e, 0, 1)
    if c['kind'] == 'synthetic':
        lo, hi = np.vstack([lo, c['q']]), np.vstack([hi, c['q']])
        a_hat = np.vstack([a_known, c['q']])
    else:
        a_hat = a_known
    return emp, a_hat, lo, hi


def widths(c, emp, a_hat, lo, hi, n, m, truth_w, truth_a):
    """Tight joint (primary), REACH joint v2, constraint region, plug-in W_cls; coverage of tight."""
    from real.audit import lp_ratio
    K, d = c['K'], c['d']
    k_all = a_hat.shape[0]
    eps = embedding_radius(n, ALPHA/3)
    r = embedding_radius(m, ALPHA/(3*(K+1)))
    ball = MMDBall(a_hat, emp, eps, r, 'l2')
    t_cell = np.sqrt(np.log(2*d/(ALPHA/3))/(2*n))
    mac_ub = np.vstack([lo.T, -hi.T]); mac_b = np.r_[emp+t_cell, -(emp-t_cell)]
    res = dict(tight=[], joint=[], mac=[], cls=[], tight_cov=[])
    for cell in c['cells']:
        yhat = int(np.argmax(a_hat[:K, cell]))
        jt = truth_w*truth_a[:, cell]
        truth = jt[yhat]/jt[:K].sum() if jt[:K].sum() > 0 else np.nan
        tl, tu = tight_interval(lo, hi, emp, eps, cell, yhat, K)
        res['tight'].append(tu-tl)
        res['tight_cov'].append(bool(tl-1e-6 <= truth <= tu+1e-6) if np.isfinite(tl) and np.isfinite(truth) else None)
        others = np.arange(k_all) != yhat; others[K:] = False
        den_lo = np.where(others, hi[:, cell], 0.); den_lo[yhat] = lo[yhat, cell]
        den_hi = np.where(others, lo[:, cell], 0.); den_hi[yhat] = hi[yhat, cell]
        low = np.zeros(k_all); low[yhat] = lo[yhat, cell]
        high = np.zeros(k_all); high[yhat] = hi[yhat, cell]
        jl, ju = ball.ratio_interval(low, low, den_lo), ball.ratio_interval(high, high, den_hi)
        res['joint'].append(ju['upper']-jl['lower'])
        res['mac'].append(lp_ratio(high, den_hi, mac_ub, mac_b, 'max')-lp_ratio(low, den_lo, mac_ub, mac_b, 'min'))
        num = np.zeros(k_all); num[yhat] = a_hat[yhat, cell]
        kd = a_hat[:, cell].copy(); kd[K:] = 0
        cl = ball.ratio_interval(num, num, kd)
        res['cls'].append(cl['upper']-cl['lower'])
    return res


def ensemble_mi(c, emp, a_hat, n, m, rng, members=20):
    K = c['K']
    ws, As = [], []
    for _ in range(members):
        a_j = np.vstack([rng.multinomial(m, row)/m for row in a_hat[:K]] + ([a_hat[K:]] if a_hat.shape[0] > K else []))
        w_j = em_batch(a_j, rng.multinomial(n, emp)[None])
        if len(w_j):
            ws.append(w_j[0]); As.append(a_j)
    if not ws:
        return np.nan  # every member's resampled log is closed-set incompatible: MI undefined
    ws, As = np.array(ws), np.array(As)
    vals = []
    for cell in c['cells']:
        joint = ws[:, :K]*As[:, :K, cell]
        tot = joint.sum(1, keepdims=True)
        if np.any(tot <= 0):
            continue
        p = np.clip(joint/tot, 1e-12, 1)
        hbar = -(p.mean(0)*np.log(p.mean(0))).sum()
        vals.append(hbar-(-(p*np.log(p)).sum(1)).mean())
    return float(np.mean(vals)) if vals else np.nan


def episode(c, seed, n0, m0):
    K, d = c['K'], c['d']
    max_lab = PR.BUDGET//(min(PR.PRICES)*K)
    log, refs, tw, ta = draws(c, seed, n0+PR.BUDGET, m0+max_lab)
    nm = lambda x: float(np.nanmean(x))
    emp0, a0, lo0, hi0 = state(c, log, refs, n0, m0)
    w0 = widths(c, emp0, a0, lo0, hi0, n0, m0, tw, ta)
    # Predictions: everything centred at the fitted mixture, current estimates held fixed.
    fit = feature_match(a0, emp0)
    proj = np.clip(np.asarray(fit['weights'])@a0, 0, None); proj /= proj.sum()
    eps0, eps1 = embedding_radius(n0, ALPHA/3), embedding_radius(n0+PR.BUDGET, ALPHA/3)
    yh = [int(np.argmax(a0[:K, cell])) for cell in c['cells']]
    def pred(eps, lo, hi):
        v = [tight_interval(lo, hi, proj, eps, cell, y, K) for cell, y in zip(c['cells'], yh)]
        return nm([u-l for l, u in v])
    p_base = pred(eps0, lo0, hi0)
    p_log = p_base-pred(eps1, lo0, hi0)
    # Realised log action.
    emp1, a1, lo1, hi1 = state(c, log, refs, n0+PR.BUDGET, m0)
    wl = widths(c, emp1, a1, lo1, hi1, n0+PR.BUDGET, m0, tw, ta)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 4242]))
    mi = ensemble_mi(c, emp0, a0, n0, m0, rng)
    r_of = lambda m: embedding_radius(m, ALPHA/(3*(K+1)))
    rows = []
    for price in PR.PRICES:
        dm = PR.BUDGET//(price*K)
        a_known = a0[:K]
        e1 = entry_half(m0+dm, K, d)
        lo_p, hi_p = lo0.copy(), hi0.copy()
        lo_p[:K], hi_p[:K] = np.clip(a_known-e1, 0, 1), np.clip(a_known+e1, 0, 1)
        p_lab = p_base-pred(eps0, lo_p, hi_p)
        emp2, a2, lo2, hi2 = state(c, log, refs, n0, m0+dm)
        wb = widths(c, emp2, a2, lo2, hi2, n0, m0+dm, tw, ta)
        row = dict(setting=c['kind'], seed=int(seed), n0=n0, m0=m0, price=price, dm=int(dm),
                   pred_log=p_log, pred_lab=p_lab, mi=mi,
                   radius_log=eps0-eps1, radius_lab=r_of(m0)-r_of(m0+dm))
        for key in ('tight', 'joint', 'mac', 'cls'):
            row[f'{key}_0'], row[f'{key}_log'], row[f'{key}_lab'] = nm(w0[key]), nm(wl[key]), nm(wb[key])
        for tag, w in (('0', w0), ('log', wl), ('lab', wb)):
            cov = [x for x in w['tight_cov'] if x is not None]
            row[f'tight_cov_{tag}'] = float(np.mean(cov)) if cov else np.nan
            row[f'tight_answer_{tag}'] = float(np.mean([x is not None for x in w['tight_cov']]))
        row['tight_le_joint_share'] = float(np.mean([a <= b+1e-6 for a, b in zip(w0['tight'], w0['joint'])
                                                     if np.isfinite(a) and np.isfinite(b)]))
        rows.append(row)
    return rows


def _init(c):
    _CTX.clear(); _CTX.update(c)


def _run(task):
    seed, n0, m0 = task
    return episode(_CTX, seed, n0, m0)


def analyse(df, n_boot=2000):
    df = df.copy()
    df['gain_log'] = df.tight_0-df.tight_log
    df['gain_lab'] = df.tight_0-df.tight_lab
    df['best'] = np.maximum(df.gain_log, df.gain_lab)
    pick = lambda choose_log: np.where(choose_log, df.gain_log, df.gain_lab)
    df['reg_reach'] = df.best-pick(df.pred_log >= df.pred_lab)
    df['reg_radius'] = df.best-pick(df.radius_log > df.radius_lab)
    df['reg_log'] = df.best-df.gain_log
    df['reg_lab'] = df.best-df.gain_lab
    cal, ev = df[df.seed.isin(PR.CALIBRATION_SEEDS)], df[df.seed.isin(PR.EVALUATION_SEEDS)].copy()
    cuts = np.nanquantile(cal.mi, [1/3, 2/3])
    # Undefined MI (no compatible ensemble member) is placed in the top tercile (most uncertain).
    tc, te = np.digitize(cal.mi.fillna(np.inf), cuts), np.digitize(ev.mi.fillna(np.inf), cuts)
    best_map = min(range(8), key=lambda code: np.nanmean(np.where(
        [(code >> t) & 1 for t in tc], cal.reg_log, cal.reg_lab)))
    ev['reg_mi'] = np.where([(best_map >> t) & 1 for t in te], ev.reg_log, ev.reg_lab)
    pols = ['reg_reach', 'reg_radius', 'reg_log', 'reg_lab', 'reg_mi']
    point = {p: float(ev[p].mean()) for p in pols}
    best_const = min(('reg_log', 'reg_lab'), key=lambda p: point[p])
    seeds = ev.seed.unique()
    by = {s: g for s, g in ev.groupby('seed')}
    rng = np.random.default_rng(0)
    pairs = [('reg_reach', best_const), ('reg_reach', 'reg_radius'), ('reg_reach', 'reg_mi')]
    draws_ = {p: [] for p in pairs}
    for _ in range(n_boot):
        f = pd.concat([by[s] for s in rng.choice(seeds, len(seeds))])
        for a, b in pairs:
            draws_[(a, b)].append(f[a].mean()-f[b].mean())
    ci = {f'{a} - {b}': [float(np.quantile(v, .025)), float(np.quantile(v, .975))] for (a, b), v in draws_.items()}
    cell = ev.groupby(['n0', 'm0', 'price']).agg(real=('gain_log', 'mean'), real_lab=('gain_lab', 'mean'),
                                                  pl=('pred_log', 'mean'), pb=('pred_lab', 'mean')).reset_index()
    cell['real_sign'] = np.sign(cell.real-cell.real_lab)
    cell['pred_sign'] = np.sign(cell.pl-cell.pb)
    e1 = float((cell.real_sign == cell.pred_sign).mean())
    return dict(
        E1=dict(share_sign_agree=e1, passes=e1 >= .75, cells=cell.round(5).to_dict('records')),
        E2=dict(regret=point, best_constant=best_const, ci95_seed_bootstrap=ci,
                passes=bool(ci[f'reg_reach - {best_const}'][1] < 0)),
        E3=dict(reach_minus_radius=ci['reg_reach - reg_radius']),
        E4=dict(tight_le_joint_share=float(df.tight_le_joint_share.mean()),
                mean_width=dict(tight=float(df.tight_0.mean()), joint=float(df.joint_0.mean()),
                                mac=float(df.mac_0.mean()), cls=float(df.cls_0.mean())),
                tight_coverage_min_by_base=float(df.groupby(['n0', 'm0']).tight_cov_0.mean().min()),
                tight_answer_rate=float(df.tight_answer_0.mean())),
        action_share_reach_log=float((ev.pred_log >= ev.pred_lab).mean()),
        optimal_share_log=float((ev.gain_log >= ev.gain_lab).mean()),
        mi_map=int(best_map), mi_cuts=cuts.tolist())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setting', choices=('synthetic', 'real'), required=True)
    parser.add_argument('--backbone', default='resnet50')
    parser.add_argument('--out', required=True)
    parser.add_argument('--seeds', type=int, default=PR.SEEDS)
    parser.add_argument('--workers', type=int, default=None)
    args = parser.parse_args(argv)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    c = synthetic_context() if args.setting == 'synthetic' else real_context(args.backbone)
    tasks = [(s, n0, m0) for s in range(args.seeds) for n0 in PR.BASES_N for m0 in c['bases_m']]
    with ProcessPoolExecutor(args.workers or os.cpu_count(), initializer=_init, initargs=(c,)) as pool:
        rows = [r for chunk in pool.map(_run, tasks, chunksize=2) for r in chunk]
    df = pd.DataFrame(rows)
    df.to_csv(out/'rows.csv.gz', index=False)
    summary = dict(setting=args.setting, backbone=args.backbone if args.setting == 'real' else None,
                   head_accuracy=c.get('head_accuracy'),
                   prereg_sha256=hashlib.sha256(Path(PR.__file__).read_bytes()).hexdigest(),
                   results=analyse(df))
    (out/'summary.json').write_text(json.dumps(summary, indent=2, default=float))
    r = summary['results']
    print(json.dumps(dict(E1=r['E1']['share_sign_agree'], E2=r['E2']['regret'], ci=r['E2']['ci95_seed_bootstrap'],
                          E3=r['E3'], E4={k: r['E4'][k] for k in ('tight_le_joint_share', 'mean_width',
                                                                     'tight_coverage_min_by_base')}), indent=1))


if __name__ == '__main__':
    main()
