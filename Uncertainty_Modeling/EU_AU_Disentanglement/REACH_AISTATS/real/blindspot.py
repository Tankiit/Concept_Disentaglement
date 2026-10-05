"""Natural-shift blind-spot test: does REACH's per-cell regime say which acquisition pays?

Protocol, claims and every constant are fixed in real/blindspot_prereg.py (hashed into the
manifest). This file implements that protocol and nothing else.

    python -m real.blindspot --out results/blindspot           # all shift splits, 20 seeds
    modal run real/modal_audit.py::blindspot                   # same, on Modal
"""
import argparse, hashlib, json, os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from models.ball import MMDBall, embedding_radius
from models.baselines import em_batch, feature_match
from real.head import LDAHead, load, CORRUPTIONS
import real.audit as A
import real.blindspot_prereg as PR

GROUPS = A.GROUPINGS['super4']
K = len(GROUPS)
SMOOTH = 1e-4   # floor on component entries so a posterior is never exactly 0 (NLL finite)
_CTX = {}


def splits():
    """(name, corruption, severity) for every registered shift split."""
    out = [('clean', 'none', 0), ('cifar10_1', 'cifar10_1', 0)]
    out += [(f'{c}_{s}', c, s) for c in CORRUPTIONS for s in range(1, 6)]
    return out


def _head_and_refs():
    train = load('cifar10_train')
    perm = np.random.default_rng(0).permutation(len(train['y']))
    dev, ref = perm[:10000], perm[10000:]
    head = LDAHead.fit(train['X'][dev], train['y'][dev], np.arange(10))
    ref_cells, ref_y = head.cell(train['X'][ref]), train['y'][ref]
    return head, {g: [ref_cells[ref_y == c] for c in g] for g in GROUPS}


def _shift_pool(name, corruption, severity):
    if corruption == 'none':
        data = load('cifar10_test'); X, y = data['X'], data['y']
    elif corruption == 'cifar10_1':
        data = load('cifar10_1_test'); X, y = data['X'], data['y']
    else:
        data = load(f'cifar10c_{corruption}'); mask = data['severity'] == severity
        X, y = data['X'][mask], data['y'][mask]
    return X, y


def build(name, corruption, severity, head, ref_cells):
    """Context for one shift split: disjoint log / label / eval pools, by source and class."""
    X, y = _shift_pool(name, corruption, severity)
    cells = head.cell(X)
    perm = np.random.default_rng(12345).permutation(len(y))
    cut1, cut2 = int(PR.POOLS['log']*len(y)), int((PR.POOLS['log']+PR.POOLS['label'])*len(y))
    parts = dict(log=perm[:cut1], label=perm[cut1:cut2], eval=perm[cut2:])
    group_of = np.full(10, -1)
    for gi, g in enumerate(GROUPS):
        group_of[list(g)] = gi
    pools = {part: {g: [cells[idx[y[idx] == c]] for c in g] for g in GROUPS} for part, idx in parts.items()}
    ev = parts['eval']
    return dict(name=name, corruption=corruption, severity=int(severity), d=head.cells,
                groups=list(GROUPS), ref_cells=ref_cells, pools=pools,
                eval_cells=cells[ev], eval_group=group_of[y[ev]],
                head_accuracy=float(np.mean(np.minimum(cells, 9) == y)))


def _draw(pools, pi, n, rng):
    """n draws: source by pi, class uniform within source, image uniform within class pool."""
    src = rng.choice(K, n, p=pi)
    cells = np.empty(n, dtype=int)
    for gi, g in enumerate(GROUPS):
        hit = src == gi
        cells[hit] = A._draw_from(pools[g], int(hit.sum()), rng)
    return cells, src


def _smooth(a):
    a = a + SMOOTH
    return a/a.sum(1, keepdims=True)


def _posterior(a, counts):
    """EMQ prevalence on the log, then P(source | cell) for every cell (sources x cells)."""
    w = em_batch(a, counts[None])
    if not len(w):
        return None
    joint = w[0][:, None]*a
    return joint/joint.sum(0, keepdims=True)


def _risk(post, ctx, pi):
    """Per-cell importance-weighted NLL of the true source on the eval pool, and eval mass."""
    cells, g = ctx['eval_cells'], ctx['eval_group']
    frac = np.bincount(g, minlength=K)/len(g)
    wt = pi[g]/frac[g]
    nll = -np.log(np.clip(post[g, cells], 1e-12, 1))
    d = ctx['d']
    mass = np.bincount(cells, weights=wt, minlength=d)
    risk = np.bincount(cells, weights=wt*nll, minlength=d)/np.maximum(mass, 1e-12)
    return risk, mass/mass.sum()


def _joint_widths(a, centre, emp, eps, ref_radius, r_entry, set_kind='l2'):
    """REACH joint width per cell at given radii (the S0 estimates held fixed)."""
    d = a.shape[1]
    ball = MMDBall(a, emp, eps, ref_radius, set_kind)
    lo, hi = np.clip(centre-r_entry, 0, 1), np.clip(centre+r_entry, 0, 1)
    out = np.full(d, np.nan)
    for cell in range(d):
        yhat = int(np.argmax(a[:, cell]))
        others = np.arange(K) != yhat
        den_lo = np.where(others, hi[:, cell], lo[yhat, cell])
        den_hi = np.where(others, lo[:, cell], hi[yhat, cell])
        low = np.zeros(K); low[yhat] = lo[yhat, cell]
        high = np.zeros(K); high[yhat] = hi[yhat, cell]
        j_lo = ball.ratio_interval(low, low, den_lo) if low[yhat] > 0 else dict(lower=0.)
        j_hi = ball.ratio_interval(high, high, den_hi) if high[yhat] > 0 else dict(upper=1.)
        out[cell] = j_hi['upper']-j_lo['lower']
    return out, ball


def episode(ctx, seed, members=20):
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 777]))
    pi = rng.dirichlet(np.ones(K))
    d = ctx['d']
    # S0: clean rows (same estimator and radii as the audit), log of n0 shifted images.
    a_clean, centre, ref_radius, r_entry = A._reference(dict(groups=ctx['groups'], d=d,
                                                            ref_cells=ctx['ref_cells']), PR.M_CLEAN, seed)
    log_full, _ = _draw(ctx['pools']['log'], pi, PR.N0+PR.BUDGET, rng)
    counts0 = np.bincount(log_full[:PR.N0], minlength=d)
    counts1 = np.bincount(log_full, minlength=d)
    emp0 = counts0/PR.N0
    a_s = _smooth(a_clean)
    post0 = _posterior(a_s, counts0)
    if post0 is None:
        return []
    # Action 1: reweight from the larger log.
    post_rw = _posterior(a_s, counts1)
    # Action 2: buy B/PRICE shifted labels; shrink the clean rows towards their shifted counts.
    n_lab = PR.BUDGET//PR.PRICE
    lab_cells, lab_src = _draw(ctx['pools']['label'], pi, n_lab, rng)
    shifted = np.zeros((K, d)); np.add.at(shifted, (lab_src, lab_cells), 1.)
    n_src = shifted.sum(1, keepdims=True)
    a_lab = _smooth((PR.M0*a_clean+shifted)/(PR.M0+n_src))
    post_lab = _posterior(a_lab, counts0)
    risk0, mass = _risk(post0, ctx, pi)
    pay_rw = risk0-_risk(post_rw, ctx, pi)[0]
    pay_lab = risk0-_risk(post_lab, ctx, pi)[0]
    # REACH regime from S0 only: incompatibility, then predicted joint-width reductions per action.
    row_a, entry_a = A.ALPHA/(3*(K+1)), A.ALPHA/(3*K*d)
    eps0 = embedding_radius(PR.N0, A.ALPHA/3)
    w0, ball0 = _joint_widths(a_clean, centre, emp0, eps0, ref_radius, r_entry)
    w_log, _ = _joint_widths(a_clean, centre, emp0, embedding_radius(PR.N0+PR.BUDGET, A.ALPHA/3),
                             ref_radius, r_entry)
    m_plus = PR.M_CLEAN+n_lab//K
    w_lab, _ = _joint_widths(a_clean, centre, emp0, eps0, embedding_radius(m_plus, row_a),
                             np.sqrt(np.log(2/entry_a)/(2*m_plus)))
    incompatible = bool(feature_match(a_clean, emp0)['residual'] > ball0.radius)
    # Baseline: ensemble MI at S0, members resample both the log and the references.
    ens_w, ens_a = [], []
    for _ in range(members):
        a_j = _smooth(np.vstack([rng.multinomial(PR.M_CLEAN, row)/PR.M_CLEAN for row in a_clean]))
        w_j = em_batch(a_j, rng.multinomial(PR.N0, emp0)[None])
        if len(w_j):
            ens_w.append(w_j[0]); ens_a.append(a_j)
    ens_w, ens_a = np.array(ens_w), np.array(ens_a)
    rows = []
    for cell in range(d):
        joint = ens_w*ens_a[:, :, cell]
        probs = joint/joint.sum(1, keepdims=True)
        mi = float(A._entropy(probs.mean(0))-A._entropy(probs).mean())
        d_log, d_lab = w0[cell]-w_log[cell], w0[cell]-w_lab[cell]
        if incompatible or cell == d-1:
            regime = 'off_frame'
        elif d_log > d_lab:
            regime = 'log_resolves'
        else:
            regime = 'label_limited'
        rows.append(dict(split=ctx['name'], corruption=ctx['corruption'], severity=ctx['severity'],
                         seed=int(seed), cell=cell, mass=float(mass[cell]), incompatible=incompatible,
                         regime=regime, reach_action='reweight' if regime == 'log_resolves' else 'add_labels',
                         W_joint=float(w0[cell]), pred_gain_log=float(d_log), pred_gain_label=float(d_lab),
                         ens_mi=mi, pay_reweight=float(pay_rw[cell]), pay_labels=float(pay_lab[cell]),
                         risk0=float(risk0[cell])))
    return rows


def _init(head_refs):
    _CTX['head'], _CTX['refs'] = head_refs


def _run_split(spec):
    name, corruption, severity = spec
    ctx = build(name, corruption, severity, _CTX['head'], _CTX['refs'])
    seeds = list(PR.CALIBRATION_SEEDS)+list(PR.EVALUATION_SEEDS)
    rows = [r for s in seeds for r in episode(ctx, s)]
    return rows, dict(split=name, head_accuracy=ctx['head_accuracy'])


def analyse(df, n_boot=2000, seed=0):
    """Pre-registered H1-H3. Units are (episode, cell) weighted by eval mass; the bootstrap
    resamples episodes (split, seed)."""
    df = df[df.mass > 0].copy()
    df['D'] = df.pay_labels-df.pay_reweight
    df['best'] = np.where(df.pay_labels > df.pay_reweight, 'add_labels', 'reweight')
    df['episode'] = df.split+'#'+df.seed.astype(str)
    cal = df[df.seed.isin(PR.CALIBRATION_SEEDS)]
    ev = df[df.seed.isin(PR.EVALUATION_SEEDS)].copy()
    cuts = np.quantile(cal.ens_mi, [1/3, 2/3])
    tercile = lambda x: np.digitize(x, cuts)
    cal = cal.assign(tercile=tercile(cal.ens_mi)); ev['tercile'] = tercile(ev.ens_mi)
    # Baseline: the tercile -> action map with the best weighted matching accuracy on calibration.
    best_map, best_acc = None, -1
    for code in range(2**3):
        amap = {t: ('add_labels' if code >> t & 1 else 'reweight') for t in range(3)}
        acc = np.average(cal.tercile.map(amap) == cal.best, weights=cal.mass)
        if acc > best_acc:
            best_map, best_acc = amap, acc
    ev['base_action'] = ev.tercile.map(best_map)
    pay = lambda frame, col: np.where(frame[col] == 'add_labels', frame.pay_labels, frame.pay_reweight)
    ev['regret_reach'] = np.maximum(ev.pay_labels, ev.pay_reweight)-pay(ev, 'reach_action')
    ev['regret_base'] = np.maximum(ev.pay_labels, ev.pay_reweight)-pay(ev, 'base_action')
    ev['match_reach'] = (ev.reach_action == ev.best).astype(float)
    ev['match_base'] = (ev.base_action == ev.best).astype(float)
    episodes = ev.episode.unique()
    rng = np.random.default_rng(seed)
    by_ep = {e: f for e, f in ev.groupby('episode')}

    def stat(frame):
        out = {}
        w = frame.mass
        out['regret_diff'] = np.average(frame.regret_reach-frame.regret_base, weights=w)
        out['match_reach'] = np.average(frame.match_reach, weights=w)
        out['match_base'] = np.average(frame.match_base, weights=w)
        for t in range(3):
            sub = frame[frame.tercile == t]
            lab = sub[sub.reach_action == 'add_labels']; rw = sub[sub.reach_action == 'reweight']
            out[f'H1_t{t}'] = (np.average(lab.D, weights=lab.mass) - np.average(rw.D, weights=rw.mass)
                               if len(lab) and len(rw) and lab.mass.sum() > 0 and rw.mass.sum() > 0 else np.nan)
        return out

    point = stat(ev)
    boots = [stat(pd.concat([by_ep[e] for e in rng.choice(episodes, len(episodes))])) for _ in range(n_boot)]
    ci = {k: [float(np.nanquantile([b[k] for b in boots], .025)), float(np.nanquantile([b[k] for b in boots], .975))]
          for k in point}
    sev = ev[ev.corruption.isin(CORRUPTIONS) | (ev.corruption == 'none')]
    by_sev = sev.groupby('severity').apply(lambda f: np.average(f.D, weights=f.mass))
    from scipy.stats import spearmanr
    h1_cells = [k for k in point if k.startswith('H1_')]
    supported_h1 = sum(1 for k in h1_cells if ci[k][0] > 0)
    table = (ev.groupby(['tercile', 'regime'])
               .apply(lambda f: pd.Series(dict(units=len(f), mass=f.mass.sum(),
                                               pay_reweight=np.average(f.pay_reweight, weights=f.mass),
                                               pay_labels=np.average(f.pay_labels, weights=f.mass),
                                               frac_labels_best=np.average(f.best == 'add_labels', weights=f.mass))))
               .reset_index())
    return dict(
        tercile_cuts=cuts.tolist(), baseline_map={int(k): v for k, v in best_map.items()},
        baseline_calibration_accuracy=float(best_acc),
        point={k: float(v) for k, v in point.items()}, ci95=ci,
        H1=dict(per_tercile={k: dict(estimate=float(point[k]), ci=ci[k]) for k in h1_cells},
                terciles_with_ci_above_zero=supported_h1, supported=supported_h1 >= 2),
        H2=dict(regret_reach_minus_base=dict(estimate=float(point['regret_diff']), ci=ci['regret_diff']),
                supported=bool(ci['regret_diff'][1] < 0),
                match_reach=float(point['match_reach']), match_base=float(point['match_base'])),
        H3=dict(D_by_severity={int(k): float(v) for k, v in by_sev.items()},
                spearman=float(spearmanr(by_sev.index, by_sev.values).correlation)),
        regime_shares=ev.groupby('regime').mass.sum().div(ev.mass.sum()).to_dict(),
        blind_spot_table=table.to_dict('records'),
        natural_shift=dict(
            cifar10_1=ev[ev.split == 'cifar10_1'].groupby('regime').apply(
                lambda f: dict(mass=float(f.mass.sum()), D=float(np.average(f.D, weights=f.mass)),
                               match_reach=float(np.average(f.match_reach, weights=f.mass)),
                               match_base=float(np.average(f.match_base, weights=f.mass)))).to_dict()),
        agreement_rate=float(np.average(ev.reach_action == ev.base_action, weights=ev.mass)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='results/blindspot')
    parser.add_argument('--splits', nargs='*', default=None, help='subset of split names (smoke tests)')
    parser.add_argument('--workers', type=int, default=None)
    args = parser.parse_args(argv)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    specs = [s for s in splits() if not args.splits or s[0] in args.splits]
    head_refs = _head_and_refs()
    with ProcessPoolExecutor(args.workers or os.cpu_count(), initializer=_init, initargs=(head_refs,)) as pool:
        results = list(pool.map(_run_split, specs))
    df = pd.DataFrame([r for rows, _ in results for r in rows])
    df.to_csv(out/'rows.csv.gz', index=False)
    summary = analyse(df)
    summary['head_accuracy'] = {m['split']: m['head_accuracy'] for _, m in results}
    summary['prereg_sha256'] = hashlib.sha256(Path(PR.__file__).read_bytes()).hexdigest()
    (out/'summary.json').write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(dict(H1=summary['H1'], H2=summary['H2'], H3=summary['H3'],
                          regimes=summary['regime_shares'], agreement=summary['agreement_rate']), indent=1, default=float))


if __name__ == '__main__':
    main()
