"""Blind-spot test v2: the protocol registered in real/blindspot_prereg_v2.py, nothing else.

Reuses v1's pools, actions, payoffs and head (real/blindspot.py); changes the REACH rule (no NaN
fall-through, label gain priced for the estimator the action produces), adds prices, M0
sensitivity, a do-nothing action, constant / flag / regret-tuned-MI policies and a two-way
cluster bootstrap over seeds x shift families.

    python -m real.blindspot_v2 --out results/blindspot_v2
"""
import argparse, hashlib, itertools, json, os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from models.ball import MMDBall, embedding_radius
from models.baselines import em_batch, feature_match
import real.audit as A
import real.blindspot as V1
import real.blindspot_prereg_v2 as PR

K = V1.K
COMBOS = [(c, PR.M0_PRIMARY) for c in PR.PRICES] + [(50, m0) for m0 in PR.M0_SENSITIVITY]
_CTX = {}


def _widths(a, centre, emp, eps, ref_radius, r_entry):
    """Joint width per cell (v1 definition); NaN only if the ball itself is empty."""
    return V1._joint_widths(a, centre, emp, eps, ref_radius, r_entry)[0]


def episode(ctx, seed, members=20):
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 777]))
    pi = rng.dirichlet(np.ones(K))
    d = ctx['d']
    a_clean, centre, ref_radius, r_entry = A._reference(dict(groups=ctx['groups'], d=d,
                                                            ref_cells=ctx['ref_cells']), PR.M_CLEAN, seed)
    log_full, _ = V1._draw(ctx['pools']['log'], pi, PR.N0+PR.BUDGET, rng)
    counts0 = np.bincount(log_full[:PR.N0], minlength=d)
    counts1 = np.bincount(log_full, minlength=d)
    emp0 = counts0/PR.N0
    a_s = V1._smooth(a_clean)
    post0 = V1._posterior(a_s, counts0)
    if post0 is None:
        return []
    risk0, mass = V1._risk(post0, ctx, pi)
    pay_rw = risk0-V1._risk(V1._posterior(a_s, counts1), ctx, pi)[0]
    # Labels: one stream at the largest purchase, prefixes for cheaper ones (paired across prices).
    max_lab = PR.BUDGET//min(PR.PRICES)
    lab_cells, lab_src = V1._draw(ctx['pools']['label'], pi, max_lab, rng)
    row_a, entry_a = A.ALPHA/(3*(K+1)), A.ALPHA/(3*K*d)
    eps0 = embedding_radius(PR.N0, A.ALPHA/3)
    w_now = _widths(a_clean, centre, emp0, eps0, ref_radius, r_entry)
    fit = feature_match(a_clean, emp0)
    incompatible = bool(fit['residual'] > eps0+ref_radius)
    # g_log: ball centred at the fitted mixture with the larger-log radius (never empty).
    proj = np.clip(np.asarray(fit['weights'])@a_clean, 0, None); proj /= proj.sum()
    w_log = _widths(a_clean, centre, proj, embedding_radius(PR.N0+PR.BUDGET, A.ALPHA/3), ref_radius, r_entry)
    # Baseline: ensemble MI at S0 (same as v1).
    ens_w, ens_a = [], []
    for _ in range(members):
        a_j = V1._smooth(np.vstack([rng.multinomial(PR.M_CLEAN, row)/PR.M_CLEAN for row in a_clean]))
        w_j = em_batch(a_j, rng.multinomial(PR.N0, emp0)[None])
        if len(w_j):
            ens_w.append(w_j[0]); ens_a.append(a_j)
    ens_w, ens_a = np.array(ens_w), np.array(ens_a)
    mi = np.empty(d)
    for cell in range(d):
        joint = ens_w*ens_a[:, :, cell]
        probs = joint/joint.sum(1, keepdims=True)
        mi[cell] = A._entropy(probs.mean(0))-A._entropy(probs).mean()
    rows = []
    for price, m0 in COMBOS:
        n_lab = PR.BUDGET//price
        shifted = np.zeros((K, d)); np.add.at(shifted, (lab_src[:n_lab], lab_cells[:n_lab]), 1.)
        n_src = shifted.sum(1, keepdims=True)
        pay_lab = risk0-V1._risk(V1._posterior(V1._smooth((m0*a_clean+shifted)/(m0+n_src)), counts0), ctx, pi)[0]
        # g_lab: radius of the shrinkage estimator the action produces, under frozen conditionals.
        n_k = n_lab/K
        r_lab = (m0*ref_radius+n_k*embedding_radius(max(n_k, 1), row_a))/(m0+n_k)
        e_lab = (m0*r_entry+n_k*np.sqrt(np.log(2/entry_a)/(2*max(n_k, 1))))/(m0+n_k)
        w_lab = _widths(a_clean, centre, emp0, eps0, r_lab, e_lab)
        for cell in range(d):
            g_log = float(w_now[cell]-w_log[cell])
            g_lab = float(w_now[cell]-w_lab[cell])
            if incompatible or cell == d-1:
                regime, action = 'off_frame', 'add_labels'
            elif not (np.isfinite(g_log) and np.isfinite(g_lab)):
                raise RuntimeError('non-finite predicted gain in a compatible frame')  # v1 defect (a)
            elif max(g_log, g_lab) <= 0:
                regime, action = 'neither', 'do_nothing'
            elif g_log >= g_lab:
                regime, action = 'log_resolves', 'reweight'
            else:
                regime, action = 'label_limited', 'add_labels'
            rows.append(dict(split=ctx['name'], corruption=ctx['corruption'], severity=ctx['severity'],
                             seed=int(seed), cell=cell, price=price, m0=m0, mass=float(mass[cell]),
                             incompatible=incompatible, off_frame=bool(incompatible or cell == d-1),
                             regime=regime, reach_action=action, g_log=g_log, g_lab=g_lab,
                             W_joint=float(w_now[cell]), ens_mi=float(mi[cell]),
                             pay_reweight=float(pay_rw[cell]), pay_labels=float(pay_lab[cell])))
    return rows


def _init(head_refs):
    _CTX['head'], _CTX['refs'] = head_refs


def _run_split(spec):
    ctx = V1.build(*spec, _CTX['head'], _CTX['refs'])
    seeds = list(PR.CALIBRATION_SEEDS)+list(PR.EVALUATION_SEEDS)
    return [r for s in seeds for r in episode(ctx, s)]


ACTIONS = ('do_nothing', 'reweight', 'add_labels')


def _pay(frame, actions):
    return np.select([actions == 'add_labels', actions == 'reweight'],
                     [frame.pay_labels.to_numpy(), frame.pay_reweight.to_numpy()], 0.)


def _best(frame):
    return np.maximum(0., np.maximum(frame.pay_labels, frame.pay_reweight)).to_numpy()


def policies(cal, ev):
    """Every registered policy's action on the evaluation rows (MI maps tuned on calibration regret)."""
    cuts = np.quantile(cal.ens_mi, [1/3, 2/3])
    tc, te = np.digitize(cal.ens_mi, cuts), np.digitize(ev.ens_mi, cuts)
    def regret_of(frame, actions):
        return np.average(_best(frame)-_pay(frame, actions), weights=frame.mass)
    maps = list(itertools.product(ACTIONS, repeat=3))
    best_mi = min(maps, key=lambda mp: regret_of(cal, np.array(mp)[tc]))
    off_c, off_e = cal.off_frame.to_numpy(), ev.off_frame.to_numpy()
    best_mif = min(maps, key=lambda mp: regret_of(cal, np.where(off_c, 'add_labels', np.array(mp)[tc])))
    n = len(ev)
    out = {'REACH-v2': ev.reach_action.to_numpy(),
           'do_nothing': np.full(n, 'do_nothing'), 'reweight': np.full(n, 'reweight'),
           'add_labels': np.full(n, 'add_labels'),
           'flag_only': np.where(off_e, 'add_labels', 'reweight'),
           'flag_or_nothing': np.where(off_e, 'add_labels', 'do_nothing'),
           'MI_regret_tuned': np.array(best_mi)[te],
           'MI_plus_flag': np.where(off_e, 'add_labels', np.array(best_mif)[te])}
    return out, dict(tercile_cuts=cuts.tolist(), MI_map=list(best_mi), MI_flag_map=list(best_mif))


def cluster_bootstrap(ev, regrets, pairs, n_boot, seed=0):
    """Two-way cluster bootstrap over seeds x shift families on mass-weighted mean regret."""
    fam = np.where(ev.severity.to_numpy() > 0, ev.corruption.to_numpy(), ev.split.to_numpy())
    seeds, s_idx = np.unique(ev.seed.to_numpy(), return_inverse=True)
    fams, f_idx = np.unique(fam, return_inverse=True)
    S, F = len(seeds), len(fams)
    w = ev.mass.to_numpy()
    mass_cell = np.zeros((S, F)); np.add.at(mass_cell, (s_idx, f_idx), w)
    sums = {}
    for name, r in regrets.items():
        m = np.zeros((S, F)); np.add.at(m, (s_idx, f_idx), w*r); sums[name] = m
    rng = np.random.default_rng(seed)
    draws = {p: [] for p in pairs}
    for _ in range(n_boot):
        cs = np.bincount(rng.integers(0, S, S), minlength=S)
        cf = np.bincount(rng.integers(0, F, F), minlength=F)
        wt = np.outer(cs, cf)
        tot = (wt*mass_cell).sum()
        for a, b in pairs:
            draws[(a, b)].append(((wt*sums[a]).sum()-(wt*sums[b]).sum())/tot)
    return {f'{a} - {b}': [float(np.quantile(v, .025)), float(np.quantile(v, .975))]
            for (a, b), v in draws.items()}


def analyse(df):
    df = df[df.mass > 0]
    result = {}
    for (price, m0), block in df.groupby(['price', 'm0']):
        cal = block[block.seed.isin(PR.CALIBRATION_SEEDS)]
        ev = block[block.seed.isin(PR.EVALUATION_SEEDS)].reset_index(drop=True)
        acts, maps = policies(cal, ev)
        best = _best(ev)
        regrets = {k: best-_pay(ev, v) for k, v in acts.items()}
        point = {k: float(np.average(v, weights=ev.mass)) for k, v in regrets.items()}
        consts = ['do_nothing', 'reweight', 'add_labels']
        best_const = min(consts, key=lambda k: point[k])
        pairs = [('REACH-v2', best_const), ('REACH-v2', 'flag_only'), ('REACH-v2', 'MI_plus_flag'),
                 ('REACH-v2', 'MI_regret_tuned')]
        ci = cluster_bootstrap(ev, regrets, pairs, PR.CLUSTER_BOOTSTRAP)
        unf = ev[~ev.off_frame]
        D = (unf.pay_labels-unf.pay_reweight).to_numpy()
        lab = unf.reach_action.to_numpy() == 'add_labels'
        h1u = (float(np.average(D[lab], weights=unf.mass.to_numpy()[lab])-np.average(D[~lab], weights=unf.mass.to_numpy()[~lab]))
               if lab.any() and (~lab).any() else None)
        by_family = {}
        fam = np.where(ev.severity > 0, 'cifar10c', ev.split)
        for f in np.unique(fam):
            sel = fam == f
            by_family[f] = {k: float(np.average(v[sel], weights=ev.mass[sel])) for k, v in regrets.items()}
        result[f'c{price}_m0{m0}'] = dict(
            price=int(price), m0=int(m0), n_labels=int(PR.BUDGET//price), regret=point,
            best_constant=best_const, ci95_two_way=ci, H1u=h1u,
            H1u_note='unflagged cells; labels-minus-reweight where REACH-v2 recommends labels vs not',
            regime_share={k: float(v) for k, v in ev.groupby('regime').mass.sum().div(ev.mass.sum()).items()},
            action_share={k: float(v) for k, v in ev.groupby('reach_action').mass.sum().div(ev.mass.sum()).items()},
            regret_by_family=by_family, tuned_maps=maps)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='results/blindspot_v2')
    parser.add_argument('--splits', nargs='*', default=None)
    parser.add_argument('--workers', type=int, default=None)
    args = parser.parse_args(argv)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    specs = [s for s in V1.splits() if not args.splits or s[0] in args.splits]
    with ProcessPoolExecutor(args.workers or os.cpu_count(), initializer=_init,
                             initargs=(V1._head_and_refs(),)) as pool:
        rows = [r for chunk in pool.map(_run_split, specs) for r in chunk]
    df = pd.DataFrame(rows)
    df.to_csv(out/'rows.csv.gz', index=False)
    summary = dict(prereg_sha256=hashlib.sha256(Path(PR.__file__).read_bytes()).hexdigest(),
                   combos=COMBOS, results=analyse(df))
    (out/'summary.json').write_text(json.dumps(summary, indent=2, default=float))
    for key, r in summary['results'].items():
        print(key, 'best const', r['best_constant'], {k: round(v, 4) for k, v in r['regret'].items()})
        print('   CI', r['ci95_two_way'], 'H1u', r['H1u'])


if __name__ == '__main__':
    main()
