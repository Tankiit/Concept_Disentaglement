"""Gaussian-feature world through the real frozen head (protocol: real/gauss_dose_prereg.py, part G).

    python -m real.gauss --arm held_out --out results/gauss/held_out
    python -m real.gauss --arm shift --out results/gauss/shift
    python -m real.gauss --analyse results/gauss

Features x ~ N(mu_c, I) in R^64 are written as feature files and go through real/head.py and
real/audit.build_context exactly like CIFAR features; evaluation is real/bridge2.evaluate unchanged.
"""
import argparse, hashlib, json, os, tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import real.audit as A
import real.head as H
import real.bridge as B
import real.bridge2 as B2
import real.gauss_dose_prereg as PR

GROOT = Path(os.environ.get('REACH_GAUSS_DIR', Path(tempfile.gettempdir())/'reach_gauss'))
_orig_load = A.load
A.load = lambda name, root=None: _orig_load(name, GROOT if name.startswith('gauss') else (root or H.FEATURES))
_STATE = {}


def world_name(delta, lam, draw):
    return f'D{delta:g}_L{lam:g}_{draw}'


def write_world(delta, lam, draw, taus=(0.,), shift_dir='random'):
    """Write gauss/<world>_t<tau>_{train,test}.npz; train is identical across tau."""
    rng = np.random.default_rng(np.random.SeedSequence([int(delta*10), int(lam*10), draw, 3030]))
    U = rng.standard_normal((6, PR.G_DIM)); U /= np.linalg.norm(U, axis=1, keepdims=True)
    mu = [delta*U[k] for k in range(4)]
    mu.append(lam*delta*U[4]+(1-lam)*(mu[0]+mu[1])/2)
    mu = np.array(mu)
    y_tr = np.repeat(np.arange(5), PR.G_TRAIN)
    X_tr = (mu[y_tr]+rng.standard_normal((len(y_tr), PR.G_DIM))).astype(np.float32)
    y_te = np.repeat(np.arange(5), PR.G_TEST)
    Z_te = rng.standard_normal((len(y_te), PR.G_DIM))
    out = GROOT/'gauss'; out.mkdir(parents=True, exist_ok=True)
    names = []
    # EXPLORATORY option shift_dir='mean': shift along the class-0 mean direction (inside the head's
    # subspace); the registered design uses a random direction U[5].
    v = U[0] if shift_dir == 'mean' else U[5]
    for tau in taus:
        nm = f'{world_name(delta, lam, draw)}_{shift_dir[0]}t{tau:g}'
        X_te = (mu[y_te]+Z_te+tau*v).astype(np.float32)
        np.savez(out/f'{nm}_train.npz', X=X_tr, y=y_tr)
        np.savez(out/f'{nm}_test.npz', X=X_te, y=y_te)
        names.append(nm)
    return names


def contexts(arm, taus=PR.G_TAU, shift_dir='random'):
    base = dict(set_kind='l2', prevalence='dirichlet', backbone='clip', grouping='classes')
    ctxs = {}
    for delta in PR.G_DELTA:
        for draw in range(PR.G_DRAWS):
            if arm == 'held_out':
                for lam in PR.G_LAM:
                    nm = write_world(delta, lam, draw)[0]
                    key = f'G/held_out/{world_name(delta, lam, draw)}'
                    c = B.context(key, dict(arm='held_out', novel=(4,), delta=.1, classes=(0, 1, 2, 3, 4),
                                            dataset=f'gauss/{nm}', **base))
                    c.update(name=key, g_delta=delta, g_lam=lam)
                    ctxs[key] = c
            else:
                names = write_world(delta, 1., draw, taus, shift_dir)
                clean = None
                for tau, nm in zip(taus, names):
                    key = f'G/shift/{world_name(delta, 1., draw)}/t{tau:g}'
                    c = B.context(key, dict(arm='label_shift', classes=(0, 1, 2, 3), dataset=f'gauss/{nm}', **base))
                    if clean is None:
                        clean = c['truth_conditionals']
                    c.update(name=key, g_delta=delta, g_tau=tau, clean_conditionals=clean,
                             row_drift=float(np.linalg.norm(c['truth_conditionals']-clean, axis=1).max()))
                    ctxs[key] = c
    return ctxs


def _init(state):
    if state is not None:
        _STATE.clear(); _STATE.update(state)


def _run(task):
    name, seed, n, m = task
    ctx = _STATE[name]
    inp = B2.real_inputs(ctx, seed, n, m)
    extra = dict(setting=name, arm=ctx['arm'], seed=int(seed), s_true=ctx.get('s_true', np.nan),
                 sigma_true=ctx['sigma_true'], g_delta=ctx['g_delta'], g_lam=ctx.get('g_lam', np.nan),
                 g_tau=ctx.get('g_tau', np.nan), row_drift=ctx.get('row_drift', np.nan),
                 head_accuracy=ctx['head_accuracy'])
    return B2.evaluate(inp, list(range(ctx['d'])), extra)


def run(arm, out, workers=None, smoke=False, taus=PR.G_TAU, shift_dir='random'):
    import real.bridge_prereg as P1
    ctxs = contexts(arm, taus, shift_dir)
    if arm == 'held_out':
        seeds, ns, ms = PR.G_CAL+PR.G_EVAL, P1.BUDGETS_N, P1.BUDGETS_M
    else:
        seeds, ns, ms = PR.G_SHIFT_SEEDS, P1.SHIFT_N, P1.SHIFT_M
    tasks = [(k, s, n, m) for k in ctxs for s in seeds for n in ns for m in ms]
    if smoke:
        tasks = tasks[::max(1, len(tasks)//24)][:24]
    _init(ctxs)
    import multiprocessing as mp
    fork = mp.get_start_method() == 'fork'
    with ProcessPoolExecutor(workers or os.cpu_count(), initializer=_init, initargs=(None if fork else ctxs,)) as pool:
        rows = [r for chunk in pool.map(_run, tasks, chunksize=2) for r in chunk]
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'rows.csv.gz', index=False)
    meta = {k: dict(head_accuracy=c['head_accuracy'], sigma_true=c['sigma_true'], s_true=c.get('s_true'),
                    row_drift=c.get('row_drift')) for k, c in ctxs.items()}
    (out/'meta.json').write_text(json.dumps(dict(arm=arm, tasks=len(tasks), settings=meta,
        prereg_sha256=hashlib.sha256(Path(PR.__file__).read_bytes()).hexdigest()), indent=2, default=float))
    print('wrote', out, len(rows), flush=True)


def analyse(root, n_boot=2000):
    from scipy.stats import spearmanr
    import real.bridge2_analysis as BA
    from real.bridge_analysis import bootstrap_spearman
    root = Path(root)
    out = dict(prereg_sha256=hashlib.sha256(Path(PR.__file__).read_bytes()).hexdigest())
    ho = pd.read_csv(root/'held_out'/'rows.csv.gz')
    ho['cluster'] = ho.setting
    r, _ = BA.evaluate_block(ho, PR.G_CAL, PR.G_EVAL)
    r['passes_G1'] = r['contrasts']['REACH_pin - budget_majority']['ci95'][1] < 0
    out['G1'] = r
    out['G2_heldout_inclusion'] = float(ho.I_covers.dropna().astype(bool).mean())
    big = ho[(ho.n == 32000) & (ho.m == 2000)].assign(unit=lambda t: t.setting)
    out['G4'] = dict(delta_vs_Wcls=bootstrap_spearman(big[~big.cell.eq(big.cell.max())], 'g_delta', 'W_cls', n_boot),
                     s_vs_Wunk=bootstrap_spearman(big, 's_true', 'W_unk', n_boot),
                     by_world=big.groupby('setting').agg(delta=('g_delta', 'first'), lam=('g_lam', 'first'),
                                                         s=('s_true', 'first'), sigma=('sigma_true', 'first'),
                                                         head_acc=('head_accuracy', 'first'),
                                                         W_cls=('W_cls', 'mean'), W_unk=('W_unk', 'mean'),
                                                         W=('W', 'mean')).round(4).to_dict('index'))
    for k in ('delta_vs_Wcls', 's_vs_Wunk'):
        out['G4'][k]['passes'] = out['G4'][k]['ci95'][1] < 0
    sh = pd.read_csv(root/'shift'/'rows.csv.gz')
    unit = sh.groupby(['setting', 'n', 'm']).agg(
        tau=('g_tau', 'first'), drift=('row_drift', 'first'), radius=('radius', 'mean'),
        inclusion=('I_covers', lambda x: x.dropna().astype(bool).mean() if x.notna().any() else np.nan),
        answer=('I_covers', lambda x: x.notna().mean()), incompatible=('incompatible', 'mean')).reset_index()
    unit['rho'] = unit.drift/unit.radius
    ans = unit.dropna(subset=['inclusion'])
    out['G2_shift_tau0_inclusion'] = float(sh[sh.g_tau == 0].I_covers.dropna().astype(bool).mean())
    out['G3'] = dict(spearman_rho_inclusion=float(spearmanr(ans.rho, ans.inclusion).statistic),
                     spearman_rho_incompatible=float(spearmanr(unit.rho, unit.incompatible).statistic),
                     inclusion_rho_lt1=float(ans[ans.rho < 1].inclusion.mean()), units_rho_lt1=int((ans.rho < 1).sum()),
                     units=int(len(unit)), units_all_abstain=int(unit.inclusion.isna().sum()),
                     by_tau=unit.groupby('tau')[['rho', 'inclusion', 'answer', 'incompatible']].mean().round(4).to_dict('index'))
    g3 = out['G3']
    g3['passes'] = bool(g3['spearman_rho_inclusion'] < 0 and g3['spearman_rho_incompatible'] > 0
                        and g3['inclusion_rho_lt1'] >= .95)
    (root/'analysis.json').write_text(json.dumps(out, indent=2, default=float))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=('held_out', 'shift')); p.add_argument('--out')
    p.add_argument('--workers', type=int); p.add_argument('--smoke', action='store_true')
    p.add_argument('--analyse')
    p.add_argument('--taus', type=float, nargs='+', default=list(PR.G_TAU))
    p.add_argument('--shift-dir', default='random', choices=('random', 'mean'))
    a = p.parse_args(argv)
    if a.analyse:
        analyse(a.analyse); return
    run(a.arm, a.out, a.workers, a.smoke, tuple(a.taus), a.shift_dir)


if __name__ == '__main__':
    main()
