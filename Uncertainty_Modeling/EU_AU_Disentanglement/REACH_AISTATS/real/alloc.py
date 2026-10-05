"""Class-wise label allocation by per-source pinning (protocol: real/alloc_prereg.py).

    python -m real.alloc --part c10_clip --out results/alloc/c10_clip
    python -m real.alloc --part syn_K3_p1 --out results/alloc/syn_K3_p1
    python -m real.alloc --analyse results/alloc
"""
import argparse, hashlib, json, os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from models.ball import embedding_radius
from models.baselines import feature_match
import real.audit as A
import real.bridge as B
import real.bridge2 as B2
import real.alloc_prereg as PR

ALPHA = A.ALPHA
_STATE = {}
RULES = ('uniform', 'proportional', 'reach', 'oracle')


# ---------------------------------------------------------------- per-source reference counts
def entry_half(m, k, d):
    return np.sqrt(np.log(2/(ALPHA/(3*k*d)))/(2*np.maximum(m, 1)))


class Episode:
    """Holds the log, the prefix-paired reference streams and the truth for one (setting, seed, n)."""

    def __init__(self, kind, state, seed, n):
        self.kind = kind
        if kind == 'real':
            ctx = state
            self.K, self.d = len(ctx['groups']), ctx['d']
            self.truth_w = A.true_weights(ctx, seed)
            self.truth_a = ctx['truth_conditionals']
            log = A._draw_log(ctx, max(PR.REAL_N), seed, self.truth_w)[:n]
            rng = np.random.default_rng(np.random.SeedSequence([int(seed), 303]))
            self.streams = [A._draw_without_replacement(ctx['ref_cells'][g], PR.REAL_CAP, rng) for g in ctx['groups']]
            self.q = None
            self.probes = list(range(self.d))
        else:
            wd = state
            self.K, self.d = wd['K'], wd['d']
            ss = lambda tag: np.random.default_rng(np.random.SeedSequence([int(seed), tag, wd['wid'], wd['K'], wd['d']]))
            pi = ss(505).dirichlet(np.ones(self.K))
            self.truth_w = np.r_[.9*pi, .1]
            self.truth_a = np.vstack([wd['rows'], wd['q']])
            log = ss(101).choice(self.d, max(PR.SYN_N), p=self.truth_w@self.truth_a)[:n]
            rr = ss(303)
            self.streams = [rr.choice(self.d, PR.SYN_CAP, p=row) for row in wd['rows']]
            qd = ss(404).choice(self.d, 2000, p=wd['q'])
            self.q = np.bincount(qd, minlength=self.d)/2000
            self.q_r = embedding_radius(2000, ALPHA/(3*(self.K+1)))
            self.probes = list(wd['probes'])
        self.n = n
        self.emp = np.bincount(log, minlength=self.d)/n
        self.eps = embedding_radius(n, ALPHA/3)
        self.cap = np.array([len(s) for s in self.streams])

    def state(self, m):
        """A_hat (all rows) and entry boxes for per-source counts m (vector)."""
        m = np.minimum(np.asarray(m, int), self.cap)
        a_known = np.vstack([np.bincount(s[:mk], minlength=self.d)/mk for s, mk in zip(self.streams, m)])
        r = entry_half(m, self.K, self.d)[:, None]
        lo, hi = np.clip(a_known-r, 0, 1), np.clip(a_known+r, 0, 1)
        if self.q is not None:
            lo = np.vstack([lo, np.clip(self.q-self.q_r, 0, 1)]); hi = np.vstack([hi, np.clip(self.q+self.q_r, 0, 1)])
            a_hat = np.vstack([a_known, self.q])
        else:
            a_hat = a_known
        return a_hat, lo, hi

    def width(self, lo, hi, yhats):
        out = []
        for c, y in zip(self.probes, yhats):
            l, u = B.tight(lo, hi, self.emp, self.eps, c, y)
            out.append(u-l if np.isfinite(l) else np.nan)
        return np.array(out)

    def prevalence_error(self, a_hat):
        w = np.asarray(feature_match(a_hat, self.emp)['weights'], float)[:self.K]
        w = w/w.sum() if w.sum() > 0 else np.full(self.K, 1/self.K)
        t = self.truth_w[:self.K]/self.truth_w[:self.K].sum()
        return float(np.abs(w-t).sum())


def allocate(weights, budget, m0, cap):
    """Integer allocation proportional to weights, capped at cap - m0 per source, excess redistributed."""
    k = len(weights)
    w = np.maximum(np.asarray(weights, float), 0)
    if w.sum() <= 0:
        w = np.ones(k)
    room = np.maximum(cap-m0, 0).astype(float)
    b = np.zeros(k)
    left = float(budget)
    free = room > 0
    for _ in range(k+1):
        if left <= 1e-9 or not free.any():
            break
        ww = np.where(free, w, 0)
        if ww.sum() <= 0:
            ww = free.astype(float)
        add = left*ww/ww.sum()
        add = np.minimum(add, room-b)
        b += add
        left = budget-b.sum()
        free = (room-b) > 1e-9
    b = np.floor(b).astype(int)
    rest = int(budget-b.sum())
    order = np.argsort(-(np.where((room-b) > 0, w, -1)))
    for i in order[:max(rest, 0)]:
        if b[i] < room[i]:
            b[i] += 1
    return b


def run_episode(kind, name, seed, n, m0):
    st = _STATE[name]
    ep = Episode(kind, st, seed, n)
    K = ep.K
    m_base = np.full(K, m0)
    a_hat, lo, hi = ep.state(m_base)
    yhats = [int(np.argmax(a_hat[:K, c])) for c in ep.probes]
    W0 = ep.width(lo, hi, yhats)
    gains = {'reach': [], 'oracle': []}
    for k in range(K):
        for tag, row in (('reach', a_hat[k]), ('oracle', ep.truth_a[k])):
            l2, h2 = lo.copy(), hi.copy()
            l2[k], h2[k] = row, row
            gains[tag].append(float(np.nanmean(W0-ep.width(l2, h2, yhats))))
    w_fit = np.asarray(feature_match(a_hat, ep.emp)['weights'], float)[:K]
    budget = K*m0
    weights = dict(uniform=np.ones(K), proportional=w_fit, reach=np.array(gains['reach']),
                   oracle=np.array(gains['oracle']))
    row = dict(setting=name, kind=kind, seed=int(seed), n=int(n), m0=int(m0), K=K, budget=budget,
               W0=float(np.nanmean(W0)), err0=ep.prevalence_error(a_hat),
               vacuous0=float(np.nanmean(W0 >= .99)),
               gain_reach=json.dumps(np.round(gains['reach'], 6).tolist()),
               gain_oracle=json.dumps(np.round(gains['oracle'], 6).tolist()),
               w_true=json.dumps(np.round(ep.truth_w[:K], 6).tolist()))
    for rule in RULES:
        b = allocate(weights[rule], budget, m_base, ep.cap)
        a1, lo1, hi1 = ep.state(m_base+b)
        W1 = ep.width(lo1, hi1, yhats)
        row.update({f'W_{rule}': float(np.nanmean(W1)), f'err_{rule}': ep.prevalence_error(a1),
                    f'alloc_{rule}': json.dumps(b.tolist()),
                    f'abstain_{rule}': float(np.mean(~np.isfinite(W1)))})
    return row


def _init(state):
    if state is not None:
        _STATE.clear(); _STATE.update(state)


def _run(task):
    return run_episode(*task)


def build(part):
    if part.startswith('syn'):
        _, K, per = part.split('_')
        state = {w['name']: w for w in B2.worlds(int(K[1:]), int(per[1:])) if w['lam'] == PR.SYN_LAM}
        tasks = [('syn', nm, s, n, m0) for nm in state for s in PR.SYN_SEEDS for n in PR.SYN_N for m0 in PR.SYN_M0]
    else:
        data, enc = part.split('_')
        specs = B.settings(f'L_{data}_{enc}')
        state = {}
        for nm, spec in specs.items():
            state[nm] = B.context(nm, spec); state[nm]['name'] = nm
        tasks = [('real', nm, s, n, m0) for nm in state for s in PR.REAL_SEEDS for n in PR.REAL_N for m0 in PR.REAL_M0]
    return state, tasks


def run_part(part, out, workers=None, smoke=False):
    state, tasks = build(part)
    if smoke:
        # Smoke runs use seed 0 only, which is not an evaluation seed of this test.
        tasks = [(t[0], t[1], 0, t[3], t[4]) for t in tasks[:4]]
    _init(state)
    import multiprocessing as mp
    fork = mp.get_start_method() == 'fork'
    with ProcessPoolExecutor(workers or os.cpu_count(), initializer=_init, initargs=(None if fork else state,)) as pool:
        rows = list(pool.map(_run, tasks, chunksize=1))
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'rows.csv.gz', index=False)
    (out/'meta.json').write_text(json.dumps(dict(part=part, tasks=len(tasks), smoke=smoke,
        prereg_sha256=hashlib.sha256(Path(PR.__file__).read_bytes()).hexdigest()), indent=2))
    print('wrote', out, len(rows), flush=True)


def boot(df, col_a, col_b, cluster, n_boot=PR.N_BOOT, seed=0):
    d = df.assign(diff=df[col_a]-df[col_b])
    sums = d.groupby(['seed', cluster])['diff'].agg(['sum', 'count'])
    si = sums.index.get_level_values(0).to_numpy(); ci = sums.index.get_level_values(1).to_numpy()
    us, uc = np.unique(si), np.unique(ci)
    si = np.searchsorted(us, si); ci = np.searchsorted(uc, ci)
    S, C = sums['sum'].to_numpy(), sums['count'].to_numpy()
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        ws = np.bincount(rng.integers(0, len(us), len(us)), minlength=len(us))
        wc = np.bincount(rng.integers(0, len(uc), len(uc)), minlength=len(uc))
        w = ws[si]*wc[ci]
        draws.append((w*S).sum()/max((w*C).sum(), 1))
    return dict(point=float(d['diff'].mean()), ci95=[float(np.quantile(draws, .025)), float(np.quantile(draws, .975))])


def summarise(df, cluster):
    out = dict(episodes=int(len(df)), mean={c: float(df[c].mean()) for c in df if c.startswith(('W_', 'err_'))},
               W0=float(df.W0.mean()), err0=float(df.err0.mean()), vacuous0=float(df.vacuous0.mean()))
    out['O1'] = {f'reach - {b}': boot(df, 'W_reach', f'W_{b}', cluster) for b in ('uniform', 'proportional', 'oracle')}
    out['O2'] = {f'reach - {b}': boot(df, 'err_reach', f'err_{b}', cluster) for b in ('uniform', 'proportional', 'oracle')}
    out['by_budget'] = {f'n={n},m0={m}': dict(
        O1_reach_minus_uniform=float((g.W_reach-g.W_uniform).mean()),
        O1_reach_minus_prop=float((g.W_reach-g.W_proportional).mean()),
        O2_reach_minus_uniform=float((g.err_reach-g.err_uniform).mean()),
        W0=float(g.W0.mean()), vacuous0=float(g.vacuous0.mean()))
        for (n, m), g in df.groupby(['n', 'm0'])}
    return out


def analyse(root):
    root = Path(root)
    df = pd.concat([pd.read_csv(f).assign(part=f.parent.name) for f in sorted(root.glob('*/rows.csv.gz'))],
                   ignore_index=True)
    df['cluster'] = df.setting
    out = dict(prereg_sha256=hashlib.sha256(Path(PR.__file__).read_bytes()).hexdigest())
    c10 = df[df.setting.str.startswith('L/c10/')]
    c100 = df[df.setting.str.startswith('L/c100/')]
    syn = df[df.kind == 'syn']
    if len(c10):
        s = summarise(c10, 'cluster')
        s['AL1'] = bool(s['O1']['reach - uniform']['ci95'][1] < 0 and s['O1']['reach - proportional']['ci95'][1] < 0)
        s['AL3'] = bool(s['O2']['reach - uniform']['ci95'][1] < 0)
        s['by_setting'] = {k: summarise(g, 'cluster')['mean'] for k, g in c10.groupby('setting')}
        out['real_c10'] = s
    if len(c100):
        out['real_c100'] = summarise(c100, 'cluster')
    if len(syn):
        s = summarise(syn, 'cluster')
        s['AL2'] = bool(s['O1']['reach - uniform']['ci95'][1] < 0 and s['O1']['reach - proportional']['ci95'][1] < 0)
        s['AL4'] = bool(s['O2']['reach - uniform']['ci95'][1] < 0)
        syn = syn.assign(Kc=syn.K)
        s['by_K'] = {int(k): dict(O1_reach_minus_uniform=float((g.W_reach-g.W_uniform).mean()),
                                  O1_reach_minus_prop=float((g.W_reach-g.W_proportional).mean()),
                                  O2_reach_minus_uniform=float((g.err_reach-g.err_uniform).mean()))
                     for k, g in syn.groupby('Kc')}
        out['synthetic'] = s
    (root/'analysis.json').write_text(json.dumps(out, indent=2, default=float))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--part'); p.add_argument('--out'); p.add_argument('--workers', type=int)
    p.add_argument('--smoke', action='store_true'); p.add_argument('--analyse')
    a = p.parse_args(argv)
    if a.analyse:
        print(json.dumps({k: v for k, v in analyse(a.analyse).items()}, indent=1, default=float)[:6000]); return
    run_part(a.part, a.out, a.workers, a.smoke)


if __name__ == '__main__':
    main()
