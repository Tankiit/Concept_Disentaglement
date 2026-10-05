"""Per-component dose-responses in classifier worlds with K = 4 fixed (protocol: real/gauss_dose_prereg.py,
part D).

    python -m real.dose --out results/dose
"""
import argparse, hashlib, json, os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from models.ball import MMDBall, embedding_radius
import real.audit as A
import real.bridge as B
import real.gauss_dose_prereg as PR

ALPHA = A.ALPHA


def cworld(acc, lam, draw, K=PR.D_K):
    """Classifier world (real/bridge2_amendment.py recipe, per = 1): d = K + 1 cells, last = outlier."""
    d = K+1
    rng = np.random.default_rng(np.random.SeedSequence([K, int(acc*100), int(lam*10), draw, 4040]))
    rows = np.zeros((K, d))
    for k in range(K):
        noise = rng.dirichlet(np.ones(d)); noise[-1] *= .2; noise /= noise.sum()
        rows[k] = acc*np.eye(d)[k]+(1-acc)*noise
    q = lam*(.5*np.eye(d)[-1]+.5*rng.dirichlet(np.ones(d)))+(1-lam)*(rng.dirichlet(np.ones(K))@rows)
    return dict(name=f'K{K}/a{acc:g}/l{lam:g}/{draw}', acc=acc, lam=lam, draw=draw, K=K, d=d, rows=rows, q=q,
                sigma=B.sigma(rows), s=A.hull_distance(q, rows))


def truth_weights(wd, seed):
    pi = np.random.default_rng(np.random.SeedSequence([int(seed), 505, wd['draw'], int(wd['acc']*100)])).dirichlet(np.ones(wd['K']))
    return np.r_[.9*pi, .1]


def log_hist(wd, seed, n, w):
    a = np.vstack([wd['rows'], wd['q']])
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 101, wd['draw'], int(wd['acc']*100), int(wd['lam']*10)]))
    full = rng.choice(wd['d'], max(PR.D_N), p=w@a)
    return np.bincount(full[:n], minlength=wd['d'])/n


def known_rows_task(task):
    """D-cls and D-unk: KNOWN rows and q (r = 0), ball radius eps_n."""
    kind, wd, seed, n = task
    K, d = wd['K'], wd['d']
    a = np.vstack([wd['rows'], wd['q']])
    w = truth_weights(wd, seed)
    emp = log_hist(wd, seed, n, w)
    ball = MMDBall(a, emp, embedding_radius(n, ALPHA), 0., 'l2')
    out = []
    for cell in range(d):
        col = a[:, cell]
        row = dict(kind=kind, world=wd['name'], acc=wd['acc'], lam=wd['lam'], sigma=wd['sigma'], s=wd['s'],
                   seed=int(seed), n=int(n), cell=cell, known_mass=float(w[:K]@col[:K]))
        if kind == 'cls' and cell < K:
            yhat = int(np.argmax(col[:K]))
            num = np.zeros(K+1); num[yhat] = col[yhat]
            den = col.copy(); den[-1] = 0
            r = ball.ratio_interval(num, num, den)
            row['W_cls'] = r['upper']-r['lower']
            out.append(row)
        elif kind == 'unk' and wd['q'][cell] > 0:
            num = np.zeros(K+1); num[-1] = col[-1]
            r = ball.ratio_interval(num, num, col)
            row['W_unk'] = r['upper']-r['lower']
            out.append(row)
    return out


def label_task(task):
    """D-label: noisy references (flip rate eta), debiased exactly as real/audit._reference."""
    wd, seed, m, eta = task
    K, d = wd['K'], wd['d']
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 202, wd['draw'], int(wd['acc']*100)]))
    pools = {(k,): [rng.choice(d, 8000, p=wd['rows'][k])] for k in range(K)}
    ctx = dict(groups=[(k,) for k in range(K)], d=d, ref_cells=pools)
    rows_hat, centre, _, r = A._reference(ctx, m, seed, eta)
    lo, hi = np.clip(centre-r, 0, 1), np.clip(centre+r, 0, 1)
    out = []
    for cell in range(K):
        yhat = int(np.argmax(rows_hat[:, cell]))
        others = np.arange(K) != yhat
        l = lo[yhat, cell]/max(lo[yhat, cell]+hi[others, cell].sum(), 1e-12)
        h = hi[yhat, cell]/max(hi[yhat, cell]+lo[others, cell].sum(), 1e-12)
        t = wd['rows'][:, cell]; e = rows_hat[:, cell]
        truth = t[yhat]/t.sum(); est = e[yhat]/e.sum() if e.sum() > 0 else np.nan
        out.append(dict(kind='label', world=wd['name'], acc=wd['acc'], seed=int(seed), m=int(m), eta=float(eta),
                        cell=cell, W_label=float(h-l), covers=bool(l-1e-9 <= truth <= h+1e-9), err=abs(est-truth)))
    return out


def _run(task):
    return label_task(task[1:]) if task[0] == 'label' else known_rows_task(task)


def run(out, workers=None):
    tasks = []
    for acc in PR.D_CLS_ACC:
        for draw in range(PR.D_DRAWS):
            wd = cworld(acc, 1., draw)
            tasks += [('cls', wd, s, n) for s in PR.D_SEEDS for n in PR.D_N]
    for lam in PR.D_UNK_LAM:
        for draw in range(PR.D_DRAWS):
            wd = cworld(PR.D_UNK_ACC, lam, draw)
            tasks += [('unk', wd, s, n) for s in PR.D_SEEDS for n in PR.D_N]
    for acc in PR.D_LAB_ACC:
        for draw in range(PR.D_DRAWS):
            wd = cworld(acc, 1., draw)
            tasks += [('label', wd, s, m, eta) for s in PR.D_SEEDS for m in PR.D_LAB_M for eta in PR.D_ETA]
    with ProcessPoolExecutor(workers or os.cpu_count()) as pool:
        rows = [r for chunk in pool.map(_run, tasks, chunksize=8) for r in chunk]
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'rows.csv.gz', index=False)
    print('wrote', len(rows), flush=True)


def _slope(x, y):
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def analyse(root, n_boot=2000):
    root = Path(root)
    df = pd.read_csv(root/'rows.csv.gz')
    rng = np.random.default_rng(3)
    seeds = df.seed.unique()
    boot = lambda f: np.array([f(rng.choice(seeds, len(seeds))) for _ in range(n_boot)])
    ci = lambda v: [float(np.nanquantile(v, .025)), float(np.nanquantile(v, .975))]
    out = dict(prereg_sha256=hashlib.sha256(Path(PR.__file__).read_bytes()).hexdigest())
    cls = df[df.kind == 'cls']
    # Dcls-1: n-rate over probes non-saturated at every n.
    pm = cls.groupby(['world', 'cell', 'n']).W_cls.mean().unstack('n')
    ok = pm[(pm < .95).all(1)].index
    sub = cls.set_index(['world', 'cell']).loc[ok].reset_index()
    def nrate(pick):
        g = pd.concat([sub[sub.seed == s] for s in pick]).groupby('n').W_cls.mean()
        return _slope(g.index.to_numpy(float), g.to_numpy())
    g = sub.groupby('n').W_cls.mean()
    b = boot(nrate)
    out['Dcls1'] = dict(slope=_slope(g.index.to_numpy(float), g.to_numpy()), ci95=ci(b), probes=int(len(ok)),
                        probes_total=int(len(pm)), mean_by_n=g.round(4).to_dict())
    out['Dcls1']['passes'] = bool(-.75 <= out['Dcls1']['slope'] <= -.25 and ci(b)[0] >= -.9 and ci(b)[1] <= -.1)
    # Dcls-2: distinguishability at n = 8000.
    c8 = cls[cls.n == 8000]
    def sig(pick):
        g = pd.concat([c8[c8.seed == s] for s in pick]).groupby('world').agg(W=('W_cls', 'mean'), s=('sigma', 'first'))
        return _slope(g.s, g.W)
    g = c8.groupby('world').agg(W=('W_cls', 'mean'), s=('sigma', 'first'), acc=('acc', 'first'))
    b = boot(sig)
    out['Dcls2'] = dict(slope=_slope(g.s, g.W), ci95=ci(b), passes=bool(ci(b)[1] < 0),
                        by_acc=g.groupby('acc')[['s', 'W']].mean().round(4).to_dict('index'))
    # Dcls-3: probe mass within world.
    rho = [spearmanr(gg.groupby('cell').known_mass.mean(), gg.groupby('cell').W_cls.mean()).statistic
           for _, gg in c8.groupby('world')]
    out['Dcls3'] = dict(share_negative=float(np.mean(np.array(rho) < 0)), worlds=len(rho),
                        passes=bool(np.mean(np.array(rho) < 0) >= .7))
    # D-label.
    lab = df[df.kind == 'label']
    per = lab.groupby(['world', 'm', 'eta']).W_label.mean().reset_index()
    sp = [spearmanr(gg.eta, gg.W_label).statistic for _, gg in per.groupby(['world', 'm'])]
    out['Dlab1'] = dict(share_monotone=float(np.mean(np.isclose(sp, 1))), groups=len(sp),
                        passes=bool(np.mean(np.isclose(sp, 1)) >= .8),
                        mean_by_eta_m=lab.groupby(['m', 'eta']).W_label.mean().round(4).unstack('eta').to_dict('index'))
    cov = lab.assign(covers=lab.covers.astype(bool)).groupby('eta').covers.mean()
    out['Dlab2'] = dict(inclusion_by_eta=cov.round(4).to_dict(), passes=bool((cov >= .95).all()))
    sp3 = [spearmanr(gg.W_label, gg.err).statistic for _, gg in lab.groupby(['world', 'm']) if gg.W_label.nunique() > 1]
    out['Dlab3'] = dict(mean_spearman=float(np.nanmean(sp3)), share_positive=float(np.mean(np.array(sp3) > 0)),
                        groups=len(sp3), passes=bool(np.nanmean(sp3) > 0))
    out['Dlab4'] = 'construction: W_label never reads the log (real/dose.py label_task), so it is identical across n'
    # D-unk.
    unk = df[df.kind == 'unk']
    big = unk[unk.n == max(PR.D_N)]
    def srho(pick):
        g = pd.concat([big[big.seed == s] for s in pick]).groupby('world').agg(W=('W_unk', 'mean'), s=('s', 'first'))
        return spearmanr(g.s, g.W).statistic
    g = big.groupby('world').agg(W=('W_unk', 'mean'), s=('s', 'first'), lam=('lam', 'first'))
    b = boot(srho)
    out['Dunk1'] = dict(spearman=float(spearmanr(g.s, g.W).statistic), ci95=ci(b), passes=bool(ci(b)[1] < 0),
                        by_lam=g.groupby('lam')[['s', 'W']].mean().round(4).to_dict('index'))
    res = {}
    for lam in (0., 1.):
        u = unk[unk.lam == lam]
        def ns(pick, u=u):
            g = pd.concat([u[u.seed == s] for s in pick]).groupby('n').W_unk.mean()
            return _slope(g.index.to_numpy(float), np.clip(g.to_numpy(), 1e-9, None))
        gg = u.groupby('n').W_unk.mean()
        b = boot(ns)
        res[f'lam{lam:g}'] = dict(slope=_slope(gg.index.to_numpy(float), np.clip(gg.to_numpy(), 1e-9, None)), ci95=ci(b),
                                  mean_by_n=gg.round(4).to_dict())
    l0, l1 = res['lam0'], res['lam1']
    res['passes_lam0'] = bool(l0['ci95'][0] <= 0 <= l0['ci95'][1] or l0['slope'] > -.1)
    res['passes_lam1'] = bool(l1['slope'] < -.25 and l1['ci95'][1] < -.1)
    out['Dunk2'] = res
    (root/'analysis.json').write_text(json.dumps(out, indent=2, default=float))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out'); p.add_argument('--workers', type=int); p.add_argument('--analyse')
    a = p.parse_args(argv)
    if a.analyse:
        print(json.dumps(analyse(a.analyse), indent=1, default=float)); return
    run(a.out, a.workers)


if __name__ == '__main__':
    main()
