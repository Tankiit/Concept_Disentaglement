"""Scores the claims of real/bridge2_prereg.py (with real/bridge2_amendment.py) from real/bridge2.py rows."""
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import real.bridge2_prereg as PR
import real.bridge2_amendment as AM

SOURCES = ('log', 'rows', 'unseen')


def load(root):
    frames = [pd.read_csv(f).assign(part=f.parent.name) for f in sorted(Path(root).glob('*/rows.csv.gz'))]
    df = pd.concat(frames, ignore_index=True)
    df['data'] = np.where(df.arm == 'synthetic', 'syn', df.setting.str.split('/').str[1])
    df['encoder'] = np.where(df.arm == 'synthetic', 'syn', df.setting.str.split('/').str[2])
    df['cluster'] = np.where(df.arm == 'synthetic', df.get('world', df.setting), df.setting)
    return df


def argmax_cols(df, cols, names):
    M = df[cols].to_numpy(float)
    M = np.where(np.isfinite(M), M, -np.inf)
    return np.array(names)[M.argmax(1)]


def label_and_gain(df, prefix, srcs):
    cols = [f'{prefix}_{s}' for s in srcs]
    M = df[cols].to_numpy(float)
    Mz = np.where(np.isfinite(M), M, -np.inf)
    return np.array(srcs)[Mz.argmax(1)], Mz.max(1)


def gain_of(df, prefix, choice, srcs):
    out = np.zeros(len(df))
    for s in srcs:
        hit = choice == s
        v = df[f'{prefix}_{s}'].to_numpy(float)
        out[hit] = np.where(np.isfinite(v[hit]), v[hit], 0.)
    return out


def choices(df, srcs, truth, cal, width_cols):
    """Every predictor's choice. truth: per-row best label (only cal rows are used for fitting)."""
    out = dict(REACH_pin=argmax_cols(df, [f'P_{s}' for s in srcs], list(srcs)),
               REACH=argmax_cols(df, width_cols, list(srcs)),
               radius=np.where(df.eps > df.r_entry, 'log', 'rows'))
    for src in srcs:
        out[f'always_{src}'] = np.full(len(df), src)
    key = pd.Series(list(zip(df.setting, df.n, df.m)), index=df.index)
    maj = pd.Series(truth[cal]).groupby(key[cal].to_numpy()).agg(lambda s: s.value_counts().index[0])
    glob = pd.Series(truth[cal]).value_counts().index[0]
    out['budget_majority'] = np.array([maj.get(k, glob) for k in key])
    from sklearn.linear_model import LogisticRegression
    X = np.log(np.clip(np.column_stack([df[width_cols].to_numpy(float), df.eps, df.r_entry]), 1e-6, None))
    X = np.nan_to_num(X, nan=np.log(1e-6))
    if len(set(truth[cal])) > 1:
        out['REACH_cal'] = LogisticRegression(max_iter=3000).fit(X[cal], truth[cal]).predict(X)
    else:
        out['REACH_cal'] = out['budget_majority']
    return out


def boot_contrast(ev, a, b, cluster, n_boot=2000, seed=0):
    """Two-way (seed x cluster) bootstrap of the mean regret difference."""
    diff = ev[f'reg_{a}']-ev[f'reg_{b}']
    sums = ev.assign(c=diff).groupby(['seed', cluster]).c.agg(['sum', 'count'])
    si, ci = sums.index.get_level_values(0).to_numpy(), sums.index.get_level_values(1).to_numpy()
    seeds, clus = np.unique(si), np.unique(ci)
    sp, cp = {s: i for i, s in enumerate(seeds)}, {c: i for i, c in enumerate(clus)}
    si, ci = np.array([sp[s] for s in si]), np.array([cp[c] for c in ci])
    rng = np.random.default_rng(seed)
    S, C = sums['sum'].to_numpy(), sums['count'].to_numpy()
    draws = []
    for _ in range(n_boot):
        ws = np.bincount(rng.integers(0, len(seeds), len(seeds)), minlength=len(seeds))
        wc = np.bincount(rng.integers(0, len(clus), len(clus)), minlength=len(clus))
        w = ws[si]*wc[ci]
        draws.append((w*S).sum()/max((w*C).sum(), 1))
    return dict(point=float(diff.mean()), ci95=[float(np.quantile(draws, .025)), float(np.quantile(draws, .975))])


def evaluate_block(df, cal_seeds, eval_seeds, prefix='D', finite=False):
    df = df.reset_index(drop=True)
    srcs = [s for s in SOURCES if f'{prefix}_{s}' in df and df[f'{prefix}_{s}'].notna().any()
            and f'P_{s}' in df and df[f'P_{s}'].notna().any()]
    if finite:
        srcs = ['log', 'rows']
    truth, best = label_and_gain(df, prefix, srcs)
    width_cols = [dict(log='W_cls', rows='W_label', unseen='W_unk')[s] for s in srcs]
    resolv = np.isfinite(best) & ((best >= PR.RESOLVABLE) if not finite else True)
    cal = (df.seed.isin(cal_seeds) & resolv).to_numpy()
    ch = choices(df, srcs, truth, cal, width_cols)
    ev = df[(df.seed.isin(eval_seeds) & resolv).to_numpy()].copy()
    pos = ev.index.to_numpy()
    ev['best'] = best[pos]
    res = dict(sources=srcs, n_eval=int(len(ev)), resolvable_share=float(np.mean(resolv)),
               truth_share=pd.Series(truth[pos]).value_counts(normalize=True).round(4).to_dict(), predictors={})
    for name, c in ch.items():
        ev[f'reg_{name}'] = ev.best-gain_of(ev, prefix, c[pos], srcs)
        res['predictors'][name] = dict(regret=float(ev[f'reg_{name}'].mean()),
                                       accuracy=float(np.mean(c[pos] == truth[pos])),
                                       share=pd.Series(c[pos]).value_counts(normalize=True).round(3).to_dict())
    res['contrasts'] = {f'{a} - {b}': boot_contrast(ev, a, b, 'cluster')
                        for a, b in (('REACH_pin', 'budget_majority'), ('REACH_pin', 'REACH_cal'),
                                     ('REACH_pin', 'REACH'), ('REACH_pin', 'radius'),
                                     ('REACH_pin', 'always_rows'))}
    # Reported after round 4 (not registered): exact ties between sources, and non-vacuous probes.
    gains = ev[[f'{prefix}_{s}' for s in srcs]].to_numpy(float)
    gains = np.where(np.isfinite(gains), gains, -np.inf)
    res['tie_share'] = float(np.mean(np.sum(np.isclose(gains, gains.max(1, keepdims=True), atol=1e-9), 1) > 1))
    nv = ev[ev.W < .99]
    res['vacuous_share'] = float(np.mean(ev.W >= .99))
    if len(nv) > 50:
        res['non_vacuous'] = dict(n_eval=int(len(nv)), **{name: float(nv[f'reg_{name}'].mean()) for name in ch})
    return res, ev


def analyse_all(root):
    root = Path(root)
    df = load(root)
    rc, re_ = PR.REAL_CAL_SEEDS, PR.REAL_EVAL_SEEDS
    sc, se = PR.SYN_CAL_SEEDS, PR.SYN_EVAL_SEEDS
    out = dict(prereg_sha256=hashlib.sha256(Path(PR.__file__).read_bytes()).hexdigest(),
               amendment_sha256=hashlib.sha256(Path(AM.__file__).read_bytes()).hexdigest())
    out['P6_inclusion'] = {f'{a}/{d}': dict(rate=float(g.I_covers.dropna().astype(bool).mean()),
                                            answer=float(g.I_covers.notna().mean()))
                           for (a, d), g in df.groupby(['arm', 'data'])}
    real, syn = df[df.arm != 'synthetic'], df[df.arm == 'synthetic']
    blocks = {}
    if len(real):
        c10 = real[real.data == 'c10']
        # Pooled H + L: unseen columns are NaN in L rows, so each row's choice set is its own arm's.
        parts = []
        for arm, g in c10.groupby('arm'):
            r, ev = evaluate_block(g, rc, re_)
            blocks[f'P1_c10_{arm}'] = r
            parts.append(ev)
        ev = pd.concat(parts, ignore_index=True)
        out['P1_c10_pooled'] = dict(n_eval=len(ev), regret={c[4:]: float(ev[c].mean()) for c in ev if c.startswith('reg_')},
                                    contrasts={f'{a} - {b}': boot_contrast(ev, a, b, 'cluster')
                                               for a, b in (('REACH_pin', 'budget_majority'), ('REACH_pin', 'REACH_cal'))})
        out['P1_c10_pooled']['passes'] = out['P1_c10_pooled']['contrasts']['REACH_pin - budget_majority']['ci95'][1] < 0
        for (data, arm, enc), g in real.groupby(['data', 'arm', 'encoder']):
            blocks[f'P1_{data}_{arm}_{enc}'] = evaluate_block(g, rc, re_)[0]
        # P4 is registered on CIFAR-10 label-shift settings only; bridge2.py's name filter ('L/c10')
        # also matched 'L/c100', so CIFAR-100 rows carry F columns and are dropped here (deviation).
        fin = real[real.F_log.notna() & (real.data == 'c10')] if 'F_log' in real else real.iloc[:0]
        if len(fin):
            r, _ = evaluate_block(fin, rc, re_, prefix='F', finite=True)
            r['passes'] = r['contrasts']['REACH_pin - budget_majority']['ci95'][1] < 0
            out['P4_finite'] = r
    if len(syn):
        r, ev = evaluate_block(syn, sc, se)
        r['passes_P2'] = r['contrasts']['REACH_pin - budget_majority']['ci95'][1] < 0
        r['passes_P3'] = r['predictors']['REACH_pin']['accuracy'] >= .70
        out['P2_P3_synthetic'] = r
        tab = []
        for key in ('K', 'acc', 'lam', 'n', 'm'):
            for v, g in ev.groupby(key):
                tab.append(dict(by=key, value=v, n_eval=len(g), **{c[4:]: float(g[c].mean()) for c in g if c.startswith('reg_')}))
        out['synthetic_breakdown'] = tab
    out['blocks'] = blocks
    (root/'analysis.json').write_text(json.dumps(out, indent=2, default=float))
    return out
