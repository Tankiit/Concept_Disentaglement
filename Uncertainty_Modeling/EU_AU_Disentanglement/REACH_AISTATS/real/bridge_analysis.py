"""Scores the registered claims of real/bridge_prereg.py from the per-part rows of real/bridge.py."""
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import real.bridge_prereg as PR

SOURCES = ('log', 'rows', 'unseen', 'shift')
WIDTH_OF = dict(log='W_cls', rows='W_label', unseen='W_unk')


def load(root):
    frames = []
    for f in sorted(Path(root).glob('*/rows.csv.gz')):
        frames.append(pd.read_csv(f).assign(part=f.parent.name))
    df = pd.concat(frames, ignore_index=True)
    df['encoder'] = df.setting.str.split('/').str[2]
    df['data'] = df.setting.str.split('/').str[1]
    return df


def oracle_labels(df):
    srcs = [s for s in SOURCES if f'D_{s}' in df and df[f'D_{s}'].notna().any()]
    D = df[[f'D_{s}' for s in srcs]].to_numpy(float)
    # A NaN removal (empty set under the oracle) cannot be scored; it is not the dominant source.
    Dz = np.where(np.isfinite(D), D, -np.inf)
    best = Dz.max(1)
    df = df.assign(oracle=np.array(srcs)[Dz.argmax(1)], best_gain=best,
                   resolvable=np.isfinite(best) & (best >= PR.RESOLVABLE))
    return df, srcs


def gain_of(df, choice):
    out = np.full(len(df), np.nan)
    for s in SOURCES:
        if f'D_{s}' in df:
            hit = choice == s
            out[hit] = df.loc[hit, f'D_{s}'].to_numpy(float)
    return np.where(np.isfinite(out), out, 0.)  # an unscorable choice gains nothing


def predictors(df, cal_seeds):
    """Every predictor's choice per row; fitted ones use calibration seeds only."""
    unseen = 'W_unk' in df and df['W_unk'].notna().any()
    comps = ['W_cls', 'W_label'] + (['W_unk'] if unseen else [])
    names = ['log', 'rows'] + (['unseen'] if unseen else [])
    W = df[comps].to_numpy(float)
    choice = {'REACH': np.array(names)[np.nan_to_num(W, nan=-1).argmax(1)]}
    cal = df.seed.isin(cal_seeds) & df.resolvable
    maj = df[cal].groupby(['setting', 'n', 'm']).oracle.agg(lambda s: s.value_counts().index[0])
    glob = df[cal].oracle.value_counts().index[0]
    key = list(zip(df.setting, df.n, df.m))
    choice['budget-majority'] = np.array([maj.get(k, glob) for k in key])
    choice['radius'] = np.where(df.eps > df.r_entry, 'log', 'rows')
    for s in names:
        choice[f'always-{s}'] = np.full(len(df), s)
    # Ensemble MI tercile map (each tercile -> one source), tuned on calibration regret.
    mi = df.Ens_MI.fillna(np.inf).to_numpy()
    cuts = np.nanquantile(df.loc[cal, 'Ens_MI'], [1/3, 2/3])
    terc = np.digitize(mi, cuts)
    best_map, best_val = None, np.inf
    import itertools
    for code in itertools.product(names, repeat=3):
        ch = np.array(code)[terc]
        val = np.mean((df.best_gain-gain_of(df, ch))[cal])
        if val < best_val:
            best_map, best_val = code, val
    choice['ens-MI'] = np.array(best_map)[terc]
    # REACH-cal: multinomial logistic regression on log widths and log radii.
    from sklearn.linear_model import LogisticRegression
    X = np.log(np.clip(np.column_stack([W, df.eps, df.r_entry]), 1e-6, None))
    X = np.nan_to_num(X, nan=np.log(1e-6))
    if df[cal].oracle.nunique() > 1:
        clf = LogisticRegression(max_iter=2000).fit(X[cal.to_numpy()], df.oracle[cal])
        choice['REACH-cal'] = clf.predict(X)
    else:
        choice['REACH-cal'] = choice['budget-majority']
    return choice


def score(df, choice, cal_seeds, cluster, n_boot=2000, pairs=(('REACH', 'budget-majority'),)):
    ev = df[~df.seed.isin(cal_seeds) & df.resolvable].copy()
    idx = ev.index.to_numpy()
    pos = df.index.get_indexer(idx)
    res = {}
    for name, ch in choice.items():
        ev[f'reg_{name}'] = ev.best_gain-gain_of(ev, ch[pos])
        ev[f'acc_{name}'] = ch[pos] == ev.oracle.to_numpy()
    point = {name: dict(regret=float(ev[f'reg_{name}'].mean()), accuracy=float(ev[f'acc_{name}'].mean()))
             for name in choice}
    # Cluster bootstrap: resample clusters (seed, or seed x setting two-way by resampling both).
    rng = np.random.default_rng(0)
    groups = {k: g for k, g in ev.groupby(cluster)}
    keys = list(groups)
    cis = {}
    if isinstance(cluster, list) and len(cluster) == 2:
        a_vals, b_vals = ev[cluster[0]].unique(), ev[cluster[1]].unique()
    for a, b in pairs:
        diffs = []
        col = ev[f'reg_{a}']-ev[f'reg_{b}']
        sums = ev.assign(c=col).groupby(cluster).c.agg(['sum', 'count'])
        for _ in range(n_boot):
            if isinstance(cluster, list) and len(cluster) == 2:
                sa = rng.choice(a_vals, len(a_vals)); sb = rng.choice(b_vals, len(b_vals))
                ca = pd.Series(sa).value_counts(); cb = pd.Series(sb).value_counts()
                wts = np.array([ca.get(i, 0)*cb.get(j, 0) for i, j in sums.index])
            else:
                draw = pd.Series(rng.choice(keys, len(keys))).value_counts()
                wts = np.array([draw.get(i, 0) for i in sums.index])
            diffs.append((wts*sums['sum']).sum()/max((wts*sums['count']).sum(), 1))
        cis[f'{a} - {b}'] = dict(point=float(col.mean()), ci95=[float(np.quantile(diffs, .025)),
                                                                 float(np.quantile(diffs, .975))])
    return dict(n_eval=int(len(ev)), oracle_share=ev.oracle.value_counts(normalize=True).round(4).to_dict(),
                predictors=point, contrasts=cis)


def attribution(df, label, cal_seeds, cluster):
    df, srcs = oracle_labels(df)
    choice = predictors(df, cal_seeds)
    pairs = [('REACH', 'budget-majority'), ('REACH-cal', 'REACH'), ('ens-MI', 'budget-majority'),
             ('REACH', 'radius'),
             # EXPLORATORY (added after the CIFAR-10 results were seen; not in the pre-registration):
             ('REACH-cal', 'budget-majority')]
    out = score(df, choice, cal_seeds, cluster, pairs=pairs)
    out['resolvable_share'] = float(df.resolvable.mean())
    out['by_budget'] = {f'n={n},m={m}': g.oracle.value_counts(normalize=True).round(3).to_dict()
                        for (n, m), g in df[df.resolvable].groupby(['n', 'm'])}
    return out


def bootstrap_spearman(units, x, y, n_boot=2000, by='seed', rng=None):
    """Spearman across units of seed-averaged y, with a seed bootstrap CI."""
    rng = rng or np.random.default_rng(1)
    agg = units.groupby(['unit', by])[[x, y]].mean().reset_index()
    seeds = agg[by].unique()
    pt = agg.groupby('unit')[[x, y]].mean()
    rho = spearmanr(pt[x], pt[y]).statistic
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(seeds, len(seeds))
        g = pd.concat([agg[agg[by] == s] for s in pick]).groupby('unit')[[x, y]].mean()
        draws.append(spearmanr(g[x], g[y]).statistic)
    perm = [spearmanr(pt[x], rng.permutation(pt[y].to_numpy())).statistic for _ in range(n_boot)]
    return dict(rho=float(rho), ci95=[float(np.nanquantile(draws, .025)), float(np.nanquantile(draws, .975))],
                perm_p_two_sided=float(np.mean(np.abs(perm) >= abs(rho))), units=int(len(pt)))


def analyse_all(root):
    root = Path(root)
    df = load(root)
    cal = PR.CAL_SEEDS
    out = dict(prereg_sha256=hashlib.sha256(Path(PR.__file__).read_bytes()).hexdigest())
    H, L, S = df[df.arm == 'held_out'], df[df.arm == 'label_shift'], df[df.arm == 'corruption']
    # Coverage sanity: the total interval includes its target.
    out['inclusion'] = {arm: dict(rate=float(g.I_covers.dropna().astype(bool).mean()),
                                  answer_rate=float(g.I_covers.notna().mean()))
                        for arm, g in df.groupby('arm')}
    # A1 / A2 / A3: attribution.
    out['A1_heldout_c10'] = attribution(H[H.data == 'c10'].reset_index(drop=True), 'H', cal, ['seed', 'setting'])
    for enc, g in H.groupby(['data', 'encoder']):
        out[f'A1_heldout_{enc[0]}_{enc[1]}'] = attribution(g.reset_index(drop=True), 'H', cal, ['seed', 'setting'])
    out['A2_labelshift_c10'] = attribution(L[L.data == 'c10'].reset_index(drop=True), 'L', cal, 'seed')
    for enc, g in L.groupby(['data', 'encoder']):
        out[f'A2_labelshift_{enc[0]}_{enc[1]}'] = attribution(g.reset_index(drop=True), 'L', cal, 'seed')
    if len(S):
        out['S_attribution'] = attribution(S.reset_index(drop=True), 'S', PR.SHIFT_CAL_SEEDS, 'seed')
    # D1: log W_cls ~ log(eps + r) + log sigma_hat over (setting, n, m) units, seed bootstrap.
    known = L[~L.outlier]
    u = known.groupby(['setting', 'n', 'm', 'seed']).agg(W=('W_cls', 'mean'), rad=('radius', 'first'),
                                                         sig=('sigma_hat', 'first')).reset_index()
    def coef(t):
        g = t.groupby(['setting', 'n', 'm'])[['W', 'rad', 'sig']].mean()
        X = np.column_stack([np.ones(len(g)), np.log(g.rad), np.log(g.sig)])
        return np.linalg.lstsq(X, np.log(g.W), rcond=None)[0]
    seeds = u.seed.unique(); rng = np.random.default_rng(2)
    bs = np.array([coef(pd.concat([u[u.seed == s] for s in rng.choice(seeds, len(seeds))])) for _ in range(500)])
    c = coef(u)
    bp = known.dropna(subset=['bprime'])
    out['D1'] = dict(coef_log_radius=float(c[1]), coef_log_sigma=float(c[2]),
                     ci95_log_sigma=[float(np.quantile(bs[:, 2], .025)), float(np.quantile(bs[:, 2], .975))],
                     passes=bool(np.quantile(bs[:, 2], .975) < 0),
                     bprime_holds=float((bp.W_cls <= bp.bprime+1e-6).mean()),
                     bprime_nontrivial_share=float((bp.bprime < 1).mean()),
                     sigma_by_setting=known.groupby('setting').sigma_hat.mean().round(4).to_dict())
    # D2: coverage width vs separation (held-out candidates).
    big = H[(H.n == max(PR.BUDGETS_N)) & (H.m == max(PR.BUDGETS_M))].assign(unit=lambda t: t.setting)
    out['D2'] = dict(spearman_s_vs_Wunk=bootstrap_spearman(big, 's_true', 'W_unk'))
    hs = H.groupby(['setting', 'seed', 'n', 'm']).agg(W=('W_unk', 'mean'), s=('s_true', 'first')).reset_index()
    hs = hs[hs.m == max(PR.BUDGETS_M)]
    def slope(t):
        return np.polyfit(np.log(t.n), np.log(np.clip(t.W, 1e-6, None)), 1)[0]
    sl = hs.groupby(['setting', 'seed'])[['n', 'W']].apply(slope).rename('slope').reset_index()
    sl = sl.merge(hs.groupby('setting').s.first().reset_index(), on='setting').rename(columns={'setting': 'unit'})
    out['D2']['spearman_s_vs_nslope'] = bootstrap_spearman(sl, 's', 'slope')
    out['D2']['by_candidate'] = big.groupby('setting').agg(s=('s_true', 'first'), W_unk=('W_unk', 'mean'),
                                                           W_unk_q90=('W_unk', lambda x: x.quantile(.9))
                                                           ).round(4).to_dict('index')
    # D3: label width vs realised anchored error.
    lab = L.dropna(subset=['lab_covers'])
    within = []
    for (mm, st), g in lab.groupby(['m', 'setting']):
        if g.W_label.nunique() > 1:
            within.append(spearmanr(g.W_label, g.anchor_err).statistic)
    out['D3'] = dict(inclusion=float(lab.lab_covers.astype(bool).mean()),
                     within_m_spearman_mean=float(np.nanmean(within)),
                     within_m_spearman_positive_share=float(np.mean(np.array(within) > 0)),
                     cells=len(within))
    # D4: domain shift as a measured violation.
    if len(S):
        S = S.assign(rho=S.row_drift/S.radius)
        unit = S.groupby(['setting', 'n', 'm']).agg(
            rho=('rho', 'first'), inclusion=('I_covers', lambda x: x.dropna().astype(bool).mean()),
            answer=('I_covers', lambda x: x.notna().mean()),
            incompatible=('incompatible', 'mean')).reset_index()
        so, _ = oracle_labels(S)
        shift_dom = so[so.resolvable & (so.oracle == 'shift')]
        # Units where every probe abstained (empty set) have no inclusion rate; they enter via 'answer'.
        ans = unit.dropna(subset=['inclusion'])
        out['D4'] = dict(spearman_rho_inclusion=float(spearmanr(ans.rho, ans.inclusion).statistic),
                         units_all_abstain=int(unit.inclusion.isna().sum()),
                         spearman_rho_answer=float(spearmanr(unit.rho, unit.answer).statistic),
                         spearman_rho_incompatible=float(spearmanr(unit.rho, unit.incompatible).statistic),
                         inclusion_rho_lt1=float(unit[unit.rho < 1].inclusion.mean()),
                         inclusion_rho_ge1=float(unit[unit.rho >= 1].inclusion.mean()),
                         units_rho_lt1=int((unit.rho < 1).sum()), units=int(len(unit)),
                         shift_dominant_share=float((so[so.resolvable].oracle == 'shift').mean()),
                         # The flag fires only where the set is empty (abstain), so every answered
                         # shift-dominant input is by construction unflagged.
                         flag_rate_on_shift_dominant=float(shift_dom.incompatible.mean())
                         if len(shift_dom) else None,
                         units_table=unit.round(4).to_dict('records'))
    (root/'analysis.json').write_text(json.dumps(out, indent=2, default=float))
    brief = {k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items()
                                                      if kk not in ('by_budget', 'units_table', 'by_candidate')})
             for k, v in out.items()}
    print(json.dumps(brief, indent=1, default=float))
    return out
