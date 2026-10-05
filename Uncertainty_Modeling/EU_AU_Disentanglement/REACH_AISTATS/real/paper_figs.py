"""Appendix figures and the attribution table, generated from saved results only.

    python -m real.paper_figs

Writes paper/figures/figA{1..6}_*.pdf and paper/tables/attribution.tex.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES, FIG, TAB = ROOT/'results', ROOT/'paper'/'figures', ROOT/'paper'/'tables'
C = dict(pin='#0072B2', raw='#D55E00', maj='#999999', cal='#009E73', flag='#CC79A7', incl='#0072B2')
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'legend.fontsize': 7,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42})


def J(path):
    return json.loads((RES/path).read_text())


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG/name, bbox_inches='tight')
    plt.close(fig)
    print('wrote', FIG/name)


ENC = dict(clip='CLIP', dinov2='DINOv2', resnet50='RN50')


def _block(key):
    """'P1_c10_held_out' or 'P1_c10_label_shift_clip' -> (data, arm, encoder or None)."""
    rest = key.split('_', 2)[2]
    arm = 'held_out' if rest.startswith('held_out') else 'label_shift'
    enc = rest[len(arm)+1:] or None
    return key.split('_')[1], arm, enc


def attribution_rows(per_encoder=True):
    """(label, data, REACH raw, budget-majority, REACH-pin, contrast): bridge v2, Gaussian, synthetic, finite."""
    b2, g = J('bridge2/analysis.json'), J('gauss/analysis.json')
    rows = []
    for key, r in b2['blocks'].items():
        data, arm, enc = _block(key)
        if (enc is None) == per_encoder and data == 'c10':
            continue
        if data == 'c100' and enc is None:
            continue
        name = ('C10' if data == 'c10' else 'C100')+(' held-out' if arm == 'held_out' else ' label shift')
        name += f', {ENC[enc]}' if enc else ', 3 encoders'
        p = r['predictors']
        rows.append((name, data, p['REACH']['regret'], p['budget_majority']['regret'], p['REACH_pin']['regret'],
                     r['contrasts']['REACH_pin - budget_majority']))
    p = g['G1']['predictors']
    rows.append(('Gaussian features, held-out', 'gauss', p['REACH']['regret'], p['budget_majority']['regret'],
                 p['REACH_pin']['regret'], g['G1']['contrasts']['REACH_pin - budget_majority']))
    s = b2['P2_P3_synthetic']; p = s['predictors']
    rows.append(('Classifier worlds (synthetic)', 'syn', p['REACH']['regret'], p['budget_majority']['regret'],
                 p['REACH_pin']['regret'], s['contrasts']['REACH_pin - budget_majority']))
    f = b2['P4_finite']; p = f['predictors']
    rows.append(('C10 label shift, $4\\times$ acquisition', 'fin', p['REACH']['regret'], p['budget_majority']['regret'],
                 p['REACH_pin']['regret'], f['contrasts']['REACH_pin - budget_majority']))
    return rows


def table_attribution():
    b2 = J('bridge2/analysis.json')
    P1 = b2['P1_c10_pooled']
    lines = [r'\begin{tabular}{lcccc}', r'\toprule',
             r' & raw widths & budget-majority & REACH-pin & pin $-$ majority [95\% CI] \\', r'\midrule']
    for label, data, raw, maj, pin, con in attribution_rows(per_encoder=False):
        ci = con['ci95']
        lines.append(f'{label} & {raw:.3f} & {maj:.3f} & {pin:.4f} & ${con["point"]:+.3f}$ $[{ci[0]:+.3f}, {ci[1]:+.3f}]$ \\\\')
    c = P1['contrasts']['REACH_pin - budget_majority']
    lines += [r'\midrule',
              f'C10 pooled (registered P1) & {P1["regret"]["REACH"]:.3f} & {P1["regret"]["budget_majority"]:.3f} & '
              f'{P1["regret"]["REACH_pin"]:.4f} & ${c["point"]:+.3f}$ $[{c["ci95"][0]:+.3f}, {c["ci95"][1]:+.3f}]$ \\\\',
              r'\bottomrule', r'\end{tabular}']
    (TAB/'attribution.tex').write_text('\n'.join(lines)+'\n')
    print('wrote', TAB/'attribution.tex')


def figA1():
    rows = attribution_rows()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    y = np.arange(len(rows))[::-1]
    for i, (lab, data, raw, maj, pin, con) in enumerate(rows):
        ax.plot([maj, raw], [y[i]]*2, color='#dddddd', lw=1, zorder=0)
    ax.scatter([r[2] for r in rows], y, color=C['raw'], label='raw widths (argmax $W_{cls},W_{label},W_{unk}$)', s=18)
    ax.scatter([r[3] for r in rows], y, color=C['maj'], marker='s', label='budget-majority (input-independent)', s=18)
    ax.scatter([r[4] for r in rows], y, color=C['pin'], marker='D', label='REACH-pin', s=18)
    ax.set_yticks(y); ax.set_yticklabels([r[0].replace('$4\\times$', '4x') for r in rows])
    ax.set_xscale('symlog', linthresh=.01); ax.set_xlim(0, .6)
    ax.set_xlabel('oracle regret (width units; lower is better), evaluation seeds')
    ax.legend(loc='upper center', bbox_to_anchor=(.45, 1.16), ncol=3, frameon=False)
    for i, r in enumerate(rows):
        lo, hi = r[5]['ci95']
        tag = 'pin loses' if lo > 0 else ('tie' if hi >= 0 else None)
        if tag:
            ax.annotate(tag, (r[4], y[i]), xytext=(4, 4), textcoords='offset points', fontsize=6, color=C['raw'])
    save(fig, 'figA1_attribution.pdf')


def figA2():
    frames = [pd.read_csv(f) for f in (RES/'bridge').glob('L_*/rows.csv.gz')]
    L = pd.concat(frames)
    L = L[~L.outlier]
    u = L.groupby(['setting', 'n', 'm']).agg(W=('W_cls', 'mean'), sig=('sigma_hat', 'mean'), rad=('radius', 'mean')).reset_index()
    u['grouping'] = u.setting.str.split('/').str[-1]
    dose = pd.read_csv(RES/'dose'/'rows.csv.gz')
    c8 = dose[(dose.kind == 'cls') & (dose.n == 8000)].groupby('world').agg(W=('W_cls', 'mean'), sig=('sigma', 'first'))
    fig, axs = plt.subplots(1, 2, figsize=(7, 2.6))
    for g, mk, col in (('super4', 'o', '#0072B2'), ('classes', '^', '#D55E00'), ('c100_2', 's', '#009E73')):
        t = u[(u.grouping == g) & (u.n == 8000) & (u.m == 500)]
        axs[0].scatter(t.sig, t.W, marker=mk, color=col, label={'super4': 'CIFAR-10, 4 sources', 'classes': 'CIFAR-10, 10 sources',
                                                                 'c100_2': 'CIFAR-100, 2 sources'}[g], s=22)
    axs[0].set_xlabel(r'$\sigma(\hat A)$'); axs[0].set_ylabel(r'mean $W_{cls}$ ($n$=8000, $m$=500)')
    axs[0].set_title('Bridge v1 D1 (FAIL): pooled across $K$, slope $>0$')
    axs[0].legend(frameon=False)
    axs[1].scatter(c8.sig, c8.W, color=C['pin'], s=14)
    co = np.polyfit(np.log(c8.sig), np.log(c8.W), 1)
    xs = np.linspace(c8.sig.min(), c8.sig.max(), 50)
    axs[1].plot(xs, np.exp(np.polyval(co, np.log(xs))), color='k', lw=.8)
    axs[1].set_xlabel(r'$\sigma(A)$'); axs[1].set_ylabel(r'mean $W_{cls}$ ($n$=8000, known rows)')
    axs[1].set_title(f'Dose Dcls-2 (PASS): $K=4$ fixed, log-log slope {co[0]:.2f}')
    save(fig, 'figA2_sigma.pdf')


def figA3():
    v1 = J('bridge/analysis.json')['D2']['by_candidate']
    g = J('gauss/analysis.json')['G4']['by_world']
    dose = pd.read_csv(RES/'dose'/'rows.csv.gz')
    du = dose[(dose.kind == 'unk') & (dose.n == 128000)].groupby('world').agg(W=('W_unk', 'mean'), s=('s', 'first'))
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for k, v in v1.items():
        c100 = '/c100/' in k
        axs[0].scatter(v['s'], v['W_unk'], color='#009E73' if c100 else '#0072B2', marker='s' if c100 else 'o', s=14)
    axs[0].scatter([], [], color='#0072B2', marker='o', label='CIFAR-10, one held-out class')
    axs[0].scatter([], [], color='#009E73', marker='s', label='CIFAR-100, held-out superclass')
    axs[0].legend(frameon=False, fontsize=6); axs[0].set_title('Bridge v1 D2 (FAIL pooled): real')
    axs[0].set_ylabel(r'mean $W_{unk}$ (largest budget)')
    gd = pd.DataFrame(g).T
    for d, mk in zip(sorted(gd.delta.unique()), ('o', '^', 's')):
        t = gd[gd.delta == d]
        axs[1].scatter(t.s, t.W_unk, marker=mk, s=14, label=f'$\\Delta$={d:g}')
    axs[1].legend(frameon=False, fontsize=6); axs[1].set_title('Gaussian G4b (FAIL): $s\\leq0.2$')
    axs[2].scatter(du.s, du.W, color=C['pin'], s=14); axs[2].set_title('Dose Dunk-1 (PASS): $\\rho=-0.97$')
    for a in axs:
        a.set_xlabel('separation $s$ in cell space')
    save(fig, 'figA3_separation.pdf')


def _units(df):
    u = df.groupby(['setting', 'n', 'm']).agg(
        drift=('row_drift', 'first'), radius=('radius', 'mean'),
        inclusion=('I_covers', lambda x: x.dropna().astype(bool).mean() if x.notna().any() else np.nan),
        answer=('I_covers', lambda x: x.notna().mean()), flag=('incompatible', 'mean')).reset_index()
    u['rho'] = np.maximum(u.drift/u.radius, .03)  # rho = 0 (no shift) drawn at 0.03 on the log axis
    return u


def figA4():
    srcs = [('CIFAR-10-C + CIFAR-10.1 (D4, PASS)', RES/'bridge'/'S_c10_clip'/'rows.csv.gz'),
            ('Gaussian, registered (G3, FAIL: $\\rho<0.4$)', RES/'gauss'/'shift'/'rows.csv.gz'),
            ('Gaussian, in-subspace shift (EXPLORATORY)', RES/'gauss'/'shift_explore_mean'/'rows.csv.gz')]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
    bins = np.array([0, .1, .25, .5, 1, 2, 4, 8, 1e9])
    for ax, (title, path) in zip(axs, srcs):
        u = _units(pd.read_csv(path))
        u['b'] = pd.cut(u.rho, bins, include_lowest=True)
        t = u.groupby('b', observed=True).agg(rho=('rho', 'median'), inc=('inclusion', 'mean'),
                                              ans=('answer', 'mean'), flag=('flag', 'mean'))
        ax.plot(t.rho, t.inc, 'o-', color=C['incl'], label='inclusion (answered)', ms=3)
        ax.plot(t.rho, t.flag, 's--', color=C['flag'], label='incompatibility flag', ms=3)
        ax.plot(t.rho, t.ans, '^:', color='k', label='answer rate', ms=3)
        ax.axhline(.95, color='#bbbbbb', lw=.6); ax.axvline(1, color='#bbbbbb', lw=.6)
        ax.set_xscale('log'); ax.set_title(title, fontsize=7)
        ax.set_xlabel(r'$\rho$ = row drift / radius (0 at 0.03)')
    axs[0].legend(frameon=False, fontsize=6, loc='center left')
    save(fig, 'figA4_shift.pdf')


def figA5():
    d = pd.read_csv(RES/'dose'/'rows.csv.gz')
    lab = d[d.kind == 'label']
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for m, col in zip(sorted(lab.m.unique()), ('#D55E00', '#009E73', '#0072B2')):
        t = lab[lab.m == m].groupby(['world', 'eta']).W_label.mean().unstack('eta')
        for _, r in t.iterrows():
            ax.plot(r.index, r.values, color=col, alpha=.25, lw=.7)
        ax.plot(t.columns, t.mean(), color=col, lw=1.8, label=f'$m$={int(m)}')
    ax.set_xlabel(r'label flip rate $\eta$'); ax.set_ylabel(r'$W_{label}$')
    ax.set_title('Dlab-1 (FAIL as registered): ties at $W_{label}=1$', fontsize=7)
    ax.legend(frameon=False)
    save(fig, 'figA5_label_noise.pdf')


def figA6():
    b2 = J('bridge2/analysis.json')
    t = pd.DataFrame(b2['synthetic_breakdown'])
    fig, axs = plt.subplots(1, 2, figsize=(7, 2.4))
    for by, ax in (('m', axs[0]), ('lam', axs[1])):
        s = t[t.by == by]
        x = np.arange(len(s))
        ax.bar(x-.2, s.budget_majority, .4, color=C['maj'], label='budget-majority')
        ax.bar(x+.2, s.REACH_pin, .4, color=C['pin'], label='REACH-pin')
        ax.set_xticks(x); ax.set_xticklabels([f'{v:g}' for v in s.value])
        ax.set_xlabel({'m': 'labels per source $m$', 'lam': 'unseen placement $\\lambda$'}[by])
        ax.set_ylabel('oracle regret')
    axs[0].set_title('Classifier worlds: pin loses at $m$=50', fontsize=7)
    axs[1].set_title('Classifier worlds by separation', fontsize=7)
    axs[0].legend(frameon=False)
    save(fig, 'figA6_pin_limits.pdf')


if __name__ == '__main__':
    table_attribution()
    for f in (figA1, figA2, figA3, figA4, figA5, figA6):
        f()
