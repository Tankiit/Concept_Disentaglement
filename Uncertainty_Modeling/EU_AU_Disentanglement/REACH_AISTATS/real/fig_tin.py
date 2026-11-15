"""Main-text Tiny ImageNet floor figure (contribution 2/3: structural floor).

    python -m real.fig_tin

Reads ONLY recorded artifacts (workbook/TinyImageNet_REACH_Results_v2full/,
delivered by the collaborator 12 Nov 2026, copied verbatim from Downloads):

  summary.csv                          mean/min/max width per regime x target x
                                       arm x budget (6 repeats)
  intervals.csv                        per-repeat rows incl. status
  exact_identification_benchmarks.csv  exact input-only widths (the floors)

Outputs paper/figures/fig_tin_floor.{pdf,png} at exact print size 6.5 x 2.2 in
(full text width of the 11pt margin=1in article class), saved WITHOUT tight
bbox. Two panels, known-membership target only: (a) image-derived (separable)
sources, (b) confounded sources. Curves: source identities (house blue) and
unlabeled inputs (house grey); shaded = min-max over the 6 repeats; dashed =
exact input-only floor (0.0 in (a), 0.8628 in (b)); hollow markers where at
least half of a budget's intervals were returned by the undefined-denominator
policy; direct end labels, no legend (house style).
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'workbook' / 'TinyImageNet_REACH_Results_v2full'
FIG = ROOT / 'paper' / 'figures'

C_MEMB = '#0072B2'   # identity / membership side (house palette, fig1_main)
C_FLAT = '#999999'   # unlabeled channel
C_TEXT = '#666666'

plt.rcParams.update({'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7,
                     'xtick.labelsize': 6.5, 'ytick.labelsize': 7,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42})

TARGET = 'known_membership'
XMAP = {0: 0, 100: 2, 1000: 3, 10000: 4, 100000: 5}

summary = pd.read_csv(SRC / 'summary.csv')
intervals = pd.read_csv(SRC / 'intervals.csv')
exact = pd.read_csv(SRC / 'exact_identification_benchmarks.csv')

policy_frac = (intervals[intervals.target == TARGET]
               .assign(policy=lambda d: d.status == 'undefined_denominator_policy')
               .groupby(['regime', 'arm', 'budget']).policy.mean())

fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.2), sharey=True)

for ax, regime, title in zip(axes, ['image_derived', 'confounded'],
                             ['(a) sources separable', '(b) sources confounded']):
    for arm, color, end_label in [('identities', C_MEMB, 'source IDs'),
                                  ('inputs', C_FLAT, 'unlabeled')]:
        d = (summary[(summary.regime == regime) & (summary.target == TARGET)
                     & (summary.arm == arm)].sort_values('budget'))
        x = d.budget.map(XMAP)
        y = d.mean_width
        ax.plot(x, y, color=color, lw=1.2, zorder=3)
        ax.fill_between(x, d.min_width, d.max_width, color=color, alpha=0.15,
                        lw=0, zorder=2)
        hollow = d.apply(lambda r: policy_frac.loc[(regime, arm, r.budget)] >= .5,
                         axis=1)
        for xi, yi, h in zip(x, y, hollow):
            ax.plot(xi, yi, marker='o', ms=3,
                    mfc=('white' if h else color), mec=color, mew=.8,
                    zorder=4)
        if regime == 'confounded':       # end labels once, right panel only
            ax.text(x.iloc[-1] + .18, y.iloc[-1], end_label, color=color,
                    fontsize=7, va='center', ha='left')
    floor = float(exact[(exact.regime == regime)
                        & (exact.target == TARGET)].width.iloc[0])
    if floor > 0:
        ax.axhline(floor, ls='--', lw=0.8, color='k', zorder=2)
        ax.text(4.95, floor + 0.022, 'exact input-only floor', fontsize=6,
                color='k', va='bottom', ha='right', style='italic')
    ax.set_title(title)
    ax.set_xticks([0, 2, 3, 4, 5])
    ax.set_xticklabels(['0', r'$10^2$', r'$10^3$', r'$10^4$', r'$10^5$'])
    ax.set_xlabel('added observations')
    ax.set_xlim(-0.25, 6.9)

axes[0].set_ylabel('bound width')
axes[0].set_ylim(0, 1.05)
axes[0].set_yticks([0, .25, .5, .75, 1])

fig.subplots_adjust(left=0.075, right=0.985, bottom=0.16, top=0.87,
                    wspace=0.06)
for ext in ('pdf', 'png'):
    fig.savefig(FIG / f'fig_tin_floor.{ext}', dpi=300)

# --- programmatic QA: canvas size + the numbers the caption quotes ---------
from pypdf import PdfReader
mb = PdfReader(FIG / 'fig_tin_floor.pdf').pages[0].mediabox
print(f'MediaBox {float(mb.width):.1f} x {float(mb.height):.1f} pt '
      f'({float(mb.width) / 72:.2f} x {float(mb.height) / 72:.2f} in)')
for regime in ('image_derived', 'confounded'):
    g = summary[(summary.regime == regime) & (summary.target == TARGET)]
    for arm in ('identities', 'inputs'):
        e = g[(g.arm == arm) & (g.budget == 100000)].mean_width.iloc[0]
        print(f'{regime:13s} {arm:11s} @1e5 mean width {e:.4f}')
km = intervals[(intervals.target == TARGET)
               & (intervals.regime == 'confounded') & (intervals.arm == 'inputs')]
print('confounded/inputs policy rows:',
      int((km.status == 'undefined_denominator_policy').sum()), 'of', len(km),
      '| widths', round(km.width.min(), 4), '-', round(km.width.max(), 4))
print('containment overall:', intervals.contains_truth.all(),
      '| rows:', len(intervals))
