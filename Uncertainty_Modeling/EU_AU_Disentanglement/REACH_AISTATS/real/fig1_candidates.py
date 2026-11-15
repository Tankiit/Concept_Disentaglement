"""Fig 1 candidates (Options 1-3) for the acquisition teaser, from recorded results.

    python -m real.fig1_candidates

Writes paper/figures/fig1_opt{1,2,3}.{pdf,png}.  All numbers from
workbook/camelyon17_acquisition_20261030T092023425816Z/acquisition_summary.csv;
patch + provenance: paper/figures/fig1_patch_camelyon.{png,json}.

Grid semantics (what you acquire x what you ask):
  "which hospital?" collapses the two membership targets to P(hospital 2) — the
  candidate-hospital question this patch actually poses (known-membership -18%
  goes in the caption).  Off-diagonal cells are exactly 0.0% by construction.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WB = ROOT / 'workbook'
FIG = ROOT / 'paper' / 'figures'

C_HOSP = '#0072B2'   # question: which hospital?
C_TUMOR = '#D55E00'  # question: tumor present?
C_FLAT = '#999999'

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7.5,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42})

pilot = pd.read_csv(WB / 'camelyon17_acquisition_20261030T092023425816Z' / 'acquisition_summary.csv')
patch = mpimg.imread(FIG / 'fig1_patch_camelyon.png')

ACQS = [('source_id', 'hospital IDs'), ('target_labels', 'tumor labels'),
        ('inputs', 'unlabeled patches')]
QUESTIONS = [('candidate_membership', 'which hospital?', C_HOSP),
             ('reference_label', 'tumor present?', C_TUMOR)]
CLAIM = 'Uncertainty shrinks only for the question the acquisition answers'


def width(action, target, budget):
    return pilot[(pilot.action == action) & (pilot.target == target)
                 & (pilot.budget == budget)].mean_width.iloc[0]


def contraction(action, target):
    w0, w2 = width(action, target, 0), width(action, target, 2048)
    return max(0.0, 1 - w2 / w0) * 100


def panel_A(ax):
    ax.imshow(patch, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color('#444444')
    ax.set_title('A  one real patch — CAMELYON17', loc='left',
                 fontweight='bold', pad=10)
    for (y, (_, q, col)) in zip((-0.16, -0.42), [(0, q, c) for _, q, c in QUESTIONS]):
        ax.text(0.5, y, q, transform=ax.transAxes, fontsize=8.5, ha='center',
                va='center', color=col, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.38', fc='white', ec=col, lw=1.1))


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(FIG / f'{name}.{ext}', bbox_inches='tight', dpi=300)
        print('wrote', FIG / f'{name}.{ext}')
    plt.close(fig)


# ================================================================ Option 1: the grid
fig = plt.figure(figsize=(7.16, 2.75))
gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 1.3, 1.3], wspace=0.42,
                      left=0.075, right=0.985, top=0.72, bottom=0.20)
fig.suptitle(CLAIM + '  (CAMELYON17, after 2,048 acquisitions)',
             fontsize=9.5, y=0.97)
panel_A(fig.add_subplot(gs[0]))
for j, (target, q, col) in enumerate(QUESTIONS):
    ax = fig.add_subplot(gs[j + 1])
    for i, (action, _) in enumerate(ACQS):
        red = contraction(action, target)
        y = len(ACQS) - 1 - i
        ax.barh(y, red, height=0.58, color=col, zorder=3)
        if red >= 0.5:
            ax.annotate(f'−{red:.0f}%', (red, y), xytext=(4, 0),
                        textcoords='offset points', fontsize=8, va='center',
                        color=col, fontweight='bold')
        else:
            txt = '≈0%' if red >= 0.05 else '0%'
            ax.annotate(txt, (0.6, y), fontsize=7.5, va='center', color=C_FLAT)
    ax.set_xlim(0, 33)
    ax.set_ylim(-0.6, 2.6)
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels([a[1] for a in ACQS] if j == 0 else ['', '', ''])
    ax.set_xticks([0, 10, 20, 30])
    ax.set_xlabel('interval-width contraction (%)', fontsize=7.5)
    ax.set_title(f'{"B" if j == 0 else "C"}  {q}', loc='left', color=col,
                 fontweight='bold', pad=6)
save(fig, 'fig1_opt1')

# ================================================================ Option 2: before/after intervals
fig = plt.figure(figsize=(7.16, 3.0))
gs = fig.add_gridspec(2, 2, width_ratios=[0.62, 2.0], hspace=0.55, wspace=0.18,
                      left=0.135, right=0.97, top=0.80, bottom=0.11)
fig.suptitle(CLAIM + '  (CAMELYON17, after 2,048 acquisitions)',
             fontsize=9.5, y=0.98)
panel_A(fig.add_subplot(gs[:, 0]))
rows_lab = ['before', '+ hospital IDs', '+ tumor labels', '+ unlabeled patches']
for j, (target, q, col) in enumerate(QUESTIONS):
    ax = fig.add_subplot(gs[j, 1])
    for i, (action, _) in enumerate([(None, 'before')] + ACQS):
        w = width(action, target, 2048) if action else width('source_id', target, 0)
        y = len(rows_lab) - 1 - i
        red = contraction(action, target) if action else 0.0
        ax.barh(y, w, height=0.6,
                color='#cccccc' if action is None else col,
                alpha=1.0 if (action is None or red >= 0.5) else 0.45, zorder=3)
        tag = f'{w:.2f}' if action is None else (
            f'{w:.2f}   (−{red:.0f}%)' if red >= 0.5 else f'{w:.2f}   (0%)')
        ax.annotate(tag, (w, y), xytext=(4, 0), textcoords='offset points',
                    fontsize=7.5 if action is None else 8, va='center',
                    color='#555555' if action is None else (col if red >= 0.5 else C_FLAT),
                    fontweight='bold' if action is not None and red >= 0.5 else 'normal')
    ax.set_xlim(0, 1.32)
    ax.set_ylim(-0.6, 3.6)
    ax.set_yticks(range(len(rows_lab)))
    ax.set_yticklabels(list(reversed(rows_lab)), fontsize=7.5)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel('interval width', fontsize=7.5)
    ax.set_title(f'{"B" if j == 0 else "C"}  {q}', loc='left', color=col,
                 fontweight='bold', pad=5)
save(fig, 'fig1_opt2')

# ================================================================ Option 3: one-line strip
fig = plt.figure(figsize=(7.16, 1.45))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 10)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title(CLAIM, fontsize=9.5, fontweight='bold', pad=4)
axin = ax.inset_axes([0.02, 0.16, 0.72, 0.72])
axin.imshow(patch, interpolation='nearest')
axin.set_xticks([])
axin.set_yticks([])
for s in axin.spines.values():
    s.set_visible(True)
    s.set_color('#444444')
X_BOX, X_BAR0, BAR_SPAN = 1.35, 5.0, 3.6
for k, (target, q, col) in enumerate(QUESTIONS):
    y = 0.70 - k * 0.44
    w_after = width('source_id', target, 2048)
    red = contraction('source_id', target)
    ax.text(X_BOX, y, q, fontsize=8.5, ha='center', va='center', color=col,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.35', fc='white', ec=col, lw=1.1))
    ax.annotate('', xy=(X_BAR0 - 0.12, y), xytext=(X_BOX + 0.72, y),
                arrowprops=dict(arrowstyle='-|>', color='#666666', lw=1.0))
    ax.text((X_BOX + X_BAR0) / 2 + 0.18, y + 0.13, 'buy hospital IDs', fontsize=7,
            ha='center', va='bottom', color='#555555', style='italic')
    ax.barh(y, w_after / 1.0 * BAR_SPAN, left=X_BAR0, height=0.16,
            color=col if red >= 0.5 else '#bbbbbb', zorder=3)
    ax.text(X_BAR0 + BAR_SPAN + 0.15, y,
            (f'width {w_after:.2f}  (−{red:.0f}%)' if red >= 0.5
             else f'width {w_after:.2f}  (0%)'),
            fontsize=8, va='center',
            color=col if red >= 0.5 else C_FLAT,
            fontweight='bold' if red >= 0.5 else 'normal')
ax.text(X_BAR0 - 0.35, 0.06, '0', fontsize=6.5, ha='right', color='#888888')
ax.text(X_BAR0 + BAR_SPAN + 0.05, 0.06, 'interval width 1.0', fontsize=6.5,
        ha='left', color='#888888')
save(fig, 'fig1_opt3')
