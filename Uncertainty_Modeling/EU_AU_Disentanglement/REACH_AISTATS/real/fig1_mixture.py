"""Figure 1 candidate (mixture-first teaser): deployment as a stacked mixture of
sources; every target is a ratio inside one bar. No new experiment — every
number is read from the recorded acquisition tutorial (paper/acquisition_tutorial):

  panel (a) rows A_s  = 'mnist source-cell laws' table, image-derived regime
           (numerical_tables.tex; s0 = digits {4,6}, s1 = {2,7}, candidate
           s2 = held-out digits), deployment w = (0.45, 0.25, 0.30) and probe
           = argmax 0.5 A_0c + 0.5 A_1c = cell 0 (recorded).
  panel (b) worked cell of tutorial section 'A worked source-membership
           calculation': A_{:,c} = (0.40, 0.20, 0.30), w = (0.45, 0.25, 0.30)
           -> z_c = (0.18, 0.05, 0.09), p_c = 0.32, known membership
           0.18/0.23 = 0.78261, candidate membership 0.09/0.32 = 0.28125.
           Second stacking = the indistinguishable family eq:family
           w(t) = (0.60 - t/2, 0.40 - t/2, t) at t = 0.60: z'_c =
           (0.12, 0.02, 0.18), SAME p_c; candidate 0.5625, known 0.857.
  panel (c) the same generating probe bar, annotated with which part of the
           picture each acquisition channel reveals.

Drawn at exactly the CVPR full text width (6.875 in), saved without tight
bbox; fonts: body 7 / titles 8 / ticks 6.5-7, pdf.fonttype 42.

    python -m real.fig1_mixture
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'paper' / 'figures'

S_COLS = ['#0072B2', '#56B4E9', '#D55E00']   # s0, s1 known; s2 candidate
C_NOTE = '#444444'

plt.rcParams.update({'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7,
                     'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42})

# ---------------------------------------------------------------- recorded world
# mnist source-cell laws (image-derived), numerical_tables.tex
A = np.array([[0.49500, 0.49010, 0.00990, 0.00500],      # s0: digits {4,6}
              [0.01480, 0.01970, 0.49990, 0.46560],      # s1: digits {2,7}
              [0.21733, 0.18467, 0.36107, 0.23693]])     # s2: held-out digits
W = np.array([0.45, 0.25, 0.30])                          # tutorial deployment
PROBE = int(np.argmax(0.5 * A[0] + 0.5 * A[1]))           # = 0, recorded
Z = W[:, None] * A                                        # joint masses z_sc
P = Z.sum(axis=0)                                         # deployment cell law

# worked cell (tutorial 11.3) and the indistinguishable family at t=0.60
AC = np.array([0.40, 0.20, 0.30])
Z1 = W * AC                                               # (0.18, 0.05, 0.09)
W2 = np.array([0.60 - 0.60 / 2, 0.40 - 0.60 / 2, 0.60])   # (0.30, 0.10, 0.60)
Z2 = W2 * AC                                              # (0.12, 0.02, 0.18)

# numeric checks against the tutorial's own numbers
assert np.allclose(A[0] + A[1], 2 * np.array([0.25490, 0.25490, 0.25490, 0.23530]))
assert abs(Z.sum() - 1) < 1e-9 and PROBE == 0
assert np.allclose(Z1, [0.18, 0.05, 0.09]) and abs(Z1.sum() - 0.32) < 1e-12
assert abs(Z2.sum() - 0.32) < 1e-12                       # same height by family
K1, C1 = Z1[0] / (Z1[0] + Z1[1]), Z1[2] / Z1.sum()        # 0.78261, 0.28125
K2, C2 = Z2[0] / (Z2[0] + Z2[1]), Z2[2] / Z2.sum()
assert abs(C1 - 0.28125) < 1e-12 and abs(K1 - 0.78261) < 1e-5

fig = plt.figure(figsize=(6.875, 2.30))
gs = fig.add_gridspec(1, 3, width_ratios=[1.18, 0.80, 1.02], wspace=0.30,
                      left=0.055, right=0.985, top=0.86, bottom=0.23)

# ---------------------------------------------------------------- (a) the mixture
axA = fig.add_subplot(gs[0])
bottom = np.zeros(A.shape[1])
for s in range(3):
    axA.bar(np.arange(4), Z[s], width=0.62, bottom=bottom, color=S_COLS[s],
            edgecolor='white', linewidth=0.4, zorder=3)
    bottom += Z[s]
axA.bar([PROBE], [P[PROBE]], width=0.70, facecolor='none', edgecolor='black',
        linewidth=0.9, zorder=4)
axA.annotate('probe $c^*$', (PROBE, P[PROBE]), xytext=(0, 4),
             textcoords='offset points', ha='center', fontsize=6.5)
axA.set_xticks(range(4))
axA.set_xticklabels([f'cell {c + 1}' for c in range(4)])
axA.set_yticks([0, 0.1, 0.2, 0.3])
axA.set_ylim(0, 0.345)
axA.set_title('(a) deployment = mixture,  $p_c=\\sum_s w_s A_{sc}$',
              loc='left', fontweight='bold', pad=5)
handles = [plt.Rectangle((0, 0), 1, 1, fc=S_COLS[s], ec='none') for s in range(3)]
axA.legend(handles, ['$s_0$', '$s_1$', '$s_2$'], ncol=3, frameon=False,
           fontsize=6.5, handlelength=1.0, handleheight=0.8, columnspacing=1.0,
           loc='lower right', borderaxespad=0.2)

# ------------------------------------------------- (b) same height, two mixtures
axB = fig.add_subplot(gs[1])
for x, zz, wl in ((0, Z1, '$w$'), (1, Z2, "$w'$")):
    bottom = 0.0
    for s in range(3):
        axB.bar([x], [zz[s]], width=0.50, bottom=bottom, color=S_COLS[s],
                edgecolor='white', linewidth=0.4, zorder=3)
        bottom += zz[s]
    axB.text(x, 0.022, wl, ha='center', va='center', fontsize=7,
             color='white', fontweight='bold', zorder=4)
axB.axhline(0.32, color=C_NOTE, lw=0.7, ls=(0, (3, 2)), zorder=2)
axB.text(0.5, 0.345, 'same height $p_{c^*}$', ha='center', va='bottom',
         fontsize=6.5, color=C_NOTE)
for x, wc, kc in ((0, C1, K1), (1, C2, K2)):
    axB.text(x, -0.042, f'cand. {wc:.2f}', ha='center', va='top', fontsize=7,
             color=S_COLS[2], fontweight='bold')
    axB.text(x, -0.104, f'known {kc:.2f}', ha='center', va='top', fontsize=6,
             color=C_NOTE)
axB.set_xlim(-0.55, 1.55)
axB.set_ylim(0, 0.42)
axB.set_xticks([])
axB.set_yticks([])
axB.set_title('(b) same $p_c$, different mixture', loc='left',
              fontweight='bold', pad=5)

# ------------------------------------------------- (c) what each channel reveals
axC = fig.add_subplot(gs[2])
axC.bar([0], [Z1[0]], width=0.42, color=S_COLS[0], edgecolor='white',
        linewidth=0.4, zorder=3)
axC.bar([0], [Z1[1]], width=0.42, bottom=Z1[0], color=S_COLS[1],
        edgecolor='white', linewidth=0.4, zorder=3)
axC.bar([0], [Z1[2]], width=0.42, bottom=Z1[0] + Z1[1], color=S_COLS[2],
        edgecolor='white', linewidth=0.4, zorder=3)
# hatch: the label-rate part inside one segment
axC.bar([0], [0.08], width=0.42, bottom=0.0, facecolor='none', hatch='////',
        edgecolor='white', linewidth=0.0, zorder=4)
axC.axhline(0.32, color=C_NOTE, lw=0.7, ls=(0, (3, 2)), zorder=2)
axC.set_xlim(-0.5, 3.1)
axC.set_ylim(0, 0.40)
axC.set_xticks([])
axC.set_yticks([])
axC.set_title('(c) what each channel reveals', loc='left',
              fontweight='bold', pad=5)
# callouts: x-position of the text anchor in data coords, feature on the bar
for (y, txt) in ((0.352, 'inputs: height $p_c$'),
                 (0.13, 'IDs: weights $w_s$'),
                 (0.055, 'labels: rate $\\theta_{sc}$'),
                 (0.008, 'references: rows $A_{s\\cdot}$')):
    axC.annotate(txt, xy=(0.28, y), xytext=(0.62, y), fontsize=6.5,
                 color=C_NOTE, va='center',
                 arrowprops=dict(arrowstyle='-', lw=0.6, color=C_NOTE))

# ---------------------------------------------------------------- output + checks
FIG.mkdir(parents=True, exist_ok=True)
for ext in ('pdf', 'png'):
    out = FIG / f'fig1_mixture.{ext}'
    fig.savefig(out, dpi=600 if ext == 'png' else None)  # exact canvas
    print('wrote', out)
print(f'probe cell {PROBE}; p = {np.round(P, 5)}')
print(f'panel b: cand {C1:.5f} -> {C2:.5f}; known {K1:.5f} -> {K2:.5f}')
