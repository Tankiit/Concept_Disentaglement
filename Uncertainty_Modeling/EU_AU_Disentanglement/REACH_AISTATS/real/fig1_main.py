"""Figure 1 (main/teaser) for the acquisition paper — the acquisition question on a
real CAMELYON17 patch. Generated from saved results + the pinned dataset only.

    python -m real.fig1_main

Layout: (A) a real deployment patch from the recorded CAMELYON17 pilot with the
two questions as color-keyed tags BELOW the framed image (never inside it);
(B) ONE grouped chart: rows = targets P(known hospital) / P(hospital 2) /
P(tumor), columns = what is acquired (hospital IDs / tumor labels / unlabeled
patches); each cell is an open circle (width before) with an arrow to a filled
circle (width after 2,048 acquisitions), % contraction above moved cells,
'unchanged' on flat cells. Every number is read from recorded artifacts; the
patch is extracted from the pinned HF mirror (provenance:
paper/figures/fig1_patch_camelyon.json):

  image   wltjr1007/Camelyon17-WILDS @ d784d534…, data/test-00002-of-00004.parquet
          row 1007 (label 1, center 2, patient 42, slide 22) — a row of the
          recorded pilot's own selected_images.csv (role=fit, source=2 candidate).
  numbers workbook/camelyon17_acquisition_20261030T092023425816Z/acquisition_summary.csv
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WB = ROOT / 'workbook'
FIG = ROOT / 'paper' / 'figures'

C_KNOWN = '#0072B2'   # P(known hospital) row
C_CAND = '#56B4E9'    # P(hospital 2) row
C_LAB = '#D55E00'     # P(tumor) row
C_FLAT = '#999999'

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8,
                     'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7.5,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42})

pilot = pd.read_csv(WB / 'camelyon17_acquisition_20261030T092023425816Z' / 'acquisition_summary.csv')
patch = mpimg.imread(FIG / 'fig1_patch_camelyon.png')

ROWS = [('known_membership', 'P(known hospital)', C_KNOWN),
        ('candidate_membership', 'P(hospital 2)', C_CAND),
        ('reference_label', 'P(tumor)', C_LAB)]
COLS = [('source_id', 'acquire hospital IDs'),
        ('target_labels', 'acquire tumor labels'),
        ('inputs', 'acquire unlabeled patches')]


def width(action, target, budget):
    return pilot[(pilot.action == action) & (pilot.target == target)
                 & (pilot.budget == budget)].mean_width.iloc[0]


fig = plt.figure(figsize=(7.16, 2.35))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 2.55], wspace=0.10,
                      left=0.02, right=0.995, top=0.80, bottom=0.17)

# ---------------------------------------------------------------- panel A
axA = fig.add_subplot(gs[0])
axA.imshow(patch, interpolation='nearest')
axA.set_xticks([])
axA.set_yticks([])
for s in axA.spines.values():
    s.set_visible(True)
    s.set_color('#444444')
axA.set_title("A  one real patch — CAMELYON17", loc="left", fontweight="bold", pad=12)
# the two questions, as color-keyed tags well below the framed image
for (y, txt, col) in ((-0.15, 'which hospital?', C_KNOWN), (-0.36, 'tumor present?', C_LAB)):
    axA.text(0.5, y, txt, transform=axA.transAxes, fontsize=8, ha='center', va='center',
             color=col, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.35', fc='white', ec=col, lw=1.0))
axA.text(0.5, -0.60, 'what should you acquire?', transform=axA.transAxes,
         fontsize=8, ha='center', va='center', style='italic')

# ---------------------------------------------------------------- panel B (one grouped bar chart)
CHANC = [('source_id', 'hospital IDs', C_KNOWN),
         ('target_labels', 'tumor labels', C_LAB),
         ('inputs', 'unlabeled patches', C_FLAT)]

axB = fig.add_subplot(gs[1])
n_ch = len(CHANC)
for i, (target, _, _) in enumerate(ROWS):
    y0 = len(ROWS) - 1 - i
    for j, (action, _, ch_col) in enumerate(CHANC):
        w0 = width(action, target, 0)
        w2 = width(action, target, 2048)
        red = max(0.0, 1 - w2 / w0) * 100
        y = y0 + (j - (n_ch - 1) / 2) * 0.26
        axB.barh(y, red, height=0.22, color=ch_col, zorder=3)
        if red > 0.5:
            axB.annotate(f'$-${red:.0f}%', (red, y), xytext=(4, 0),
                         textcoords='offset points', fontsize=7, va='center',
                         color=ch_col, fontweight='bold')
axB.set_xlim(0, 34)
axB.set_ylim(-0.65, 2.65)
axB.set_yticks([2, 1, 0])
axB.set_yticklabels([r[1] for r in ROWS])
axB.set_xticks([0, 10, 20, 30])
axB.set_xlabel('interval-width contraction after 2,048 acquisitions (%)', fontsize=7.5)
axB.set_title('B  what 2,048 acquisitions change', loc='left', fontweight='bold', pad=8)
for j, (_, lab, ch_col) in enumerate(CHANC):
    axB.barh([], [], height=0.22, color=ch_col, label=lab)
axB.legend(frameon=False, fontsize=7, loc='lower right', handlelength=1.2,
           handleheight=0.9, borderaxespad=0.3, labelspacing=0.35)

# ---------------------------------------------------------------- output
FIG.mkdir(parents=True, exist_ok=True)
for ext in ('pdf', 'png'):
    out = FIG / f'fig1_main.{ext}'
    fig.savefig(out, bbox_inches='tight', dpi=300)
    print('wrote', out)
