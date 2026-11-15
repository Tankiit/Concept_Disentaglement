"""Figure 1 (main/teaser) for the acquisition paper — the acquisition question on a
real CAMELYON17 patch. Generated from saved results + the pinned dataset only.

    python -m real.fig1_main

Outputs (paper/figures/), all at exact print sizes, saved WITHOUT tight bbox:

  fig1_main.{pdf,png}          combined single-column teaser, 3.25 x 3.85 in
                               (panels stacked; included unscaled via
                               figure[t] + width=\\linewidth)
  fig1_a_patch.{pdf,png}       panel (a) alone, 2.35 x 2.10 in
  fig1_b_contraction.{pdf,png} panel (b) alone, 4.50 x 2.10 in

The two single-panel files sum to 6.85 in ~= the CVPR full text width (6.875),
so they can sit side by side in a figure* unscaled; being vector, they also
rescale cleanly for any other layout the LaTeX side chooses.

Panel (a): a real deployment patch from the recorded CAMELYON17 pilot, framed,
with the two questions as color-keyed tags BELOW the image (never inside it).
Panel (b): ONE grouped chart — rows = targets P(seen hosp.) / P(hosp. 2) /
P(tumor), columns = what is acquired (IDs / labels / unlabeled). Every cell is
drawn: moving bars carry direct end labels (channel + % width contraction, no
legend), flat cells show a gray tick + gray 0%. n, the dataset name, the unit
spelled out, and the framing question live in the LaTeX caption, not in the
figure (the % unit rides on the last x-tick). Every number is read from
recorded artifacts; the patch is extracted from the pinned HF mirror
(provenance: paper/figures/fig1_patch_camelyon.json):

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
from matplotlib.patches import Rectangle

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WB = ROOT / 'workbook'
FIG = ROOT / 'paper' / 'figures'

C_MEMB = '#0072B2'   # hospital side: membership targets + the IDs channel
C_LAB = '#D55E00'    # tumor side: P(tumor) target + the labels channel
C_FLAT = '#999999'   # unlabeled channel / flat cells
C_TEXT_FLAT = '#666666'

plt.rcParams.update({'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7,
                     'xtick.labelsize': 6.5, 'ytick.labelsize': 7,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42})

pilot = pd.read_csv(WB / 'camelyon17_acquisition_20261030T092023425816Z' / 'acquisition_summary.csv')
patch = mpimg.imread(FIG / 'fig1_patch_camelyon.png')

ROWS = [('known_membership', 'P(seen hosp.)'),
        ('candidate_membership', 'P(hosp. 2)'),
        ('reference_label', 'P(tumor)')]
ROWCOL = {'known_membership': C_MEMB, 'candidate_membership': C_MEMB,
          'reference_label': C_LAB}
CHANC = [('source_id', 'IDs'), ('target_labels', 'labels'), ('inputs', 'unlabeled')]
CHCOL = {'source_id': C_MEMB, 'target_labels': C_LAB, 'inputs': C_FLAT}


def width(action, target, budget):
    return pilot[(pilot.action == action) & (pilot.target == target)
                 & (pilot.budget == budget)].mean_width.iloc[0]


def panel_a(ax, img_in, title=True):
    """Panel (a): framed real patch + the two question tags below it."""
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if title:
        ax.set_title('(a) CAMELYON17 patch', loc='left', fontweight='bold', pad=5)
    fig = ax.figure
    pos = ax.get_position()
    w_in = pos.width * fig.get_size_inches()[0]
    h_in = pos.height * fig.get_size_inches()[1]
    # image box square in inches, top-anchored so it can never run into the title
    s = min(img_in, 0.76 * h_in, 0.80 * w_in)
    sx, sy = s / w_in, s / h_in
    box = [0.5 - sx / 2, 0.5 + sx / 2, 0.97 - sy, 0.97]
    ax.imshow(patch, extent=box, interpolation='nearest', zorder=1)
    ax.add_patch(Rectangle((box[0], box[2]), sx, sy, fill=False, ec='#444444',
                           lw=0.8, zorder=2))
    # the two questions, as color-keyed tags well below the framed image
    y_tag = box[2] - 0.09
    for (x, ha, txt, col) in ((0.5 - 0.06, 'right', 'hospital?', C_MEMB),
                              (0.5 + 0.06, 'left', 'tumor?', C_LAB)):
        ax.text(x, y_tag, txt, fontsize=7, ha=ha, va='top', color=col,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.30', fc='white', ec=col, lw=0.8))


def panel_b(ax, title=True):
    """Panel (b): grouped width-contraction bars with direct labels, no legend."""
    n_ch = len(CHANC)
    for i, (target, _) in enumerate(ROWS):
        y0 = len(ROWS) - 1 - i
        for j, (action, ch_name) in enumerate(CHANC):
            w0 = width(action, target, 0)
            w2 = width(action, target, 2048)
            red = max(0.0, 1 - w2 / w0) * 100
            y = y0 + (j - (n_ch - 1) / 2) * 0.27
            if red > 0.5:
                ax.barh(y, red, height=0.20, color=CHCOL[action], zorder=3)
                ax.annotate(f'{ch_name} $-${red:.0f}%', (red, y), xytext=(3, 0),
                            textcoords='offset points', fontsize=6.5,
                            va='center',
                            color=CHCOL[action] if action != 'inputs' else C_TEXT_FLAT,
                            fontweight='bold')
            else:
                ax.barh(y, 0.7, height=0.08, color='#BBBBBB', zorder=3)
                ax.annotate('0%', (1.2, y), fontsize=6, va='center',
                            color=C_FLAT, zorder=3)
    ax.set_xlim(0, 38)
    ax.set_ylim(-0.70, 2.70)
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels([r[1] for r in ROWS])
    for tick, (target, _) in zip(ax.get_yticklabels(), ROWS):
        tick.set_color(ROWCOL[target])
    ax.set_xticks([0, 10, 20, 30])
    ax.set_xticklabels(['0', '10', '20', '30%'])
    if title:
        ax.set_title('(b) Width contraction', loc='left', fontweight='bold', pad=5)


# ------------------------------------------------- combined single-column teaser
fig = plt.figure(figsize=(3.25, 3.85))
gs = fig.add_gridspec(2, 1, height_ratios=[1.12, 1.0], hspace=0.30,
                      left=0.250, right=0.965, top=0.955, bottom=0.095)
axA = fig.add_subplot(gs[0])
axB = fig.add_subplot(gs[1])
panel_a(axA, img_in=1.16)
panel_b(axB)

# ------------------------------------------------- panel (a) alone (no title:
# the LaTeX side supplies panel letters/captions when combining)
fig_a = plt.figure(figsize=(2.35, 2.10))
ax_a = fig_a.add_axes([0.03, 0.02, 0.94, 0.96])
panel_a(ax_a, img_in=1.50, title=False)

# ------------------------------------------------- panel (b) alone (no title)
fig_b = plt.figure(figsize=(4.50, 2.10))
ax_b = fig_b.add_axes([0.215, 0.125, 0.755, 0.85])
panel_b(ax_b, title=False)

# ------------------------------------------------- output
FIG.mkdir(parents=True, exist_ok=True)
for name, f in (('fig1_main', fig), ('fig1_a_patch', fig_a),
                ('fig1_b_contraction', fig_b)):
    for ext in ('pdf', 'png'):
        out = FIG / f'{name}.{ext}'
        f.savefig(out, dpi=600 if ext == 'png' else None)  # exact canvas: no tight bbox
        print('wrote', out)
