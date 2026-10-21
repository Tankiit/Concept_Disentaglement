"""Plot saved original REACH widths; no training or interval recomputation."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'fashion_blood_reach_2026-10-19' / 'reach_rows.csv'
WIDTHS = ['W_cls', 'W_label', 'W_joint', 'W_unk', 'W_unk_joint']
LABELS = [r'$W_{\rm cls}$', r'$W_{\rm label}$', r'$W_{\rm joint}$',
          r'$W_{\rm unk}$', r'$W_{\rm unk,joint}$']
SCENARIOS = ['fashion_05', 'fashion_20', 'blood_05', 'blood_20']
TITLES = ['Fashion • 5% novelty', 'Fashion • 20% novelty',
          'Blood • 5% novelty', 'Blood • 20% novelty']
SETTINGS = ['baseline', 'n_low', 'n_high', 'm_low', 'm_high', 'r_low', 'r_high']
TICKS = ['Baseline', 'n = 250', 'n = 4000', 'm = 50', 'm = 1000', 'r = 10', 'r = 200']


def main():
    rows = pd.read_csv(SOURCE)
    assert len(rows) == 924
    assert np.isfinite(rows[WIDTHS].to_numpy()).all()
    assert rows[WIDTHS].min().min() >= -1e-6
    assert rows[WIDTHS].max().max() <= 1 + 1e-6
    for width, prefix in [('W_cls', 'cls'), ('W_joint', 'joint'),
                          ('W_unk', 'unk'), ('W_unk_joint', 'unk_joint')]:
        np.testing.assert_allclose(rows[width], rows[prefix + '_upper'] -
                                   rows[prefix + '_lower'], atol=1e-6)
    # Average cells within each seed before summarizing independent seed runs.
    per_seed = rows.groupby(['scenario', 'setting', 'region', 'seed'])[WIDTHS].mean()
    per_seed.to_csv(HERE / 'widths_per_seed.csv')
    stats = per_seed.groupby(level=['scenario', 'setting', 'region']).agg(['mean', 'min', 'max'])
    stats.to_csv(HERE / 'width_summary.csv')
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    for region, suffix in [('on_support', ''), ('outlier_cell', '_outlier')]:
        fig, axes = plt.subplots(2, 2, figsize=(15, 7.5), layout='constrained')
        for ax, scenario, title in zip(axes.flat, SCENARIOS, TITLES):
            values = np.array([[stats.loc[(scenario, s, region), (w, 'mean')]
                                for s in SETTINGS] for w in WIDTHS])
            im = ax.imshow(values, vmin=0, vmax=1, cmap='YlOrRd', aspect='auto')
            ax.set(title=title, xticks=range(7), xticklabels=TICKS,
                   yticks=range(5), yticklabels=LABELS)
            ax.tick_params(axis='both', length=0, pad=8)
            for (i, j), value in np.ndenumerate(values):
                ax.text(j, i, f'{value:.3f}', ha='center', va='center',
                        color='white' if value > .78 else '#222222', fontsize=11)
        fig.colorbar(im, ax=axes, shrink=.82, label='Interval width (1 = full [0, 1] range)')
        region_label = 'mean over 10 known-class cells' if not suffix else 'outlier cell only'
        fig.suptitle('Original REACH widths • ' + region_label + '\n'
                     'Mean of 3 seeds; baseline n = 1000, m = 200, r = 50; one budget changes at a time', fontsize=15)
        fig.savefig(HERE / f'reach_widths{suffix}.png', dpi=180)
        fig.savefig(HERE / f'reach_widths{suffix}.pdf')
        plt.close(fig)

    # Separate panels prevent five nearly coincident width curves hiding each other.
    fig, axes = plt.subplots(5, 3, figsize=(13, 13), sharey=True, layout='constrained')
    sweeps = [('n', [250, 1000, 4000]), ('m', [50, 200, 1000]), ('r', [10, 50, 200])]
    colors = ['#2166ac', '#67a9cf', '#b2182b', '#ef8a62']
    for i, (width, label) in enumerate(zip(WIDTHS, LABELS)):
        for j, (budget, x) in enumerate(sweeps):
            ax = axes[i, j]
            settings = [budget + '_low', 'baseline', budget + '_high']
            for scenario, title, color in zip(SCENARIOS, TITLES, colors):
                get = lambda metric: [stats.loc[(scenario, s, 'on_support'), (width, metric)] for s in settings]
                ax.plot(x, get('mean'), marker='o', color=color, label=title,
                        linestyle='--' if scenario.endswith('20') else '-')
                ax.fill_between(x, get('min'), get('max'), color=color, alpha=.10)
            ax.set_xscale('log')
            ax.set_xticks(x, [str(v) for v in x])
            ax.set_ylim(-.025, 1.025)
            ax.grid(alpha=.2)
            if j == 0:
                ax.set_ylabel(label + '\nInterval width')
            if i == 0:
                ax.set_title({'n': 'Deployment samples (n)', 'm': 'Known references / class (m)',
                              'r': 'EMNIST references / letter (r)'}[budget])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=4)
    fig.suptitle('Original REACH widths versus acquisition budget\n'
                 'Lines: mean of 3 seed-level cell averages; shading: seed min–max (not confidence intervals)', fontsize=14)
    fig.savefig(HERE / 'reach_width_sweeps.png', dpi=180)
    fig.savefig(HERE / 'reach_width_sweeps.pdf')
    plt.close(fig)
    print('Validated 924 saved rows; wrote heatmaps, budget sweeps, and seed summaries to', HERE)


if __name__ == '__main__':
    main()
