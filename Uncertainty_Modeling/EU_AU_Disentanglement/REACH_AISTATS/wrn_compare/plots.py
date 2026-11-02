"""Figures from saved results, with seed variability when replicated."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_results(out):
    out = Path(out)
    data = pd.read_csv(out/'summary_per_seed.csv')
    methods = [('density', '#0072B2', 'o', '-'), ('EU', '#D55E00', 's', '--'),
               ('ensemble_entropy', '#009E73', '^', ':'), ('WRN_entropy', '#CC79A7', 'D', '-.')]
    kinds = list(data.transformation.unique())
    with plt.rc_context({'font.size':9, 'axes.spines.top':False, 'axes.spines.right':False,
                         'pdf.fonttype':42, 'savefig.bbox':'tight'}):
        fig, axes = plt.subplots(2, len(kinds), figsize=(10,5.6), squeeze=False)
        for j, kind in enumerate(kinds):
            for name, color, marker, style in methods:
                block = data[(data.transformation==kind)&(data.score==name)]
                for row, metric in enumerate(('rejection_rate','shift_AUROC')):
                    group = block.groupby('strength')[metric]
                    mean, sd = group.mean(), group.std()
                    axes[row,j].plot(mean.index, mean, label=name.replace('_',' '),
                                     color=color, marker=marker, linestyle=style, markersize=4)
                    if sd.notna().any():
                        axes[row,j].fill_between(mean.index, (mean-sd).clip(0,1),
                                                 (mean+sd).clip(0,1), color=color, alpha=.10)
                    axes[row,j].set_ylim(-.025,1.025)
                    axes[row,j].set_xlabel('Transformation strength')
            axes[0,j].set_title(kind.replace('_',' '))
            axes[1,j].axhline(.5,color='.6',linewidth=.7)
        axes[0,0].set_ylabel('Rejection rate (fixed ID threshold)')
        axes[1,0].set_ylabel('Transformed-vs-clean AUROC')
        axes[0,-1].legend(frameon=False, fontsize=8)
        title = f'CIFAR-10 / frozen WRN: {data.seed.nunique()} seed(s)'
        title += '; shading = seed SD' if data.seed.nunique()>1 else '; exploratory smoke run'
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        fig.savefig(out/'comparison.pdf'); fig.savefig(out/'comparison.png',dpi=170)
        plt.close(fig)
        previews = sorted(out.glob('preview_*.npz'))
        fig, axes = plt.subplots(len(previews), 5, figsize=(7,1.25*len(previews)), squeeze=False)
        for i, path in enumerate(previews):
            with np.load(path, allow_pickle=False) as f:
                images = f['images']
            for j in range(5):
                axes[i,j].imshow(images[j % len(images)])
                axes[i,j].set_xticks([]); axes[i,j].set_yticks([])
            axes[i,0].set_ylabel(path.stem.removeprefix('preview_').replace('_',' '),
                                 rotation=0, ha='right', va='center', fontsize=8)
        fig.tight_layout()
        fig.savefig(out/'transformed_examples.pdf'); fig.savefig(out/'transformed_examples.png',dpi=140)
        plt.close(fig)
