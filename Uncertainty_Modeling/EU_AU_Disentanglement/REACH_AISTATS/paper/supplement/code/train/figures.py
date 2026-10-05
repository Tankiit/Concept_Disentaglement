"""Fig 3: coverage of REACH vs EMQ + bootstrap, synthetic and real, from saved summaries.

    python -m train.figures --response results/response_s128 --audit results/audit_super4_s50 \
                            --corruption results/audit_super4_corruption --out results/figures
"""
import argparse, json
from pathlib import Path
import pandas as pd


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--response', required=True)
    parser.add_argument('--audit', required=True)
    parser.add_argument('--corruption', default=None)
    parser.add_argument('--out', default='results/figures')
    args = parser.parse_args(argv)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    panels = 3 if args.corruption else 2
    fig, axes = plt.subplots(1, panels, figsize=(4*panels, 3.2))
    syn = pd.DataFrame(json.loads((Path(args.response)/'summary.json').read_text())['probe_coverage'])
    syn = syn[syn.region == 'on_support']
    for world, colour in (('separated', 'C0'), ('confounded', 'C3')):
        block = syn[syn.world_id == world].sort_values('n')
        axes[0].plot(block.n, block.cls_coverage, color=colour, marker='o', ms=3, label=f'REACH, {world}')
        axes[0].plot(block.n, block.EMQ_cls_coverage, color=colour, ls=':', marker='s', ms=3,
                     label=f'EMQ+bootstrap, {world}')
    axes[0].set(title='synthetic: $P(Z{=}0\\mid$cell, known$)$')
    real = pd.DataFrame(json.loads((Path(args.audit)/'summary.json').read_text())['n_axis_coverage'])
    real = real[(real.arm == 'label_shift') & (real.region == 'on_support')].sort_values('n')
    for col, style, label in (('joint_coverage', '-', 'REACH joint'), ('cls_coverage', '--', 'REACH $W_{cls}$ (plug-in)'),
                              ('EMQ_cls_coverage', ':', 'EMQ+bootstrap')):
        axes[1].plot(real.n, real[col], ls=style, marker='o', ms=3, label=label)
    axes[1].set(title='CIFAR-10 label shift (CLIP)')
    for ax in axes[:2]:
        ax.axhline(.95, color='k', lw=.8, ls='--')
        ax.set(xscale='log', xlabel='unlabelled log size $n$', ylabel='probe coverage', ylim=(-.03, 1.03))
        ax.legend(fontsize=6, loc='lower left')
    if args.corruption:
        sev = pd.DataFrame(json.loads((Path(args.corruption)/'summary.json').read_text())['severity'])
        sev = sev[sev.region == 'on_support'].sort_values('severity')
        axes[2].plot(sev.severity, sev.incompatible_rate, marker='o', label='incompatibility flagged')
        axes[2].plot(sev.severity, sev.joint_coverage, marker='s', label='REACH joint coverage')
        axes[2].plot(sev.severity, sev.EMQ_cls_coverage, marker='^', ls=':', label='EMQ+bootstrap coverage')
        axes[2].set(title='CIFAR-10-C (15 corruptions)', xlabel='severity', ylabel='rate', ylim=(-.03, 1.03))
        axes[2].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out/'fig3_coverage.pdf'); fig.savefig(out/'fig3_coverage.png', dpi=160)
    print('saved', out/'fig3_coverage.pdf')


if __name__ == '__main__':
    main()
