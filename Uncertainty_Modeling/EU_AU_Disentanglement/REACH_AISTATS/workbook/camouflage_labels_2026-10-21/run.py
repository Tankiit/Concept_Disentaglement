"""Target-label acquisition with all source/deployment evidence held fixed."""
from pathlib import Path
import importlib.util
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent/'camouflage_references_2026-10-21'
BASE = HERE.parent/'camouflage_2026-10-21'
spec = importlib.util.spec_from_file_location('references', PARENT/'run.py')
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)
BUDGETS = [512, 2048, 8192, 32768]
MODES = ['baseline_references', '16x_references', 'exact_sources']


def main():
    prior = json.loads((BASE/'config.json').read_text())
    theta = np.array(prior['theta'])
    draws = json.loads((PARENT/'reference_draws.json').read_text())
    old = pd.read_csv(PARENT/'intervals.csv')
    records, label_draws = [], []
    for seed in prior['seeds']:
        rng = np.random.default_rng(np.random.SeedSequence([20261025, seed]))
        rng.random((10, 1000)); rng.random(6500); rng.random(16000)
        original = np.logical_xor(rng.random((10, 512)) < theta[:, None], rng.random((10, 512)) < .2)
        extra = np.random.default_rng(np.random.SeedSequence([20261028, seed]))
        added = np.logical_xor(extra.random((10, 32768-512)) < theta[:, None], extra.random((10, 32768-512)) < .2)
        labels = np.concatenate([original, added], axis=1)
        for budget in BUDGETS:
            label_draws.append(dict(seed=seed, budget=budget, positives=labels[:, :budget].sum(1).tolist()))
        for lam in [.99, 1.]:
            A = np.array(prior['matrices'][str(lam)])
            for mode in MODES:
                scale = 1 if mode == 'baseline_references' else 16
                arm = 'exact' if mode == 'exact_sources' else 'both'
                parent = old[(old.seed == seed) & (old.camouflage == lam) & (old.arm == arm)
                             & (old.scale == (0 if arm == 'exact' else scale))].set_index(['cell', 'target'])
                if mode == 'exact_sources':
                    lo, hi = A[:-1], A[:-1]
                else:
                    draw = next(x for x in draws if x['seed'] == seed and x['camouflage'] == lam and x['arm'] == 'both' and x['scale'] == scale)
                    lo, hi = ref.ident.bounds(np.array(draw['known_counts']), .01)
                for budget in BUDGETS + [-1]:
                    if budget == -1:
                        tl, th = theta, theta
                    else:
                        h = np.sqrt(np.log(20/.01)/(2*budget))
                        tl = np.clip((labels[:, :budget].mean(1)-h-.2)/.6, 0, 1)
                        th = np.clip((labels[:, :budget].mean(1)+h-.2)/.6, 0, 1)
                    for cell in range(11):
                        # These two targets do not use target-label evidence. Reuse
                        # their validated endpoints rather than rerun identical LPs.
                        results = {t: tuple(parent.loc[(cell, t), ['lower', 'upper', 'status']]) for t in ['W_cls', 'W_unk']}
                        results['W_label'] = ref.ident.label_interval(lo, hi, tl, th, cell)
                        for target, (lower, upper, status) in results.items():
                            truth = float(parent.loc[(cell, target), 'truth'])
                            records.append(dict(seed=seed, camouflage=lam, mode=mode, budget=budget,
                                cell=cell, region='known_cells' if cell < 10 else 'outlier', target=target,
                                lower=lower, upper=upper, width=upper-lower, status=status, truth=truth,
                                covers=bool(lower-1e-6 <= truth <= upper+1e-6)))
        print('Completed seed', seed, flush=True)
    df = pd.DataFrame(records)
    df.to_csv(HERE/'intervals.csv', index=False)
    keys = ['camouflage', 'mode', 'budget', 'target', 'region']
    ps = df.groupby(keys+['seed']).agg(width=('width', 'mean'), all_cover=('covers', 'all'))
    ps.to_csv(HERE/'per_seed_summary.csv')
    summary = ps.groupby(level=keys).agg(width=('width', 'mean'), min_seed=('width', 'min'),
        max_seed=('width', 'max'), simultaneous_coverage=('all_cover', 'mean')).reset_index()
    summary.to_csv(HERE/'summary.csv', index=False)
    (HERE/'label_draws.json').write_text(json.dumps(label_draws))
    (HERE/'config.json').write_text(json.dumps(dict(seeds=prior['seeds'], budgets=BUDGETS,
        exact_theta_budget_code=-1, modes=MODES, camouflage=[.99, 1.], identities=8192,
        deployment_inputs=16000, flip_probability=.2, alpha_per_family=.01, seed_prefix=20261028,
        scope='Synthetic source-specific binary target labels; no new model training. W_cls and W_unk endpoints reused from parent.',
        hashes={str(p.relative_to(HERE.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [BASE/'config.json', PARENT/'reference_draws.json', PARENT/'intervals.csv']}), indent=2))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout='constrained', sharey=True)
    for i, lam in enumerate([.99, 1.]):
        for j, target in enumerate(['W_cls', 'W_label', 'W_unk']):
            ax = axes[i, j]
            group = summary[(summary.camouflage == lam) & (summary.target == target) & (summary.region == 'known_cells')]
            for mode in MODES:
                sub = group[(group['mode'] == mode) & (group.budget > 0)].sort_values('budget')
                line, = ax.plot(BUDGETS, sub.width, 'o-', label=mode.replace('_', ' '))
                ax.fill_between(BUDGETS, sub.min_seed, sub.max_seed, color=line.get_color(), alpha=.12)
                if target == 'W_label':
                    benchmark = group[(group['mode'] == mode) & (group.budget == -1)].width.iloc[0]
                    ax.axhline(benchmark, color=line.get_color(), linestyle=':', alpha=.8)
            ax.set(title=f'Camouflage {lam}: {target}', xlabel='Target labels per digit', ylabel='Mean width over known-class cells',
                   xscale='log', xticks=BUDGETS, xticklabels=[str(b) for b in BUDGETS], ylim=(-.02, .4))
            ax.grid(alpha=.2)
    axes[0, 2].legend(fontsize=8)
    fig.suptitle('Target-label acquisition; all other evidence fixed\nBands = seed min–max; dotted label lines = exact target-rate controls')
    fig.savefig(HERE/'label_response.png', dpi=180)
    fig.savefig(HERE/'label_response.pdf')
    print(summary[(summary.camouflage == 1) & (summary.region == 'known_cells') & (summary.target == 'W_label')].to_string(index=False))


if __name__ == '__main__':
    main()
