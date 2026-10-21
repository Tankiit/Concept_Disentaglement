"""Reference acquisition with fixed deployment, identities, and target labels."""
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
PARENT = HERE.parent / 'camouflage_identities_2026-10-21'
BASE = HERE.parent / 'camouflage_2026-10-21'
spec = importlib.util.spec_from_file_location('identities', PARENT / 'run.py')
ident = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ident)
SCALES = [1, 4, 16]
ARMS = ['known_only', 'candidate_only', 'both']
ALPHA = .01


def main():
    prior = json.loads((BASE / 'config.json').read_text())
    draws = json.loads((BASE / 'draws.json').read_text())
    identities = json.loads((PARENT / 'identity_draws.json').read_text())
    weights, theta = np.array(prior['weights']), np.array(prior['theta'])
    records, saved = [], []
    for seed in prior['seeds']:
        rng = np.random.default_rng(np.random.SeedSequence([20261025, seed]))
        rng.random((10, 1000)); rng.random(6500); rng.random(16000)
        labels = np.logical_xor(rng.random((10, 512)) < theta[:, None], rng.random((10, 512)) < .2)
        h = np.sqrt(np.log(20 / ALPHA) / (2 * 512))
        tl = np.clip((labels.mean(1) - h - .2) / .6, 0, 1)
        th = np.clip((labels.mean(1) + h - .2) / .6, 0, 1)
        counts = next(x['counts'] for x in identities if x['seed'] == seed and x['a'] == 8192)
        wl, wh = ident.bounds(np.array(counts), ALPHA)
        # Append independent observations to the exact previous reference counts.
        # Shared uniforms pair camouflage levels; prefixes nest acquisition budgets.
        extra = np.random.default_rng(np.random.SeedSequence([20261027, seed]))
        ku, qu = extra.random((10, 15000)), extra.random(97500)
        for lam in [.99, 1.]:
            A = np.array(prior['matrices'][str(lam)])
            old = next(x for x in draws if x['seed'] == seed and x['camouflage'] == lam and x['n'] == 16000)
            pl, ph = ident.bounds(np.array(old['log_counts']), ALPHA)
            settings = [(arm, scale) for arm in ARMS for scale in SCALES] + [('exact', 0)]
            for arm, scale in settings:
                nk = 1000 * (scale if arm in ['known_only', 'both'] else 1)
                nq = 6500 * (scale if arm in ['candidate_only', 'both'] else 1)
                kc, qc = np.array(old['known_counts']), np.array(old['candidate_counts'])
                for s in range(10):
                    kc[s] += np.bincount(np.searchsorted(np.cumsum(A[s]), ku[s, :nk-1000]), minlength=11)
                qc += np.bincount(np.searchsorted(np.cumsum(A[-1]), qu[:nq-6500]), minlength=11)
                if arm == 'exact':
                    lo, hi = A, A
                else:
                    kl, kh = ident.bounds(kc, ALPHA)
                    ql, qh = ident.bounds(qc, ALPHA)
                    lo, hi = np.vstack([kl, ql]), np.vstack([kh, qh])
                    saved.append(dict(seed=seed, camouflage=lam, arm=arm, scale=scale,
                                      known_counts=kc.tolist(), candidate_counts=qc.tolist()))
                for cell in range(11):
                    source = cell if cell < 10 else int(A[:-1, cell].argmax())
                    jt = weights * A[:, cell]
                    truths = dict(W_cls=jt[source]/jt[:-1].sum(), W_unk=jt[-1]/jt.sum(),
                                  W_label=theta@A[:-1, cell]/A[:-1, cell].sum())
                    results = dict(W_cls=ident.interval(lo, hi, pl, ph, wl, wh, cell, source),
                                   W_unk=ident.interval(lo, hi, pl, ph, wl, wh, cell, None),
                                   W_label=ident.label_interval(lo[:-1], hi[:-1], tl, th, cell))
                    for target, (lower, upper, status) in results.items():
                        records.append(dict(seed=seed, camouflage=lam, arm=arm, scale=scale,
                            known_refs=nk if arm != 'exact' else -1, candidate_refs=nq if arm != 'exact' else -1,
                            cell=cell, region='known_cells' if cell < 10 else 'outlier', target=target,
                            lower=lower, upper=upper, width=upper-lower, status=status,
                            truth=truths[target], covers=bool(lower-1e-6 <= truths[target] <= upper+1e-6)))
        pd.DataFrame(records).to_csv(HERE / 'intervals.csv', index=False)
        print('Completed seed', seed, flush=True)
    df = pd.DataFrame(records)
    keys = ['camouflage', 'arm', 'scale', 'target', 'region']
    ps = df.groupby(keys+['seed']).agg(width=('width', 'mean'), all_cover=('covers', 'all'))
    ps.to_csv(HERE / 'per_seed_summary.csv')
    summary = ps.groupby(level=keys).agg(width=('width', 'mean'), min_seed=('width', 'min'),
        max_seed=('width', 'max'), simultaneous_coverage=('all_cover', 'mean')).reset_index()
    summary.to_csv(HERE / 'summary.csv', index=False)
    (HERE / 'reference_draws.json').write_text(json.dumps(saved))
    config = dict(seeds=prior['seeds'], scales=SCALES, arms=ARMS, camouflage=[.99, 1.],
        identities=8192, deployment_inputs=16000, target_labels_per_digit=512,
        baseline_known_refs_per_digit=1000, baseline_candidate_refs=6500, alpha_per_family=ALPHA,
        extra_reference_seed_prefix=20261027,
        input_hashes={str(p.relative_to(HERE.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [BASE/'config.json', BASE/'draws.json', PARENT/'identity_draws.json']})
    (HERE / 'config.json').write_text(json.dumps(config, indent=2))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout='constrained', sharey=True)
    for i, lam in enumerate([.99, 1.]):
        for j, target in enumerate(['W_cls', 'W_label', 'W_unk']):
            ax = axes[i, j]
            group = summary[(summary.camouflage == lam) & (summary.target == target) & (summary.region == 'known_cells')]
            for arm in ARMS:
                sub = group[group.arm == arm].sort_values('scale')
                ax.plot(SCALES, sub.width, 'o-', label=arm.replace('_', ' '))
                ax.fill_between(SCALES, sub.min_seed, sub.max_seed, alpha=.12)
            ax.axhline(group[group.arm == 'exact'].width.iloc[0], color='black', linestyle=':', label='Exact source laws')
            ax.set(title=f'Camouflage {lam}: {target}', xlabel='Reference budget multiplier',
                   ylabel='Mean width over known-class cells', xscale='log', xticks=SCALES,
                   xticklabels=['1×', '4×', '16×'], ylim=(-.02, .55))
            ax.grid(alpha=.2)
    axes[0, 2].legend(fontsize=8)
    fig.suptitle('Reference acquisition with 8,192 identities fixed\n10 paired seeds; bands = seed min–max')
    fig.savefig(HERE / 'reference_response.png', dpi=180)
    fig.savefig(HERE / 'reference_response.pdf')
    print(summary[(summary.camouflage == 1) & (summary.region == 'known_cells')].to_string(index=False))


if __name__ == '__main__':
    main()
