"""probly quantification adapter; Gaussian bootstrap, not a deep-ensemble replication."""
from pathlib import Path
import importlib.util
import importlib.metadata
import json
import csv
import hashlib
import platform
import numpy as np
from scipy.special import xlogy
from probly.representation.distribution import create_categorical_distribution
from probly.representation.sample import ArraySample
from probly.quantification import SecondOrderEntropyDecomposition, sample_variance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('full_cell', OUT.parent/'06_full_cell_frequentist/run.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
SEED, REPS, MEMBERS = 20261028, 30, 199


def quantify(probabilities):
    assert np.all(probabilities >= 0)
    np.testing.assert_allclose(probabilities.sum(axis=-1), 1, atol=1e-12)
    sample = ArraySample(create_categorical_distribution(probabilities), sample_axis=0)
    dec = SecondOrderEntropyDecomposition(sample)
    return float(dec.epistemic), float(dec.total), float(dec.aleatoric)


def theta(t):
    result = t.copy()
    result[..., 2] = np.exp(result[..., 2])
    return result


def hausdorff(a, b):
    return np.linalg.norm(a[..., :2]-b[..., :2], axis=-1) + previous.C*abs(a[..., 2]-b[..., 2])


def checks():
    p = np.array([[.9, .1], [.1, .9]])
    expected = -np.sum(xlogy(p.mean(0), p.mean(0))) + np.mean(np.sum(xlogy(p, p), axis=-1))
    score, total, aleatoric = quantify(p)
    np.testing.assert_allclose(score, expected, atol=1e-14)
    np.testing.assert_allclose(total, score+aleatoric, atol=1e-14)
    np.testing.assert_allclose(quantify(np.tile(p[0], (5, 1)))[0], 0, atol=1e-14)
    np.testing.assert_allclose(quantify(p[::-1, ::-1])[0], score, atol=1e-14)
    np.testing.assert_allclose(sample_variance(ArraySample(p, sample_axis=0)), [.16, .16])
    return {'known_binary_mi_nats': score, 'formula_and_axis_checks': 'passed'}


def main():
    validation = checks()
    rng = np.random.default_rng(SEED)
    left, right = np.array([-.2, 0., 1.]), np.array([.2, 0., 1.])
    mixture, _ = previous.h.mixture(.2)
    rows, raw = [], {}
    max_init_gap = 0.
    for name, law, truths in [('clean', previous.laws(left[None])[0], [left]),
                              ('shared_mixture', mixture, [left, right])]:
        for n in [1000, 10000, 100000]:
            counts = rng.multinomial(n, law, size=REPS)
            estimates, _, _ = previous.fit(counts)
            # Same-data random-start fits assess optimization variability only.
            starts = rng.uniform([-.4, -.4, -.25], [.4, .4, .25], size=(5, 3))
            init = np.stack([previous.fit(counts, start=s)[0] for s in starts])
            max_init_gap = max(max_init_gap, float(np.max(abs(init-estimates))))
            values = []
            for rep, (count, estimate) in enumerate(zip(counts, estimates)):
                boot_counts = rng.multinomial(n, count/n, size=MEMBERS)
                boot, _, _ = previous.fit(boot_counts, start=estimate)
                members = theta(boot)
                probabilities = previous.laws(members)
                mi, total, aleatoric = quantify(probabilities)
                # Check every adapter result against the mathematical definition.
                direct = -np.sum(xlogy(probabilities.mean(0), probabilities.mean(0))) + np.mean(np.sum(xlogy(probabilities, probabilities), axis=-1))
                np.testing.assert_allclose(mi, direct, atol=2e-12)
                center = theta(estimate)
                radius = float(np.quantile(hausdorff(members, center), .95))
                support = members[:, :2] @ np.array([1., 0.]) + previous.C*members[:, 2]
                support_var = float(sample_variance(ArraySample(support, sample_axis=0)))
                init_mi = quantify(previous.laws(theta(init[:, rep])))[0]
                row = [mi, total, aleatoric, radius, support_var, init_mi]
                row += [float(hausdorff(center, truth)) for truth in truths]
                row += [quantify(probabilities[:k])[0] for k in [5, 20, 50]]
                values.append(row)
            v = np.array(values)
            raw[f'{name}_{n}'] = v
            for i, truth in enumerate(truths):
                error = v[:, 6+i]
                rows.append(dict(scenario=name, n=n, truth=i, repetitions=REPS,
                    mi_nats=float(v[:, 0].mean()), mi_mcse=float(v[:, 0].std(ddof=1)/np.sqrt(REPS)),
                    contour_error=float(error.mean()), bootstrap_radius=float(v[:, 3].mean()),
                    bootstrap_coverage=float(np.mean(error <= v[:, 3])),
                    support_variance=float(v[:, 4].mean()), initialization_mi=float(v[:, 5].mean()),
                    mi_5=float(v[:, -3].mean()), mi_20=float(v[:, -2].mean()), mi_50=float(v[:, -1].mean())))
            print(name, n, 'MI', v[:, 0].mean(), 'radius', v[:, 3].mean(), flush=True)
    validation['max_random_start_parameter_gap'] = max_init_gap
    assert max_init_gap < 1e-5
    validation.update(seed=SEED, repetitions=REPS, members=MEMBERS,
        python=platform.python_version(), probly=importlib.metadata.version('probly'),
        raw_columns='mi,total,aleatoric,radius,support_variance,init_mi,error_truth(s),mi_5,mi_20,mi_50')
    import probly
    root = Path(probly.__file__).parent
    paths = ['transformation/ensemble/_common.py', 'quantification/decomposition/entropy/_common.py',
             'representation/sample/array.py', 'representation/distribution/array_categorical.py',
             'quantification/measure/sample/_common.py']
    validation['library_source_sha256'] = {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}
    (OUT/'validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    with (OUT/'summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(OUT/'raw.npz', **raw)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for name in ['clean', 'shared_mixture']:
        selected = [r for r in rows if r['scenario']==name and r['truth']==0]
        ns = [r['n'] for r in selected]
        axes[0].loglog(ns, [r['mi_nats'] for r in selected], 'o-', label=name)
        axes[1].loglog(ns, [r['contour_error'] for r in selected], 'o-', label=name+' error')
        axes[1].loglog(ns, [r['bootstrap_radius'] for r in selected], 'x--', label=name+' radius')
    axes[0].set_ylabel('Cell predictive MI (nats)'); axes[1].set_ylabel('Contour distance')
    for ax in axes: ax.set_xlabel('Sample size'); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT/'adapter_comparison.png', dpi=160); plt.close(fig)


if __name__ == '__main__':
    main()
