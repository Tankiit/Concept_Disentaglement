"""Exact Gaussian contamination construction; run with Python + numpy/scipy/matplotlib."""
from pathlib import Path
import json
import numpy as np
from scipy.special import ndtr, ndtri
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

OUT = Path(__file__).resolve().parent
EPSILON = 0.20
SEPARATION = 0.40
ALPHA = 0.95
SIGMA = 1.0


def normal_bins(edges, mean):
    return np.diff(ndtr((edges - mean) / SIGMA))


def construct(step):
    edges = np.r_[-np.inf, np.linspace(-6, 6, round(12 / step) + 1), np.inf]
    x0 = normal_bins(edges, -SEPARATION / 2)
    x1 = normal_bins(edges, SEPARATION / 2)
    y = normal_bins(edges, 0)
    p0, p1 = np.outer(y, x0), np.outer(y, x1)
    tv = 2 * ndtr(SEPARATION / (2 * SIGMA)) - 1
    remainder = 1 - (1 - EPSILON) * (1 + tv)
    assert remainder >= 0
    # All grids have an edge at x=0, where the two densities cross.
    # Hence max of cell masses equals the integral of the maximum density.
    m = (1 - EPSILON) * np.maximum(p0, p1) + remainder * (p0 + p1) / 2
    q0 = (m - (1 - EPSILON) * p0) / EPSILON
    q1 = (m - (1 - EPSILON) * p1) / EPSILON
    for law in (p0, p1, m, q0, q1):
        assert law.min() >= -1e-14
        np.testing.assert_allclose(law.sum(), 1, atol=2e-14, rtol=0)
    reconstruction_error = max(
        np.max(np.abs(m - ((1 - EPSILON) * p0 + EPSILON * q0))),
        np.max(np.abs(m - ((1 - EPSILON) * p1 + EPSILON * q1))),
    )
    assert reconstruction_error < 1e-14
    cell_tv = np.abs(p0 - p1).sum() / 2
    np.testing.assert_allclose(cell_tv, tv, atol=2e-14, rtol=0)
    return edges, p0, p1, m, q0, q1, {
        'step': step, 'cells': int(m.size), 'cell_tv': float(cell_tv),
        'max_reconstruction_error': float(reconstruction_error),
        'max_normalization_error': float(max(abs(z.sum()-1) for z in (p0,p1,m,q0,q1))),
        'minimum_contamination_mass': float(min(q0.min(), q1.min())),
    }


def main():
    rows = [construct(step)[-1] for step in (1.0, 0.5, 0.25, 0.125)]
    edges, p0, p1, m, q0, q1, _ = construct(0.25)
    tv = 2 * ndtr(SEPARATION / (2 * SIGMA)) - 1
    summary = {
        'epsilon': EPSILON, 'sigma': SIGMA, 'alpha': ALPHA,
        'means': [[-SEPARATION/2, 0], [SEPARATION/2, 0]],
        'continuous_tv': float(tv), 'minimum_feasible_epsilon': float(tv/(1+tv)),
        'contour_radius': float(SIGMA*np.sqrt(-2*np.log(1-ALPHA))),
        'contour_hausdorff_distance': SEPARATION,
        'minimax_expected_hausdorff_error_lower_bound': SEPARATION/2,
        'maximum_feasible_center_separation_at_epsilon': float(2*SIGMA*ndtri(1/(2*(1-EPSILON)))),
        'resolution_checks': rows,
    }
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    np.savez_compressed(OUT/'cell_laws.npz', edges=edges, p0=p0, p1=p1, m=m, q0=q0, q1=q1)

    fig, axs = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    radius = summary['contour_radius']
    for ax, law, title in zip(axs, (p0, m, p1), ('World 0: clean source', 'Shared observed distribution', 'World 1: clean source')):
        mesh = ax.pcolormesh(edges[1:-1], edges[1:-1], law[1:-1,1:-1]/0.25**2,
                             shading='flat', cmap='Blues', vmin=0, vmax=1/(2*np.pi))
        for mean, color, style in ((-SEPARATION/2, 'darkorange', '-'), (SEPARATION/2, 'purple', '--')):
            ax.add_patch(Circle((mean, 0), radius, fill=False, color=color, linestyle=style, linewidth=2,
                                label=f'Clean contour, center x={mean:+.1f}'))
        ax.set(xlim=(-3.5,3.5), ylim=(-3.5,3.5), xlabel='x', ylabel='y', title=title, aspect='equal')
    axs[1].legend(loc='upper right', fontsize=7)
    fig.colorbar(mesh, ax=axs, label='Probability mass / cell area', shrink=0.75)
    fig.savefig(OUT/'indistinguishable_worlds.png', dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
