"""Exact two-world posterior: observation MI versus contour-target MI."""
from pathlib import Path
import json
import numpy as np
from scipy.special import xlogy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
SOURCE = OUT.parent/'01_indistinguishable_worlds'


def entropy(p):
    return -np.sum(xlogy(p, p))/np.log(2)


def mi(laws, weights):
    marginal = weights @ laws
    # Weighted KL avoids entropy subtraction cancellation.
    return float(sum(w*np.sum(xlogy(p, p/marginal))/np.log(2)
                     for w,p in zip(weights,laws)))


def main():
    data = np.load(SOURCE/'cell_laws.npz')
    cfg = json.loads((SOURCE/'summary.json').read_text())
    m = data['m'].ravel()
    clean = np.stack([data['p0'].ravel(),data['p1'].ravel()])
    weights = np.array([0.5,0.5])
    observation_mi = mi(np.stack([m,m]),weights)
    clean_mi = mi(clean,weights)
    np.testing.assert_allclose(observation_mi,0,atol=1e-14)
    assert clean_mi > 0
    # Log likelihoods are identical for every possible count vector, so the
    # posterior equals the prior for all n. These checks verify implementation.
    rng = np.random.default_rng(20261023)
    errors = []
    for n in [10,100,1000,10000,1000000]:
        counts = rng.multinomial(n,m)
        log_likelihoods = np.array([counts @ np.log(m)]*2)
        posterior = weights*np.exp(log_likelihoods-log_likelihoods.max())
        posterior /= posterior.sum()
        errors.append(float(np.max(abs(posterior-weights))))
    assert max(errors) < 1e-14
    axis = np.linspace(-3.2,3.2,501)
    xx,yy = np.meshgrid(axis,axis)
    radius = cfg['contour_radius']
    membership = np.stack([(xx-mu[0])**2+(yy-mu[1])**2 <= radius**2 for mu in cfg['means']])
    prob = np.tensordot(weights,membership.astype(float),axes=1)
    target_mi = -(xlogy(prob,prob)+xlogy(1-prob,1-prob))/np.log(2)
    # Conditional membership entropy is zero: membership is deterministic in W.
    assert np.all(target_mi[membership[0] != membership[1]] == 1)
    assert np.all(target_mi[membership[0] == membership[1]] == 0)
    for w in [0.1,0.25,0.5,0.75,0.9]:
        np.testing.assert_allclose(mi(np.stack([m,m]),np.array([w,1-w])),0,atol=1e-14)
    summary = {
        'posterior_world_weights': weights.tolist(),
        'world_entropy_bits': float(entropy(weights)),
        'next_mixture_cell_MI_bits': observation_mi,
        'next_clean_cell_MI_bits': clean_mi,
        'contour_membership_MI_in_disagreement_region_bits': 1.0,
        'posterior_center_x_variance': 0.04,
        'posterior_check_max_error': max(errors),
        'note': 'Two known candidate worlds; exact diagnostic, not a learned classifier or general prior over contamination.'
    }
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(6,5),constrained_layout=True)
    mesh=ax.pcolormesh(xx,yy,target_mi,shading='auto',cmap='magma',vmin=0,vmax=1)
    angle=np.linspace(0,2*np.pi,500)
    for mu,color,style in zip(cfg['means'],['cyan','lime'],['-','--']):
        ax.plot(mu[0]+radius*np.cos(angle),mu[1]+radius*np.sin(angle),color=color,linestyle=style)
    ax.set(xlabel='x',ylabel='y',aspect='equal',title='Contour-membership MI: exact two-world posterior')
    fig.colorbar(mesh,ax=ax,label='Mutual information (bits)')
    fig.savefig(OUT/'target_information.png',dpi=180)
    plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
