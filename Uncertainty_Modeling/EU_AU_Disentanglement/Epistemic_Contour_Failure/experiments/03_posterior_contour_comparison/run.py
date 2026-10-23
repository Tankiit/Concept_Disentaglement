"""Finite-cell Gaussian posterior, clean controls, and two-world aware control."""
from pathlib import Path
import json
import numpy as np
from scipy.special import ndtr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
REPS = 400
SEED = 20261024
NS = [100, 1000, 10000, 100000]


def model(edges, step):
    grid = np.linspace(-2,2,round(4/step)+1)
    z = edges[None,:]-grid[:,None]
    # Survival differences in the positive tail avoid CDF cancellation.
    p = np.where(z[:,:-1]>=0,
                 ndtr(-z[:,:-1])-ndtr(-z[:,1:]),
                 ndtr(z[:,1:])-ndtr(z[:,:-1]))
    assert np.all(p>0)
    np.testing.assert_allclose(p.sum(axis=1),1,atol=1e-14)
    return grid, np.log(p)


def fit(counts,grid,logp):
    scores = counts @ logp.T
    posterior = np.exp(scores-scores.max(axis=1,keepdims=True))
    posterior /= posterior.sum(axis=1,keepdims=True)
    cdf=np.cumsum(posterior,axis=1)
    lo=grid[(cdf>=.025).argmax(axis=1)]
    hi=grid[(cdf>=.975).argmax(axis=1)]
    means=posterior @ grid
    assert np.all(posterior[:,[0,-1]].sum(axis=1)<1e-8)
    return lo,hi,means


def wilson(hits,n):
    p=hits/n; z=1.95996398454
    mid=(p+z*z/(2*n))/(1+z*z/n)
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [float(mid-half),float(mid+half)]


def main():
    saved=np.load(OUT.parent/'01_indistinguishable_worlds/cell_laws.npz')
    grid,logp=model(saved['edges'],.001)
    fine,fine_logp=model(saved['edges'],.0005)
    m=saved['m'].sum(axis=0)
    # Population KL projection checked on the declared parameter grid.
    population_scores=logp @ m
    assert abs(grid[np.argmax(population_scores)])<1e-12
    rng=np.random.default_rng(SEED)
    rows=[]; raw={}; max_refinement=0
    for setting,law in [('mixture',m),('clean_0',saved['p0'].sum(axis=0)),('clean_1',saved['p1'].sum(axis=0))]:
        for n in NS:
            counts=rng.multinomial(n,law,size=REPS)
            lo,hi,mean=fit(counts,grid,logp)
            fl,fh,fm=fit(counts[:20],fine,fine_logp)
            max_refinement=max(max_refinement,float(max(np.max(abs(fl-lo[:20])),np.max(abs(fh-hi[:20])))))
            raw[f'{setting}_{n}']=np.column_stack([lo,hi,mean])
            truths=[-.2,.2] if setting=='mixture' else ([-.2] if setting=='clean_0' else [.2])
            for truth in truths:
                hits=int(np.sum((lo<=truth+1e-12)&(hi>=truth-1e-12)))
                rows.append(dict(setting=setting,n=n,true_center=truth,coverage=hits/REPS,
                                 coverage_wilson95=wilson(hits,REPS),
                                 mean_contour_set_diameter=float(np.mean(hi-lo)),
                                 mean_hausdorff_error=float(np.mean(abs(mean-truth))),
                                 mean_estimated_center=float(mean.mean())))
    assert max_refinement<=.00101
    summary=dict(seed=SEED,replicates_per_setting=REPS,grid_step=.001,
                 population_grid_projection_center=float(grid[np.argmax(population_scores)]),
                 maximum_endpoint_change_under_grid_refinement=max_refinement,
                 aware_control=dict(coverage_each_world=1.0,contour_set_diameter=.4,posterior_weights=[.5,.5]),rows=rows)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    np.savez_compressed(OUT/'replicates.npz',**raw)
    fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for setting,truth,label in [('mixture',-.2,'Ignoring contamination, world 0'),('mixture',.2,'Ignoring contamination, world 1'),('clean_0',-.2,'Clean control, world 0'),('clean_1',.2,'Clean control, world 1')]:
        sub=[r for r in rows if r['setting']==setting and r['true_center']==truth]
        axs[0].plot(NS,[r['coverage'] for r in sub],'o-',label=label)
    axs[0].axhline(.95,color='gray',linestyle=':',label='Nominal 95%')
    axs[0].axhline(1,color='black',linestyle='--',label='Two-world aware control')
    axs[0].set(xscale='log',xlabel='Sample size',ylabel='Whole-contour set coverage',ylim=(-.03,1.05))
    axs[0].legend(fontsize=6)
    sub=[r for r in rows if r['setting']=='mixture' and r['true_center']==-.2]
    axs[1].loglog(NS,[r['mean_contour_set_diameter'] for r in sub],'o-',label='Ignoring contamination')
    axs[1].axhline(.4,color='black',linestyle='--',label='Two-world aware control')
    axs[1].set(xlabel='Sample size',ylabel='Mean Hausdorff diameter of contour set')
    axs[1].legend(fontsize=8)
    fig.savefig(OUT/'coverage_and_width.png',dpi=180)
    plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
