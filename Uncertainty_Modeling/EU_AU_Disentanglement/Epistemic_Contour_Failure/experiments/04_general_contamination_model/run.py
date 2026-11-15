"""General finite-grid contamination model with a transparent proper prior."""
from pathlib import Path
import json
import numpy as np
from scipy.special import ndtr
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
SEED=20261025
SAMPLES=2048
RADIUS=np.sqrt(-2*np.log(.05))
EDGES=np.array([-np.inf,-2,-1,0,1,2,np.inf])
EPSILONS=[.15,.20,.30]
NS=[1000,10000,100000,1000000]


def laws(theta,edges=EDGES):
    z=(edges[None,None,:]-theta[:,:2,None])/theta[:,2,None,None]
    p=np.where(z[:,:,:-1]>=0,ndtr(-z[:,:,:-1])-ndtr(-z[:,:,1:]),ndtr(z[:,:,1:])-ndtr(z[:,:,:-1]))
    ans=(p[:,1,:,None]*p[:,0,None,:]).reshape(len(theta),-1)
    np.testing.assert_allclose(ans.sum(axis=1),1,atol=1e-14)
    assert np.all(ans>0)
    return ans


def parameters(step=.1,scale_step=.05):
    x=np.linspace(-.5,.5,round(1/step)+1)
    s=np.linspace(.65,1.35,round(.7/scale_step)+1)
    return np.array(np.meshgrid(x,x,s,indexing='ij')).reshape(3,-1).T


def mixture(eps,edges=EDGES):
    p= laws(np.array([[-.2,0,1],[.2,0,1]]),edges)
    tv=2*ndtr(.2)-1
    remainder=1-(1-eps)*(1+tv)
    assert remainder>0
    m=(1-eps)*p.max(axis=0)+remainder*p.mean(axis=0)
    q=(m[None,:]-(1-eps)*p)/eps
    assert q.min()>-1e-14
    np.testing.assert_allclose(q.sum(axis=1),1,atol=1e-13)
    return m,q


def compatible(ms,clean,eps):
    # No absolute slack: tiny positive cell probabilities must remain meaningful.
    keep=np.ones((len(ms),len(clean)),dtype=bool)
    for j in range(clean.shape[1]):
        keep &= ms[:,j,None]>=(1-eps)*clean[None,:,j]*(1-1e-12)
    return keep


def diameter(theta):
    if len(theta)<2:return 0.
    largest=0.
    for start in range(0,len(theta),256):
        block=theta[start:start+256]
        d=cdist(block[:,:2],theta[:,:2])+RADIUS*np.abs(block[:,2,None]-theta[None,:,2])
        largest=max(largest,float(d.max()))
    return largest


def describe(theta):
    return dict(count=len(theta),minimum=theta.min(axis=0).tolist() if len(theta) else None,
                maximum=theta.max(axis=0).tolist() if len(theta) else None,
                hausdorff_diameter=diameter(theta))


def posterior_samples(counts,clean,eps,rng):
    collected=[];accepted=0;attempted=0;eligible=0
    while accepted<SAMPLES and attempted<32768:
        ms=rng.dirichlet(counts+1.,size=512)
        mask=compatible(ms,clean,eps)
        valid=mask.any(axis=1)
        eligible+=int(valid.sum());attempted+=len(ms)
        if valid.any():
            take=mask[valid][:SAMPLES-accepted]
            collected.append(take);accepted+=len(take)
    diagnostics=dict(accepted=accepted,proposals=attempted,eligible=eligible,acceptance=eligible/attempted)
    if accepted<SAMPLES:
        return None,diagnostics
    return np.concatenate(collected),diagnostics


def summarize_posterior(mask,theta,weights):
    conditional=mask*weights[None,:]
    conditional/=conditional.sum(axis=1,keepdims=True)
    mass=conditional.mean(axis=0)
    np.testing.assert_allclose(mass.sum(),1,atol=1e-13)
    order=np.argsort(-mass,kind='stable')
    cutoff=np.searchsorted(np.cumsum(mass[order]),.95)+1
    selected=order[:cutoff]
    mean=mass@theta
    se_mean=conditional@theta
    se_mean=se_mean.std(axis=0,ddof=1)/np.sqrt(len(mask))
    truth_indices=[np.argmin(np.linalg.norm(theta-t,axis=1)) for t in [[-.2,0,1],[.2,0,1]]]
    return dict(mean=mean.tolist(),mean_mcse=se_mean.tolist(),
                hpd_mass=float(mass[selected].sum()),hpd_count=len(selected),
                hpd_diameter=diameter(theta[selected]),
                true_contours_in_hpd=[bool(i in selected) for i in truth_indices],
                true_contour_probabilities=mass[truth_indices].tolist(),
                left_probability=float(mass[theta[:,0]<-1e-8].sum()),
                center_probability=float(mass[np.abs(theta[:,0])<1e-8].sum()),
                right_probability=float(mass[theta[:,0]>1e-8].sum())),mass


def main():
    theta=parameters();clean=laws(theta)
    refined=parameters(.05,.025);refined_clean=laws(refined)
    weights={'equal':np.ones(len(theta)),'left':np.exp(-8*theta[:,0]),'right':np.exp(8*theta[:,0])}
    rng=np.random.default_rng(SEED)
    results=[];populations=[];failed_sampling=[];arrays=dict(theta=theta,clean_laws=clean,edges=EDGES)
    for eps in EPSILONS:
        m,q=mixture(eps)
        mask=compatible(m[None,:],clean,eps)[0]
        fine_mask=compatible(m[None,:],refined_clean,eps)[0]
        assert all(np.any(np.all(np.isclose(theta[mask],t),axis=1)) for t in [[-.2,0,1],[.2,0,1]])
        # Known Q0 identifies the original clean law exactly on this grid.
        known=np.max(abs((1-eps)*clean+eps*q[0]-m),axis=1)<1e-12
        assert known.sum()==1
        tv=np.abs(clean-m).sum(axis=1)/2
        mle=np.argmax(np.log(clean)@m)
        tv_best=np.flatnonzero(np.isclose(tv,tv.min(),atol=1e-12,rtol=0))
        def errors(indices):
            return [[float(np.linalg.norm(theta[i,:2]-np.array(t))+RADIUS*abs(theta[i,2]-1)) for t in [[-.2,0],[.2,0]]] for i in indices]
        populations.append(dict(epsilon=eps,identified=describe(theta[mask]),refined_identified=describe(refined[fine_mask]),
                                known_q0=theta[known].tolist(),mle=theta[mle].tolist(),mle_errors=errors([mle])[0],
                                minimum_tv=float(tv.min()),minimum_tv_parameters=theta[tv_best].tolist(),minimum_tv_errors=errors(tv_best)))
        populations[-1]['conditional_at_exact_m'] = {name:summarize_posterior(np.tile(mask,(2,1)),theta,w)[0] for name,w in weights.items()}
        assert not np.any(np.isclose(theta[mask,2],.65)|np.isclose(theta[mask,2],1.35))
        arrays[f'population_mask_{eps}']=mask
        for n in NS:
            counts=rng.multinomial(n,m)
            samples,diagnostics=posterior_samples(counts,clean,eps,rng)
            arrays[f'counts_{eps}_{n}']=counts
            if samples is None:
                failed_sampling.append(dict(epsilon=eps,n=n,**diagnostics))
                print(f'Insufficient sampling eps={eps}, n={n}: {diagnostics}',flush=True)
                continue
            for prior,w in weights.items():
                row,mass=summarize_posterior(samples,theta,w)
                row.update(epsilon=eps,n=n,prior=prior,diagnostics=diagnostics)
                results.append(row);arrays[f'posterior_{eps}_{n}_{prior}']=mass
            print(f'eps={eps}, n={n}, rejection acceptance={diagnostics["acceptance"]:.3f}',flush=True)
    # Follow-up motivated by exclusion of both true contours at eps=.15, n=1e6.
    eps=.15;n=100000000
    extra_rng=np.random.default_rng(SEED+3)
    extra_m,_=mixture(eps)
    extra_counts=extra_rng.multinomial(n,extra_m)
    extra_samples,extra_diag=posterior_samples(extra_counts,clean,eps,extra_rng)
    assert extra_samples is not None
    arrays[f'counts_{eps}_{n}']=extra_counts
    for prior,w in weights.items():
        row,mass=summarize_posterior(extra_samples,theta,w)
        row.update(epsilon=eps,n=n,prior=prior,diagnostics=extra_diag)
        results.append(row);arrays[f'posterior_{eps}_{n}_{prior}']=mass
    # Fixed-data-law sensitivity to declared epsilon; these sets are nested.
    fixed_m,_=mixture(.2)
    declared=[];previous=np.zeros(len(theta),bool)
    for assumed in [.05,.1,.15,.2,.3]:
        mask=compatible(fixed_m[None,:],clean,assumed)[0]
        assert np.all(~previous|mask);previous=mask
        fine_mask=compatible(fixed_m[None,:],refined_clean,assumed)[0]
        declared.append(dict(assumed_epsilon=assumed,**describe(theta[mask]),refined=describe(refined[fine_mask])))
    # Finer observation partition: verifies which coarse compatible candidates
    # survive extra information (not continuous-data compatibility certification).
    fine_edges=np.r_[-np.inf,np.arange(-4,4.01,.5),np.inf]
    mf,_=mixture(.2,fine_edges)
    fine_observation_mask=compatible(mf[None,:],laws(theta,fine_edges),.2)[0]
    coarse=arrays['population_mask_0.2']
    assert np.all(~fine_observation_mask|coarse)
    # Independent Monte Carlo rerun of one anchor posterior.
    anchor_samples,anchor_diag=posterior_samples(arrays['counts_0.2_100000'],clean,.2,np.random.default_rng(SEED+1))
    anchor,anchor_mass=summarize_posterior(anchor_samples,theta,weights['equal'])
    base=next(r for r in results if r['epsilon']==.2 and r['n']==100000 and r['prior']=='equal')
    dif=np.array(anchor['mean'])-base['mean'];se=np.hypot(anchor['mean_mcse'],base['mean_mcse'])
    assert np.all(np.abs(dif)<6*se+1e-12)
    # Parameter-grid sensitivity for one high-sample posterior; same data,
    # equal conditional weights, same Dirichlet marginal base prior.
    refined_samples,refined_diag=posterior_samples(arrays['counts_0.2_1000000'],refined_clean,.2,np.random.default_rng(SEED+2))
    assert refined_samples is not None
    refined_post,_=summarize_posterior(refined_samples,refined,np.ones(len(refined)))
    summary=dict(refined_anchor_posterior=dict(posterior=refined_post,diagnostics=refined_diag),seed=SEED,posterior_draws=SAMPLES,parameter_candidates=len(theta),refined_candidates=len(refined),
                 populations=populations,posteriors=results,failed_sampling=failed_sampling,declared_epsilon_sensitivity=declared,
                 finer_partition_identified=describe(theta[fine_observation_mask]),
                 independent_anchor_check=dict(mean_difference=dif.tolist(),combined_mcse=se.tolist(),diagnostics=anchor_diag))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    np.savez_compressed(OUT/'arrays.npz',**arrays)
    fig,axs=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for eps in EPSILONS:
        sub=[r for r in results if r['epsilon']==eps and r['prior']=='equal']
        axs[0].plot([r['n'] for r in sub],[r['hpd_diameter'] for r in sub],'o-',label=f'epsilon={eps}')
    axs[0].set(xscale='log',xlabel='Sample size',ylabel='95% posterior contour-set diameter');axs[0].legend()
    for prior in weights:
        sub=[r for r in results if r['epsilon']==.2 and r['prior']==prior]
        axs[1].plot([r['n'] for r in sub],[r['mean'][0] for r in sub],'o-',label=prior)
    axs[1].set(xscale='log',xlabel='Sample size',ylabel='Posterior mean horizontal center');axs[1].legend(title='Conditional prior')
    subset=theta[coarse]
    pairs=np.unique(subset[:,[0,2]],axis=0)
    max_vertical=[np.max(np.abs(subset[np.all(np.isclose(subset[:,[0,2]],pair),axis=1),1])) for pair in pairs]
    scatter=axs[2].scatter(pairs[:,0],pairs[:,1],c=max_vertical,cmap='viridis',s=50)
    axs[2].set(xlabel='Horizontal center',ylabel='Scale',title='Compatible parameters at epsilon=0.20')
    fig.colorbar(scatter,ax=axs[2],label='Maximum compatible absolute vertical center')
    fig.savefig(OUT/'general_model.png',dpi=180);plt.close(fig)
    print(json.dumps(dict(populations=populations,declared=declared,independent_check=summary['independent_anchor_check']),indent=2))


if __name__=='__main__':main()
