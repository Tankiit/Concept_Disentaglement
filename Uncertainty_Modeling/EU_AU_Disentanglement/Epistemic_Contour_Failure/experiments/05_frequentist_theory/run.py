"""Exact frequentist coverage plus independent checks of confidence projections."""
from pathlib import Path
import importlib.util,json
import numpy as np
from scipy.stats import beta,binom
from scipy.special import ndtr,ndtri,logsumexp
from scipy.optimize import linprog
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
BETA=.05
SEED=20261026


def cp(counts,n,error):
    counts=np.asarray(counts)
    lo=np.zeros(counts.shape,dtype=float);hi=np.ones(counts.shape,dtype=float)
    use=counts>0;lo[use]=beta.ppf(error/2,counts[use],n-counts[use]+1)
    use=counts<n;hi[use]=beta.ppf(1-error/2,counts[use]+1,n-counts[use])
    return lo,hi


def feasible(p,lo,hi,eps):
    a=np.maximum(lo,(1-eps)*p)
    return np.all(a<=hi+1e-14,axis=-1)&(a.sum(axis=-1)<=1+1e-14)&(hi.sum()>=1-1e-14)


def interval(lo,hi,eps):
    low=ndtri(np.clip((lo-eps)/(1-eps),0,1))
    high=ndtri(np.clip(hi/(1-eps),0,1))
    return np.maximum(low,-1),np.minimum(high,1)


def main():
    rows=[];arrays={}
    for n in [100,1000,10000,100000]:
        counts=np.arange(n+1)
        lo,hi=cp(counts,n,BETA)
        for setting,r,eps in [('clean_left',ndtr(-.2),0),('clean_right',ndtr(.2),0),('shared_mixture',.5,.2)]:
            pmf=binom.pmf(counts,n,r)
            np.testing.assert_allclose(pmf.sum(),1,atol=1e-10)
            truths=[-.2,.2] if setting=='shared_mixture' else ([-.2] if setting=='clean_left' else [.2])
            for method,declared in [('clean_model',0),('contamination_aware',eps)]:
                lower,upper=interval(lo,hi,declared)
                width=np.maximum(upper-lower,0)
                for truth in truths:
                    included=(lower<=truth)&(truth<=upper)
                    coverage=float(pmf@included)
                    log10_coverage=float(logsumexp(binom.logpmf(counts[included],n,r))/np.log(10))
                    rows.append(dict(n=n,setting=setting,method=method,true_center=truth,
                                     exact_coverage=coverage,log10_coverage=log10_coverage,expected_diameter=float(pmf@width)))
                    if method=='contamination_aware' or setting!='shared_mixture':
                        assert coverage>=1-BETA-1e-10
        arrays[f'cp_lower_{n}']=lo;arrays[f'cp_upper_{n}']=hi
    # Compare the closed-form simplex feasibility condition against a generic LP.
    rng=np.random.default_rng(SEED);checks=0
    for i in range(300):
        k=6;p=rng.dirichlet(np.ones(k));m=rng.dirichlet(np.ones(k));eps=rng.uniform(0,.4)
        lo=np.maximum(0,m-rng.uniform(0,.12,k));hi=np.minimum(1,m+rng.uniform(0,.12,k))
        if i%3==0:hi[0]=lo[0]/2 # Include invalid boxes, not just statistical ones.
        expected=bool(feasible(p,lo,hi,eps))
        a=np.maximum(lo,(1-eps)*p)
        if np.any(a>hi):actual=False
        else:
            result=linprog(np.zeros(k),A_eq=np.ones((1,k)),b_eq=[1],bounds=list(zip(a,hi)),method='highs')
            actual=bool(result.success)
        assert expected==actual
        checks+=1
    # Reuse only deterministic Gaussian utilities; no posterior is used here.
    path=OUT.parent/'04_general_contamination_model/run.py'
    spec=importlib.util.spec_from_file_location('gaussian_helpers',path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    theta=helper.parameters();pgrid=helper.laws(theta);eps=.2
    center_radius=np.sqrt(2)*.05;scale_radius=.025
    sigma_min=np.maximum(.65,theta[:,2]-scale_radius)
    eta=np.minimum(1,center_radius/(sigma_min*np.sqrt(2*np.pi))+scale_radius/sigma_min)
    lower_laws=np.maximum(pgrid-eta[:,None],0)
    geometry_radius=center_radius+helper.RADIUS*scale_radius
    offgrid=np.array([.173,-.137,.973])
    ptrue=helper.laws(offgrid[None,:])[0]
    q=helper.laws(np.array([[2.,-1.,1.2]]))[0]
    m=(1-eps)*ptrue+eps*q
    owner=np.argmin(np.max(np.abs((theta-offgrid)/np.array([.1,.1,.05])),axis=1))
    assert np.all(np.abs(theta[owner]-offgrid)<=np.array([.05,.05,.025])+1e-12)
    # Verify the bound on random interior points in all sampled parameter boxes.
    bound_checks=0
    for _ in range(500):
        i=rng.integers(len(theta));point=theta[i]+rng.uniform(-1,1,3)*np.array([.05,.05,.025])
        point=np.clip(point,[-.5,-.5,.65],[.5,.5,1.35])
        actual=helper.laws(point[None,:])[0]
        assert np.max(abs(actual-pgrid[i]))<=eta[i]+1e-12
        assert np.all(actual>=lower_laws[i]-1e-12)
        bound_checks+=1
    offgrid_checks=[]
    for n in [1000,10000,100000]:
        counts=rng.multinomial(n,m,size=200)
        all_lo,all_hi=cp(counts,n,BETA/len(m))
        covered=0;retained=0;cell_covered=0
        for lo,hi in zip(all_lo,all_hi):
            event=bool(np.all((lo<=m)&(m<=hi)))
            truth=bool(feasible(ptrue,lo,hi,eps))
            box=bool(feasible(lower_laws[owner],lo,hi,eps))
            assert not event or (truth and box)
            covered+=truth;retained+=box;cell_covered+=event
        offgrid_checks.append(dict(n=n,replicates=200,cell_region_coverage=cell_covered/200,
                                   exact_true_parameter_coverage=covered/200,owner_box_retention=retained/200))
    summary=dict(exact_binomial_results=rows,random_lp_checks=checks,random_box_bound_checks=bound_checks,
                 offgrid_sanity_checks=offgrid_checks,seed=SEED,
                 offgrid_truth=offgrid.tolist(),box_contour_inflation=geometry_radius,
                 robust_sign_population_diameter=float(2*ndtri(.5/(1-.2))))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    np.savez_compressed(OUT/'binomial_intervals.npz',**arrays)
    fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for setting,method,label in [('shared_mixture','clean_model','Clean-model interval, contaminated data'),('shared_mixture','contamination_aware','Contamination-aware interval'),('clean_left','clean_model','Clean-model interval, clean data')]:
        sub=[r for r in rows if r['setting']==setting and r['method']==method and r['true_center']==-.2]
        axs[0].semilogx([r['n'] for r in sub],[r['exact_coverage'] for r in sub],'o-',label=label)
        axs[1].loglog([r['n'] for r in sub],[r['expected_diameter'] for r in sub],'o-',label=label)
    axs[0].axhline(.95,linestyle=':',color='gray');axs[0].set(xlabel='Sample size',ylabel='Exact whole-contour coverage',ylim=(-.02,1.03));axs[0].legend(fontsize=7)
    axs[1].axhline(summary['robust_sign_population_diameter'],linestyle=':',color='gray');axs[1].set(xlabel='Sample size',ylabel='Expected contour-set diameter')
    fig.savefig(OUT/'exact_frequentist_comparison.png',dpi=180);plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
