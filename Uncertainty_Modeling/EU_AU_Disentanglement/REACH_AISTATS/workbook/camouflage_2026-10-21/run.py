"""Continuous candidate camouflage with ten fixed known digit sources."""
from pathlib import Path
import importlib.util,json,hashlib
import numpy as np
import pandas as pd
from scipy.optimize import linprog
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('digit_controls',HERE.parent/'digit_sources_2026-10-21/run.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
coherent,bounds,label_interval=helpers.coherent,helpers.bounds,helpers.label_interval
LAMBDAS=[0.,.5,.9,.99,1.]
BUDGETS=[1000,16000]
SEEDS=10


def identified(A,p,cell,source):
    """Exact source and marginal laws: LP over mixture weights only."""
    col=A[:,cell];num=np.zeros(len(A));den=col.copy()
    if source is None:num[-1]=col[-1]
    else:num[source]=col[source];den[-1]=0
    eq=np.vstack([A.T,np.ones(len(A))]);rhs=np.r_[p,1.]
    mass=linprog(den,A_eq=eq,b_eq=rhs,bounds=(0,None))
    assert mass.success,mass.message
    if mass.fun<=1e-10:return 0.,1.,'zero_mass_policy'
    hom=np.column_stack([eq,-rhs])
    eq2=np.vstack([hom,np.r_[den,0]])
    rhs2=np.r_[np.zeros(len(rhs)),1.]
    values=[]
    for sign in [1.,-1.]:
        fit=linprog(np.r_[sign*num,0],A_eq=eq2,b_eq=rhs2,bounds=(0,None))
        assert fit.success,fit.message
        assert np.max(np.abs(eq2@fit.x-rhs2))<1e-7
        values.append(float(np.clip(sign*fit.fun,0,1)))
    return *values,'optimal'


def draw(uniform,law):
    cdf=np.cumsum(law);cdf[-1]=1.
    return np.searchsorted(cdf,uniform,side='right')


def main():
    source=HERE.parent/'mnist_reach_novelty_2026-10-19/single_run_inputs.npz'
    original=np.load(source)['truth_conditionals'];known=original[:-1];q0=original[-1]
    camouflage=known.mean(0)
    theta=np.linspace(.1,.9,10);weights=np.r_[.8*np.arange(1,11)/55,.2]
    matrices={lam:np.vstack([known,(1-lam)*q0+lam*camouflage]) for lam in LAMBDAS}
    floors=[];geometry=[]
    for lam,A in matrices.items():
        p=weights@A;s=np.linalg.svd(A,compute_uv=False)
        geometry.append(dict(camouflage=lam,rank=int(np.linalg.matrix_rank(A)),smallest_singular_value=s[-1],condition_number=float(np.linalg.cond(A))))
        for cell in range(11):
            source_id=cell if cell<10 else int(known[:,cell].argmax())
            for target,src in [('W_cls',source_id),('W_unk',None)]:
                lo,hi,status=identified(A,p,cell,src)
                floors.append(dict(camouflage=lam,cell=cell,target=target,lower=lo,upper=hi,width=hi-lo,status=status))
    pd.DataFrame(floors).to_csv(HERE/'exact_floors.csv',index=False)
    pd.DataFrame(geometry).to_csv(HERE/'geometry.csv',index=False)
    records=[];manifests=[]
    for seed in range(SEEDS):
        rng=np.random.default_rng(np.random.SeedSequence([20261025,seed]))
        refs=np.stack([np.bincount(draw(rng.random(1000),row),minlength=11) for row in known])
        q_uniform=rng.random(6500);log_uniform=rng.random(max(BUDGETS))
        labels=np.logical_xor(rng.random((10,512))<theta[:,None],rng.random((10,512))<.2)
        h=np.sqrt(np.log(20/.0125)/(2*512))
        tl=np.clip((labels.mean(1)-h-.2)/.6,0,1);th=np.clip((labels.mean(1)+h-.2)/.6,0,1)
        kl,kh=bounds(refs,.0125)
        for lam,A in matrices.items():
            qc=np.bincount(draw(q_uniform,A[-1]),minlength=11);ql,qh=bounds(qc,.0125)
            log=draw(log_uniform,weights@A)
            for n in BUDGETS:
                lc=np.bincount(log[:n],minlength=11);pl,ph=bounds(lc,.0125)
                manifests.append(dict(seed=seed,camouflage=lam,n=n,known_counts=refs.tolist(),candidate_counts=qc.tolist(),log_counts=lc.tolist()))
                for mode in ['exact_sources','estimated_sources']:
                    lo,hi=(A,A) if mode=='exact_sources' else (np.vstack([kl,ql]),np.vstack([kh,qh]))
                    for cell in range(11):
                        source_id=cell if cell<10 else int(known[:,cell].argmax())
                        jt=weights*A[:,cell]
                        truths={'W_cls':jt[source_id]/jt[:-1].sum(),'W_unk':jt[-1]/jt.sum(),
                                'W_label':theta@known[:,cell]/known[:,cell].sum()}
                        results={'W_cls':coherent(lo,hi,pl,ph,cell,source_id),
                                 'W_unk':coherent(lo,hi,pl,ph,cell,None),
                                 'W_label':label_interval(lo[:-1],hi[:-1],tl,th,cell)}
                        for target,(lower,upper,status) in results.items():
                            records.append(dict(seed=seed,camouflage=lam,n=n,mode=mode,cell=cell,
                                region='known_cells' if cell<10 else 'outlier',target=target,
                                lower=lower,upper=upper,width=upper-lower,status=status,truth=truths[target],
                                covers=bool(lower-1e-6<=truths[target]<=upper+1e-6)))
        pd.DataFrame(records).to_csv(HERE/'intervals.csv',index=False)
        print('Finished seed',seed,flush=True)
    df=pd.DataFrame(records)
    assert len(df)==6600
    finite=df[df.status!='empty_set']
    assert np.isfinite(finite[['lower','upper']]).all().all()
    assert (finite.width>=-1e-6).all()
    label=df[df.target=='W_label']
    assert label.groupby(['seed','mode','cell']).width.agg(lambda v:v.max()-v.min()).max()<1e-12
    keys=['camouflage','n','mode','target','region']
    per_seed=df.groupby(keys+['seed']).agg(width=('width','mean'),all_cover=('covers','all'))
    per_seed.to_csv(HERE/'per_seed_summary.csv')
    summary=per_seed.groupby(level=keys).agg(width=('width','mean'),min_seed=('width','min'),max_seed=('width','max'),simultaneous_coverage=('all_cover','mean'))
    summary.to_csv(HERE/'summary.csv')
    df.groupby(['camouflage','n','mode','target','cell']).agg(width=('width','mean'),coverage=('covers','mean')).to_csv(HERE/'per_cell_summary.csv')
    (HERE/'draws.json').write_text(json.dumps(manifests))
    (HERE/'config.json').write_text(json.dumps(dict(seeds=list(range(SEEDS)),camouflage=LAMBDAS,n=BUDGETS,
        known_refs_per_digit=1000,candidate_refs=6500,target_labels_per_digit=512,eta=.2,alpha_per_event=.0125,
        candidate_prevalence=.2,weights=weights.tolist(),theta=theta.tolist(),camouflage_weights=[.1]*10,
        matrices={str(k):v.tolist() for k,v in matrices.items()},input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        scope='Frozen ten-digit model and 11 cells; candidate law interpolated in probability space, not image morphing. Independent finite-pool streams paired by uniforms across camouflage; nested log budgets. No model retraining.'),indent=2))
    (HERE/'validation.json').write_text(json.dumps(dict(records=len(df),finite_nonempty=True,ordered=True,label_invariance=True,
        empty_records=int((df.status=='empty_set').sum()),missed_targets=int((~df.covers).sum())),indent=2))
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained',sharey=True)
    floor=pd.DataFrame(floors)
    for i,mode in enumerate(['exact_sources','estimated_sources']):
        for j,target in enumerate(['W_cls','W_label','W_unk']):
            ax=axes[i,j]
            for n,color in [(1000,'#d95f02'),(16000,'#1b9e77')]:
                sub=summary.xs((n,mode,target,'known_cells'),level=['n','mode','target','region']).loc[LAMBDAS]
                ax.plot(range(5),sub.width,'o-',label=f'n={n}',color=color)
                ax.fill_between(range(5),sub.min_seed,sub.max_seed,color=color,alpha=.13)
            if target!='W_label':
                v=floor[(floor.target==target)&(floor.cell<10)].groupby('camouflage').width.mean().reindex(LAMBDAS)
                ax.plot(range(5),v,'kD:',label='Exact input-only floor')
            ax.set(title=mode.replace('_',' ')+' • '+target,xticks=range(5),xticklabels=[str(v) for v in LAMBDAS],
                xlabel='Camouflage fraction (discrete settings)',ylabel='Mean width over ten known-class cells',ylim=(-.025,1.025))
            ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Candidate camouflage: finite evidence versus exact identification\nSame ten known sources, 20% candidate mass; 10 paired seeds; bands = seed min–max',fontsize=14)
    fig.savefig(HERE/'camouflage_response.png',dpi=180);fig.savefig(HERE/'camouflage_response.pdf')
    print(summary.to_string());print(pd.DataFrame(geometry).to_string(index=False))
    print(floor[floor.cell<10].groupby(['camouflage','target']).width.mean().to_string())


if __name__=='__main__':main()
