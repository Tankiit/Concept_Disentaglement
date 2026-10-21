"""PDF-aligned acquisition controls on frozen empirical image-cell laws."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.optimize import linprog
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'mnist_reach_novelty_2026-10-19'
ALPHA=.05/4
SETTINGS={'baseline':(8192,512,0),'n_low':(1024,512,0),'n_high':(65536,512,0),
          'm_low':(8192,32,0),'m_high':(8192,8192,0),
          'a_mid':(8192,512,512),'a_high':(8192,512,8192)}


def ratio(A, marginal, radius, identities, cell, novel):
    """Exact-A PDF coordinate region, optional reliable source identities."""
    ub=np.vstack([A.T,-A.T]); rhs=np.r_[marginal+radius,-marginal+radius]
    limits=[(0.,1.)]*3
    if identities is not None:
        counts,total=identities
        h=np.sqrt(np.log(6/ALPHA)/(2*total))
        limits=list(zip(np.maximum(counts/total-h,0),np.minimum(counts/total+h,1)))
    den=A[:,cell].copy()
    if not novel: den[-1]=0
    num=np.zeros(3); num[2 if novel else 0]=A[2 if novel else 0,cell]
    mass=linprog(den,A_ub=ub,b_ub=rhs,A_eq=np.ones((1,3)),b_eq=[1],bounds=limits)
    if mass.status==2: return np.nan,np.nan,'empty_set'
    assert mass.success,mass.message
    if mass.fun<=1e-10: return 0.,1.,'zero_mass_policy'
    mat=np.column_stack([ub,-rhs])
    for k,(lo,hi) in enumerate(limits):
        e=np.zeros(4);e[k]=1;e[-1]=-hi
        f=np.zeros(4);f[k]=-1;f[-1]=lo
        mat=np.vstack([mat,e,f])
    eq=np.array([[1,1,1,-1],np.r_[den,0]])
    values=[]
    for sign in [1,-1]:
        fit=linprog(np.r_[sign*num,0],A_ub=mat,b_ub=np.zeros(len(mat)),
                    A_eq=eq,b_eq=[0,1],bounds=(0,None))
        assert fit.success,fit.message
        assert np.max(mat@fit.x)<1e-6
        assert np.max(np.abs(eq@fit.x-[0,1]))<1e-6
        values.append(float(np.clip(sign*fit.fun,0,1)))
    return *values,'optimal'


def main(fashion_blood=False):
    global HERE
    source_csv=HERE.parent/'fashion_blood_reach_2026-10-19/evaluation_scores.csv'
    if fashion_blood:
        HERE=HERE/'fashion_blood'
        HERE.mkdir(exist_ok=True)
    path=BASE/'single_run_inputs.npz'
    old=np.load(path)['truth_conditionals']
    known=np.stack([old[:10:2].mean(0),old[1:10:2].mean(0)])
    worlds={'emnist':np.vstack([known,old[-1]]),
            'confounded':np.vstack([known,known.mean(0)])}
    candidate_provenance={}
    if fashion_blood:
        frame=pd.read_csv(source_csv)
        worlds={}
        for domain in ['fashion','blood']:
            sub=frame[frame.dataset==domain]
            laws=np.stack([np.bincount(g.cell,minlength=11)/len(g)
                           for _,g in sub.groupby('label')])
            worlds[domain]=np.vstack([known,laws.mean(0)])
            candidate_provenance[domain]={'images':len(sub),'classes':len(laws),
                'aggregation':'equal weight per class; empirical evaluation pool',
                'source_csv_sha256':hashlib.sha256(source_csv.read_bytes()).hexdigest()}
    w=np.array([.55,.25,.20]); theta=np.array([.2,.8]); eta=.2
    anchor=known/known.sum(0)
    target_label=theta@anchor
    records=[];floors=[];metadata={}
    for world,A in worlds.items():
        marginal=w@A
        metadata[world]={'rank':int(np.linalg.matrix_rank(A)),
                         'singular_values':np.linalg.svd(A,compute_uv=False).tolist()}
        for cell in range(11):
            for target in ['W_cls','W_unk']:
                lower,upper,status=ratio(A,marginal,0,None,cell,target=='W_unk')
                floors.append(dict(world=world,cell=cell,target=target,lower=lower,upper=upper,width=upper-lower,status=status))
        for seed in range(32):
            # Separate streams; same seed gives paired randomness across worlds.
            log=np.random.default_rng(np.random.SeedSequence([seed,101])).choice(11,65536,p=marginal)
            rng=np.random.default_rng(np.random.SeedSequence([seed,202]))
            clean=rng.random((2,8192))<theta[:,None]
            labels=np.logical_xor(clean,rng.random((2,8192))<eta)
            ids=np.random.default_rng(np.random.SeedSequence([seed,303])).choice(3,8192,p=w)
            for setting,(n,m,a) in SETTINGS.items():
                phat=np.bincount(log[:n],minlength=11)/n
                radius=(1+np.sqrt(2*np.log(1/ALPHA)))/np.sqrt(n)
                identity=(np.bincount(ids[:a],minlength=3),a) if a else None
                h=np.sqrt(np.log(4/ALPHA)/(2*m))
                low=np.clip((labels[:,:m].mean(1)-h-eta)/(1-2*eta),0,1)
                high=np.clip((labels[:,:m].mean(1)+h-eta)/(1-2*eta),0,1)
                for cell in range(11):
                    jt=w*A[:,cell]
                    truths={'W_cls':jt[0]/jt[:2].sum(),'W_unk':jt[-1]/jt.sum(),
                            'W_label':target_label[cell]}
                    cases={t:ratio(A,phat,radius,identity,cell,t=='W_unk') for t in ['W_cls','W_unk']}
                    cases['W_label']=(float(low@anchor[:,cell]),float(high@anchor[:,cell]),'analytic')
                    for target,(lower,upper,status) in cases.items():
                        records.append(dict(world=world,seed=seed,setting=setting,n=n,m=m,a=a,cell=cell,
                            target=target,lower=lower,upper=upper,width=upper-lower,truth=truths[target],
                            covers=bool(lower-1e-7<=truths[target]<=upper+1e-7),status=status))
            if seed%8==7: print(world,'seed',seed,flush=True)
    df=pd.DataFrame(records);floor=pd.DataFrame(floors)
    assert len(df)==2*32*7*11*3
    assert np.isfinite(df[['lower','upper']]).all().all()
    assert (df.width>=-1e-7).all()
    # Definition-level invariances must hold for each seed and cell.
    for target,settings in [('W_label',['baseline','n_low','n_high','a_mid','a_high']),
                            ('W_cls',['baseline','m_low','m_high']),('W_unk',['baseline','m_low','m_high'])]:
        p=df[(df.target==target)&df.setting.isin(settings)].pivot(index=['world','seed','cell'],columns='setting',values='width')
        assert np.max(p.max(axis=1)-p.min(axis=1))<1e-10
    for world in worlds:
        if metadata[world]['rank']==3:
            assert floor[floor.world==world].width.max()<1e-6
    # In the confounded world w=(.65-t/2,.35-t/2,t), 0<=t<=.7.
    if 'confounded' in worlds:
        assert floor[floor.world=='confounded'].width.max()>.1
        A=worlds['confounded']
        expected={'W_cls':1-.65*A[0]/(.65*A[0]+.35*A[1]),
                  'W_unk':.7*A[2]/(w@A)}
        for target,value in expected.items():
            np.testing.assert_allclose(floor[(floor.world=='confounded')&(floor.target==target)].width,value,atol=1e-8)
    df.to_csv(HERE/'intervals.csv',index=False);floor.to_csv(HERE/'exact_floors.csv',index=False)
    seed_summary=df.groupby(['world','setting','target','seed']).agg(width=('width','mean'),all_cover=('covers','all'))
    seed_summary.to_csv(HERE/'per_seed_summary.csv')
    summary=seed_summary.groupby(level=['world','setting','target']).agg(width=('width','mean'),min_seed=('width','min'),max_seed=('width','max'),simultaneous_coverage=('all_cover','mean'))
    summary.to_csv(HERE/'summary.csv')
    df.groupby(['world','setting','target','cell']).agg(width=('width','mean'),coverage=('covers','mean')).to_csv(HERE/'per_cell_summary.csv')
    (HERE/'config.json').write_text(json.dumps(dict(seeds=list(range(32)),settings=SETTINGS,weights=w.tolist(),theta=theta.tolist(),eta=eta,
        alpha_allocation=ALPHA,reference_weights=[.5,.5],source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_matrices={k:v.tolist() for k,v in worlds.items()},identification=metadata,
        candidate_provenance=candidate_provenance,
        scope='Exact empirical source laws; simulated independent binary target labels and reliable source identities. Candidate law known exactly. Any confounded world is constructed; named image domains use their saved cell laws.'),indent=2))
    targets=['W_cls','W_label','W_unk'];colors=['#2166ac','#b2182b','#1b7837']
    sweeps=[('Deployment inputs n',['n_low','baseline','n_high'],[1024,8192,65536]),
            ('Reference target labels m/source',['m_low','baseline','m_high'],[32,512,8192]),
            ('Deployment source identities a',['baseline','a_mid','a_high'],[0,512,8192])]
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained',sharey=True)
    for i,world in enumerate(worlds):
        for j,(xlabel,settings,x) in enumerate(sweeps):
            ax=axes[i,j]
            for target,color in zip(targets,colors):
                sub=summary.loc[(world,),:].xs(target,level='target').loc[settings]
                ax.plot(x,sub.width,'o-',label=target,color=color)
                ax.fill_between(x,sub.min_seed,sub.max_seed,color=color,alpha=.12)
                if j==0 and target!='W_label':
                    f=floor[(floor.world==world)&(floor.target==target)].width.mean()
                    ax.axhline(f,color=color,linestyle=':',alpha=.8)
            ax.set_xscale('symlog',linthresh=512) if j==2 else ax.set_xscale('log')
            ax.set_xticks(x,[str(v) for v in x]);ax.set(xlabel=xlabel,ylabel='Mean per-cell width',ylim=(-.025,1.025),title=world)
            ax.grid(alpha=.2)
    axes[0,0].legend()
    fig.suptitle('Three epistemic targets: acquisition responses on frozen image-cell laws\n32 seeds; mean across 11 cells; shading = seed min–max; dotted = exact input-only floors',fontsize=14)
    fig.savefig(HERE/'acquisition_responses.png',dpi=180);fig.savefig(HERE/'acquisition_responses.pdf')
    print(summary.to_string());print('Exact floors:',floor.groupby(['world','target']).width.mean().to_string())
    (HERE/'validation.json').write_text(json.dumps(dict(records=len(df),finite_ordered=True,invariance_checks=True,exact_floor_checks=True,
        analytic_confounded_floors_verified='confounded' in worlds,
        empty_sets=int((df.status=='empty_set').sum()),zero_mass=int((df.status=='zero_mass_policy').sum()),coverage_failures=int((~df.covers).sum())),indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--fashion-blood',action='store_true')
    main(parser.parse_args().fashion_blood)
