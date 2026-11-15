"""Paired identity acquisition at near/exact camouflage, with fixed other evidence."""
from pathlib import Path
import importlib.util,json,hashlib
import numpy as np
import pandas as pd
from scipy.optimize import linprog
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent;BASE=HERE.parent/'camouflage_2026-10-21'
spec=importlib.util.spec_from_file_location('camouflage',BASE/'run.py')
cam=importlib.util.module_from_spec(spec);spec.loader.exec_module(cam)
bounds,label_interval=cam.bounds,cam.label_interval
BUDGETS=[0,128,512,2048,8192]
ALPHA=.01


def interval(lo,hi,pl,ph,wl,wh,cell,source):
    k,d=lo.shape;size=k*d;rows=[]
    for s in range(k):
        total=np.zeros(size);total[s*d:(s+1)*d]=1
        for c in range(d):
            unit=np.zeros(size);unit[s*d+c]=1
            rows.extend([np.r_[unit-hi[s,c]*total,0.],np.r_[lo[s,c]*total-unit,0.]])
        rows.extend([np.r_[total,-wh[s]],np.r_[-total,wl[s]]])
    for c in range(d):
        col=np.zeros(size);col[c::d]=1
        rows.extend([np.r_[col,-ph[c]],np.r_[-col,pl[c]]])
    matrix=np.array(rows);den=np.zeros(size);den[cell::d]=1
    if source is not None:den[(k-1)*d+cell]=0
    num=np.zeros(size);num[(k-1 if source is None else source)*d+cell]=1
    mass=linprog(den,A_ub=matrix[:,:-1],b_ub=-matrix[:,-1],A_eq=np.ones((1,size)),b_eq=[1.],bounds=(0,None))
    if mass.status==2:return np.nan,np.nan,'empty_set'
    assert mass.success,mass.message
    if mass.fun<=1e-10:return 0.,1.,'zero_mass_policy'
    eq=np.stack([np.r_[np.ones(size),-1],np.r_[den,0.]])
    values=[]
    for sign in [1.,-1.]:
        fit=linprog(np.r_[sign*num,0],A_ub=matrix,b_ub=np.zeros(len(matrix)),A_eq=eq,b_eq=[0,1],bounds=(0,None))
        assert fit.success,fit.message
        assert np.max(matrix@fit.x)<1e-6
        assert np.max(np.abs(eq@fit.x-[0,1]))<1e-6
        values.append(float(np.clip(sign*fit.fun,0,1)))
    return *values,'optimal'


def main():
    prior=json.loads((BASE/'config.json').read_text());draws=json.loads((BASE/'draws.json').read_text())
    weights=np.array(prior['weights']);theta=np.array(prior['theta'])
    records=[];identity_draws=[]
    for seed in prior['seeds']:
        # Reproduce target-label draws from the earlier camouflage run exactly.
        rng=np.random.default_rng(np.random.SeedSequence([20261025,seed]))
        rng.random((10,1000));rng.random(6500);rng.random(16000)
        labels=np.logical_xor(rng.random((10,512))<theta[:,None],rng.random((10,512))<.2)
        h=np.sqrt(np.log(20/ALPHA)/(2*512))
        tl=np.clip((labels.mean(1)-h-.2)/.6,0,1);th=np.clip((labels.mean(1)+h-.2)/.6,0,1)
        ids=np.random.default_rng(np.random.SeedSequence([20261026,seed])).choice(11,max(BUDGETS),p=weights)
        for a in BUDGETS:
            counts=np.bincount(ids[:a],minlength=11)
            identity_draws.append(dict(seed=seed,a=a,counts=counts.tolist()))
        for lam in [.99,1.]:
            A=np.array(prior['matrices'][str(lam)])
            old=next(v for v in draws if v['seed']==seed and v['camouflage']==lam and v['n']==16000)
            kl,kh=bounds(np.array(old['known_counts']),ALPHA)
            ql,qh=bounds(np.array(old['candidate_counts']),ALPHA)
            pl,ph=bounds(np.array(old['log_counts']),ALPHA)
            for mode in ['exact_sources','estimated_sources']:
                lo,hi=(A,A) if mode=='exact_sources' else (np.vstack([kl,ql]),np.vstack([kh,qh]))
                for a in BUDGETS:
                    wl,wh=bounds(np.bincount(ids[:a],minlength=11),ALPHA) if a else (np.zeros(11),np.ones(11))
                    for cell in range(11):
                        source=cell if cell<10 else int(A[:-1,cell].argmax())
                        jt=weights*A[:,cell]
                        truths={'W_cls':jt[source]/jt[:-1].sum(),'W_unk':jt[-1]/jt.sum(),
                            'W_label':theta@A[:-1,cell]/A[:-1,cell].sum()}
                        results={'W_cls':interval(lo,hi,pl,ph,wl,wh,cell,source),
                            'W_unk':interval(lo,hi,pl,ph,wl,wh,cell,None),
                            'W_label':label_interval(lo[:-1],hi[:-1],tl,th,cell)}
                        for target,(lower,upper,status) in results.items():
                            records.append(dict(seed=seed,camouflage=lam,mode=mode,a=a,cell=cell,
                                region='known_cells' if cell<10 else 'outlier',target=target,
                                lower=lower,upper=upper,width=upper-lower,status=status,truth=truths[target],
                                covers=bool(lower-1e-6<=truths[target]<=upper+1e-6)))
        pd.DataFrame(records).to_csv(HERE/'intervals.csv',index=False)
        print('Completed seed',seed,flush=True)
    df=pd.DataFrame(records);assert len(df)==6600
    assert (df[df.status!='empty_set'].width>=-1e-6).all()
    keys=['camouflage','mode','a','target','region']
    per_seed=df.groupby(keys+['seed']).agg(width=('width','mean'),all_cover=('covers','all'))
    per_seed.to_csv(HERE/'per_seed_summary.csv')
    summary=per_seed.groupby(level=keys).agg(width=('width','mean'),min_seed=('width','min'),max_seed=('width','max'),simultaneous_coverage=('all_cover','mean'))
    summary.to_csv(HERE/'summary.csv')
    df.groupby(['camouflage','mode','a','target','cell']).agg(width=('width','mean'),coverage=('covers','mean')).to_csv(HERE/'per_cell_summary.csv')
    (HERE/'identity_draws.json').write_text(json.dumps(identity_draws))
    (HERE/'config.json').write_text(json.dumps(dict(parent='camouflage_2026-10-21',parent_config_sha256=hashlib.sha256((BASE/'config.json').read_bytes()).hexdigest(),
        parent_draws_sha256=hashlib.sha256((BASE/'draws.json').read_bytes()).hexdigest(),seeds=prior['seeds'],a=BUDGETS,
        n=16000,known_refs=1000,candidate_refs=6500,target_labels=512,alpha_per_event=.01,
        camouflage=[.99,1.],scope='Only reliable iid deployment source identities vary within each paired arm; includes candidate identities. Source labels are distinct from binary target labels. Five alpha shares, including reserved unused identity share at a=0.'),indent=2))
    floor=pd.read_csv(BASE/'exact_floors.csv')
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained',sharey=True)
    for i,mode in enumerate(['exact_sources','estimated_sources']):
        for j,target in enumerate(['W_cls','W_label','W_unk']):
            ax=axes[i,j]
            for lam,color in [(.99,'#2166ac'),(1.,'#b2182b')]:
                sub=summary.xs((lam,mode,target,'known_cells'),level=['camouflage','mode','target','region']).loc[BUDGETS]
                ax.plot(range(5),sub.width,'o-',color=color,label=f'camouflage={lam}')
                ax.fill_between(range(5),sub.min_seed,sub.max_seed,color=color,alpha=.12)
            if target!='W_label':
                f=floor[(floor.camouflage==1)&(floor.cell<10)&(floor.target==target)].width.mean()
                ax.axhline(f,linestyle=':',color='black',label='100% camouflage: input-only floor')
            ax.set(title=mode.replace('_',' ')+' • '+target,xticks=range(5),xticklabels=[str(a) for a in BUDGETS],
                xlabel='Reliable source identities (discrete budgets)',ylabel='Mean width over known-class cells',ylim=(-.025,1.025))
            ax.grid(alpha=.2)
    axes[0,2].legend(fontsize=7)
    fig.suptitle('Does source-identity evidence resolve camouflage?\nFixed inputs, source references and target labels; 10 paired seeds; bands = seed min–max',fontsize=14)
    fig.savefig(HERE/'identity_response.png',dpi=180);fig.savefig(HERE/'identity_response.pdf')
    print(summary.to_string())


if __name__=='__main__':main()
