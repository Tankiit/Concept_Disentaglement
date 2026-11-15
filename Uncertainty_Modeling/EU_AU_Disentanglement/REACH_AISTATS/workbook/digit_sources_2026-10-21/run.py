"""Individual-digit epistemic targets, paired exact/finite reference controls."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
from scipy.optimize import linprog
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'mnist_emnist_controlled_2026-10-21'))
# Load by explicit path: that workbook also has a module named run.
import importlib.util
spec=importlib.util.spec_from_file_location('controlled',HERE.parent/'mnist_emnist_controlled_2026-10-21/run.py')
controlled=importlib.util.module_from_spec(spec);spec.loader.exec_module(controlled)
coherent,bounds=controlled.coherent,controlled.bounds
SEEDS=20


def label_interval(lo,hi,tl,th,cell):
    # Project each normalized source-row box onto this cell. The fixed anchor
    # is uniform across known sources. Source rows vary independently here;
    # the deployment log does not enter the reference-anchored target.
    lower=np.maximum(lo[:,cell],1-(hi.sum(1)-hi[:,cell]))
    upper=np.minimum(hi[:,cell],1-(lo.sum(1)-lo[:,cell]))
    if lower.sum()<=1e-10: return 0.,1.,'zero_mass_policy'
    k=len(lower);mat=[]
    for j in range(k):
        e=np.zeros(k+1);e[j]=1;e[-1]=-upper[j];mat.append(e)
        e=np.zeros(k+1);e[j]=-1;e[-1]=lower[j];mat.append(e)
    values=[]
    for cost,sign in [(tl,1),(th,-1)]:
        result=linprog(np.r_[cost*sign,0],A_ub=mat,b_ub=np.zeros(2*k),
            A_eq=[np.r_[np.ones(k),0]],b_eq=[1],bounds=(0,None))
        assert result.success,result.message
        assert np.max(np.array(mat)@result.x)<1e-6
        values.append(float(np.clip(sign*result.fun,0,1)))
    return *values,'optimal'


def known_only_feasible(lo,hi,pl,ph):
    k,d=lo.shape;rows=[];rhs=[]
    for s in range(k):
        row=np.zeros(k*d);row[s*d:(s+1)*d]=1
        for c in range(d):
            unit=np.zeros(k*d);unit[s*d+c]=1
            rows.extend([unit-hi[s,c]*row,lo[s,c]*row-unit]);rhs.extend([0,0])
    for c in range(d):
        col=np.zeros(k*d);col[c::d]=1
        rows.extend([col,-col]);rhs.extend([ph[c],-pl[c]])
    fit=linprog(np.zeros(k*d),A_ub=rows,b_ub=rhs,A_eq=np.ones((1,k*d)),b_eq=[1],bounds=(0,None))
    assert fit.status in (0,2),fit.message
    return fit.status==0


def main():
    old_path=HERE.parent/'mnist_reach_novelty_2026-10-19/single_run_inputs.npz'
    old=np.load(old_path)['truth_conditionals']
    ext_path=HERE.parent/'fashion_blood_reach_2026-10-19/evaluation_scores.csv'
    frame=pd.read_csv(ext_path)
    worlds={'ten_emnist':old}
    for name in ['fashion','blood']:
        sub=frame[frame.dataset==name]
        q=np.stack([np.bincount(g.cell,minlength=11)/len(g) for _,g in sub.groupby('label')]).mean(0)
        worlds['ten_'+name]=np.vstack([old[:-1],q])
    held=np.load(HERE/'heldout_source_laws.npz')['laws']
    worlds['heldout_6_9']=np.vstack([held[:6],held[6:].mean(0)])
    records=[];reject=[];manifest=[]
    for world,A in worlds.items():
        k,d=A.shape[0]-1,A.shape[1]
        theta=np.linspace(.1,.9,k)
        known_weights=np.arange(1,k+1,dtype=float);known_weights/=known_weights.sum()
        deltas=[.0,.2] if world=='heldout_6_9' else [.2]
        for delta in deltas:
            weights=np.r_[(1-delta)*known_weights,delta];marginal=weights@A
            for seed in range(SEEDS):
                rng=np.random.default_rng(np.random.SeedSequence([20261022,k,seed]))
                refs=[rng.choice(d,1000,p=row) for row in A[:-1]]
                qrefs=rng.choice(d,6500,p=A[-1]);log=rng.choice(d,8192,p=marginal)
                clean=rng.random((k,512))<theta[:,None]
                labels=np.logical_xor(clean,rng.random((k,512))<.2)
                # Four simultaneous evidence allocations, all .0125.
                h=np.sqrt(np.log(2*k/.0125)/(2*512))
                tl=np.clip((labels.mean(1)-h-.2)/.6,0,1)
                th=np.clip((labels.mean(1)+h-.2)/.6,0,1)
                lc=np.bincount(log,minlength=d);pl,ph=bounds(lc,.0125)
                modes=['exact','estimated_200','estimated_1000']
                if world=='heldout_6_9':modes+=['unrestricted_200','unrestricted_1000']
                for mode in modes:
                    r=1000 if mode.endswith('1000') else 200
                    q=6500 if r==1000 else 1300
                    kc=np.stack([np.bincount(x[:r],minlength=d) for x in refs])
                    qc=np.bincount(qrefs[:q],minlength=d)
                    if mode=='exact':lo,hi=A.copy(),A.copy()
                    else:
                        kl,kh=bounds(kc,.0125);ql,qh=bounds(qc,.0125)
                        if mode.startswith('unrestricted'):ql,qh=np.zeros(d),np.ones(d)
                        lo,hi=np.vstack([kl,ql]),np.vstack([kh,qh])
                    if mode in ['exact','estimated_200','estimated_1000']:
                        reject.append(dict(world=world,delta=delta,seed=seed,mode=mode,
                            known_only_rejected=not known_only_feasible(lo[:-1],hi[:-1],pl,ph)))
                    manifest.append(dict(world=world,delta=delta,seed=seed,mode=mode,
                        log_counts=lc.tolist(),known_counts=kc.tolist(),candidate_counts=None if mode.startswith('unrestricted') else qc.tolist()))
                    for cell in range(d):
                        source=cell if cell<k else int(A[:-1,cell].argmax())
                        jt=weights*A[:,cell]
                        truths={'W_cls':jt[source]/jt[:-1].sum(),'W_unk':jt[-1]/jt.sum(),
                                'W_label':theta@A[:-1,cell]/A[:-1,cell].sum()}
                        results={'W_cls':coherent(lo,hi,pl,ph,cell,source),
                                 'W_unk':coherent(lo,hi,pl,ph,cell,None),
                                 'W_label':label_interval(lo[:-1],hi[:-1],tl,th,cell)}
                        for target,(lower,upper,status) in results.items():
                            records.append(dict(world=world,delta=delta,seed=seed,mode=mode,cell=cell,
                                region='known_cells' if cell<k else 'outlier',target=target,yhat=source,
                                lower=lower,upper=upper,width=upper-lower,truth=truths[target],status=status,
                                covers=bool(lower-1e-6<=truths[target]<=upper+1e-6)))
                if seed%5==4:print(world,delta,'seed',seed,flush=True)
        pd.DataFrame(records).to_csv(HERE/'intervals.csv',index=False)
    df=pd.DataFrame(records)
    finite=df[df.status!='empty_set']
    assert np.isfinite(finite[['lower','upper','truth']]).all().all()
    assert (finite.width>=-1e-6).all() and (finite.width<=1+1e-6).all()
    keys=['world','delta','mode','target','region']
    per_seed=df.groupby(keys+['seed']).agg(width=('width','mean'),all_cover=('covers','all'))
    per_seed.to_csv(HERE/'per_seed_summary.csv')
    summary=per_seed.groupby(level=keys).agg(width=('width','mean'),min_seed=('width','min'),max_seed=('width','max'),simultaneous_coverage=('all_cover','mean'))
    summary.to_csv(HERE/'summary.csv')
    df.groupby(['world','delta','mode','target','cell']).agg(width=('width','mean'),coverage=('covers','mean')).to_csv(HERE/'per_cell_summary.csv')
    pd.DataFrame(reject).to_csv(HERE/'known_only_rejections.csv',index=False)
    (HERE/'draw_manifest.json').write_text(json.dumps(manifest))
    (HERE/'config.json').write_text(json.dumps(dict(seeds=list(range(SEEDS)),n=8192,m_target=512,
        known_reference_budgets=[200,1000],candidate_reference_budgets=[1300,6500],eta=.2,alpha=.05,
        alpha_allocation=.0125,theta='linspace(.1,.9,K)',known_weights='proportional to 1..K',
        matrices={name:A.tolist() for name,A in worlds.items()},
        original_input_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest(),
        candidate_input_sha256=hashlib.sha256(ext_path.read_bytes()).hexdigest(),
        scope='Paired finite-pool iid acquisitions; fixed individual-digit sources. Exact/estimated source laws use the same joint-mass region with CP log bounds. Simulated target labels. Unrestricted unknown source has no supplied candidate law.'),indent=2))
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained',sharey=True)
    for ax,world in zip(axes.flat,worlds):
        for target,color in zip(['W_cls','W_label','W_unk'],['#2166ac','#b2182b','#1b7837']):
            values=[];mins=[];maxs=[]
            for mode in ['exact','estimated_200','estimated_1000']:
                row=summary.loc[(world,.2,mode,target,'known_cells')]
                values.append(row.width);mins.append(row.min_seed);maxs.append(row.max_seed)
            ax.plot(range(3),values,'o-',label=target,color=color)
            ax.fill_between(range(3),mins,maxs,color=color,alpha=.12)
        ax.set(title=world,xticks=range(3),xticklabels=['Exact source laws','200 / 1300 refs','1000 / 6500 refs'],ylim=(-.02,1.02),ylabel='Mean width: known-class cells')
        ax.grid(alpha=.2)
    axes[0,0].legend()
    fig.suptitle('Individual-digit sources: exact versus estimated reference laws\n20 paired seeds; 20% candidate prevalence; same targets, log, labels and confidence-set construction',fontsize=13)
    fig.savefig(HERE/'reference_comparison.png',dpi=180);fig.savefig(HERE/'reference_comparison.pdf')
    print(summary.to_string())
    print(pd.DataFrame(reject).groupby(['world','delta','mode']).known_only_rejected.mean().to_string())
    (HERE/'validation.json').write_text(json.dumps(dict(records=len(df),finite_nonempty=True,ordered=True,
        empty_records=int((df.status=='empty_set').sum()),zero_mass_records=int((df.status=='zero_mass_policy').sum()),
        missed_targets=int((~df.covers).sum())),indent=2))


if __name__=='__main__':main()
