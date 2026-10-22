"""Fixed Fashion deployment; progressively misspecified candidate references."""
from pathlib import Path
import importlib.util
import json,hashlib
import numpy as np
import pandas as pd
from scipy.optimize import linprog
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'candidate_mismatch_2026-10-21'
spec=importlib.util.spec_from_file_location('mismatch',PARENT/'run.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
H=prior.helper
GRID=[0,.05,.1,.2,.4,.7,1.]


def population(A,p):
    k,d=A.shape
    # min ||A.T w - p||_1 over simplex, with absolute residual auxiliaries.
    fit=linprog(np.r_[np.zeros(k),np.ones(d)],
        A_ub=np.vstack([np.c_[A.T,-np.eye(d)],np.c_[-A.T,-np.eye(d)]]),
        b_ub=np.r_[p,-p],A_eq=[np.r_[np.ones(k),np.zeros(d)]],b_eq=[1],bounds=(0,None))
    assert fit.success,fit.message
    return float(fit.fun/2),fit.x[:k]


def main():
    base=json.loads((PARENT/'config.json').read_text())
    laws={k:np.array(v) for k,v in base['laws'].items()}
    w=np.array(base['weights']);theta=np.array(base['theta']);truthA=laws['fashion'];p=w@truthA
    cfg=dict(date='2026-10-22',deployed='fashion',bad_laws=['blood','emnist'],grid=GRID,seeds=list(range(10)),
        n=16000,known_refs=1000,candidate_refs=6500,labels_per_digit=512,identities=[0,8192],
        alpha_per_family=.01,narrow_threshold=.1,seed_prefix=20261029,weights=w.tolist(),theta=theta.tolist(),
        laws=base['laws'],parent_config_sha256=hashlib.sha256((PARENT/'config.json').read_bytes()).hexdigest(),
        design='Grid fixed before run; change declared q only; preserve parent draws and actual Fashion target.')
    (HERE/'config.json').write_text(json.dumps(cfg,indent=2))
    pop=[]
    for bad in cfg['bad_laws']:
        for t in GRID:
            A=truthA.copy();A[-1]=(1-t)*truthA[-1]+t*laws[bad][-1]
            distance,v=population(A,p)
            pop.append(dict(bad=bad,t=t,min_marginal_tv=distance,compatible=distance<1e-8,
                fixed_identity_tv=np.abs(w@A-p).sum()/2,candidate_tv=np.abs(A[-1]-truthA[-1]).sum()/2,
                fitted_candidate_weight=v[-1]))
    pd.DataFrame(pop).to_csv(HERE/'population.csv',index=False)
    records=[];runs=[];draws=[]
    for seed in cfg['seeds']:
        rng=np.random.default_rng(np.random.SeedSequence([20261029,seed]))
        ku=rng.random((10,1000));qu=rng.random(6500);lu=rng.random(16000)
        ids=prior.counts(rng.random(8192),w)
        labels=np.logical_xor(rng.random((10,512))<theta[:,None],rng.random((10,512))<.2)
        h=np.sqrt(np.log(20/.01)/(2*512));tl=np.clip((labels.mean(1)-h-.2)/.6,0,1);th=np.clip((labels.mean(1)+h-.2)/.6,0,1)
        kc=np.stack([prior.counts(u,row) for u,row in zip(ku,truthA[:-1])]);kl,kh=H.bounds(kc,.01)
        lc=prior.counts(lu,p);pl,ph=H.bounds(lc,.01)
        for bad in cfg['bad_laws']:
            for t in GRID:
                A=truthA.copy();A[-1]=(1-t)*truthA[-1]+t*laws[bad][-1]
                qc=prior.counts(qu,A[-1]);ql,qh=H.bounds(qc,.01)
                draws.append(dict(seed=seed,bad=bad,t=t,known_counts=kc.tolist(),candidate_counts=qc.tolist(),log_counts=lc.tolist(),identity_counts=ids.tolist(),target_positives=labels.sum(1).tolist()))
                for mode in ['estimated','exact']:
                    lo,hi=(A,A) if mode=='exact' else (np.vstack([kl,ql]),np.vstack([kh,qh]))
                    for budget in cfg['identities']:
                        wl,wh=H.bounds(ids,.01) if budget else (np.zeros(11),np.ones(11))
                        rejects=[]
                        for cell in range(11):
                            src=cell if cell<10 else int(truthA[:-1,cell].argmax())
                            jt=w*truthA[:,cell]
                            truths=dict(W_cls=jt[src]/jt[:-1].sum(),W_unk=jt[-1]/jt.sum(),W_label=theta@truthA[:-1,cell]/truthA[:-1,cell].sum())
                            results=dict(W_cls=H.interval(lo,hi,pl,ph,wl,wh,cell,src),W_unk=H.interval(lo,hi,pl,ph,wl,wh,cell,None),W_label=H.label_interval(lo[:-1],hi[:-1],tl,th,cell))
                            rejects.append(results['W_cls'][2]=='empty_set')
                            assert rejects[-1]==(results['W_unk'][2]=='empty_set')
                            for target,(lower,upper,status) in results.items():
                                truth=truths[target];cover=bool(lower-1e-6<=truth<=upper+1e-6)
                                gap=max(lower-truth,truth-upper,0) if status!='empty_set' else np.nan
                                records.append(dict(seed=seed,bad=bad,t=t,mode=mode,identities=budget,cell=cell,
                                    region='known_cells' if cell<10 else 'outlier',target=target,lower=lower,upper=upper,
                                    width=upper-lower,truth=truth,status=status,covers=cover,truth_distance=gap,
                                    narrow_wrong=bool(status!='empty_set' and upper-lower<=.1 and not cover)))
                        assert len(set(rejects))==1
                        runs.append(dict(seed=seed,bad=bad,t=t,mode=mode,identities=budget,rejected=rejects[0]))
        pd.DataFrame(records).to_csv(HERE/'intervals.csv',index=False)
        print('Completed seed',seed,flush=True)
    rf=pd.DataFrame(runs);rf.to_csv(HERE/'compatibility.csv',index=False)
    (HERE/'draws.json').write_text(json.dumps(draws))
    (HERE/'solver_retries.json').write_text(json.dumps(prior.SOLVER_RETRIES,indent=2))
    summarize()


def summarize():
    df=pd.read_csv(HERE/'intervals.csv');runs=pd.read_csv(HERE/'compatibility.csv')
    keys=['bad','t','mode','identities'];rows=[]
    for key,g in df.groupby(keys+['target','region']):
        valid=g[g.status!='empty_set']
        rows.append(dict(zip(keys+['target','region'],key),records=len(g),nonempty=len(valid),
            mean_width=valid.width.mean(),coverage=valid.covers.mean(),misses=int((~valid.covers).sum()),
            narrow_wrong=int(valid.narrow_wrong.sum()),mean_truth_distance=valid.truth_distance.mean(),
            max_truth_distance=valid.truth_distance.max()))
    s=pd.DataFrame(rows);s.to_csv(HERE/'summary.csv',index=False)
    r=runs.groupby(keys).rejected.mean().rename('rejection_rate').reset_index()
    r.to_csv(HERE/'rejection_summary.csv',index=False)
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for i,bad in enumerate(['blood','emnist']):
        for mode,color in [('estimated','#2166ac'),('exact','#b2182b')]:
            for budget,style in [(0,'--'),(8192,'-')]:
                label=f'{mode}, {budget} identities'
                sr=r[(r.bad==bad)&(r['mode']==mode)&(r.identities==budget)].sort_values('t')
                ss=s[(s.bad==bad)&(s['mode']==mode)&(s.identities==budget)&(s.target=='W_unk')&(s.region=='known_cells')].sort_values('t')
                for j,y in enumerate([sr.rejection_rate,ss.coverage,ss.mean_width]):
                    axes[i,j].plot(GRID,y,style,marker='o',color=color,label=label)
        for j,title in enumerate(['Run rejection rate','W_unk coverage among nonempty','Mean W_unk among nonempty']):
            axes[i,j].set(title=f'Toward {bad}: {title}',xlabel='Declared-law mismatch t',ylim=(-.03,1.03))
            axes[i,j].grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    fig.suptitle('Actual Fashion deployment fixed; graded candidate-law error\nKnown-cell summaries, 10 paired seeds; missing points mean all runs rejected')
    fig.savefig(HERE/'failure_boundary.png',dpi=180);fig.savefig(HERE/'failure_boundary.pdf')
    print(r.to_string(index=False))


if __name__=='__main__':main()
