"""Matched versus misspecified candidate laws, with paired evidence."""
from pathlib import Path
import importlib.util
import json
import hashlib
import numpy as np
import pandas as pd
from scipy.optimize import linprog
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'digit_sources_2026-10-21/config.json'
HELPER=HERE.parent/'camouflage_identities_2026-10-21/run.py'
spec=importlib.util.spec_from_file_location('identity_helper',HELPER)
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
SOLVER_RETRIES=[]


def checked_linprog(*args,**kwargs):
    result=linprog(*args,**kwargs)
    if result.status not in (0,2):
        initial=result.message
        # A numerical/unknown status is not evidence of incompatibility.
        result=linprog(*args,**kwargs,method='highs-ds',options={'presolve':False})
        SOLVER_RETRIES.append(dict(initial=initial,retry_status=int(result.status),retry_message=result.message))
    if result.status not in (0,2):
        raise RuntimeError(result.message)
    return result


helper.linprog=checked_linprog


def counts(u,p):
    cdf=np.cumsum(p);cdf[-1]=1.
    return np.bincount(np.searchsorted(cdf,u),minlength=len(p))


def main():
    config=json.loads(BASE.read_text())
    laws={name:np.array(config['matrices']['ten_'+name]) for name in ['emnist','fashion','blood']}
    w=np.r_[.8*np.arange(1,11)/55,.2];theta=np.linspace(.1,.9,10)
    records=[];runs=[];draws=[]
    for seed in range(10):
        rng=np.random.default_rng(np.random.SeedSequence([20261029,seed]))
        ku=rng.random((10,1000));qu=rng.random(6500);lu=rng.random(16000)
        ids=counts(rng.random(8192),w)
        labels=np.logical_xor(rng.random((10,512))<theta[:,None],rng.random((10,512))<.2)
        h=np.sqrt(np.log(20/.01)/(2*512))
        tl=np.clip((labels.mean(1)-h-.2)/.6,0,1);th=np.clip((labels.mean(1)+h-.2)/.6,0,1)
        kc=np.stack([counts(u,p) for u,p in zip(ku,laws['blood'][:-1])])
        kl,kh=helper.bounds(kc,.01)
        for deployed in ['fashion','blood']:
            truthA=laws[deployed];lc=counts(lu,w@truthA);pl,ph=helper.bounds(lc,.01)
            for declared in ['fashion','blood','emnist']:
                A=laws[declared];qc=counts(qu,A[-1]);ql,qh=helper.bounds(qc,.01)
                draws.append(dict(seed=seed,deployed=deployed,declared=declared,known_counts=kc.tolist(),candidate_counts=qc.tolist(),log_counts=lc.tolist(),identity_counts=ids.tolist(),target_positives=labels.sum(1).tolist()))
                for mode in ['estimated','exact']:
                    lo,hi=(A,A) if mode=='exact' else (np.vstack([kl,ql]),np.vstack([kh,qh]))
                    for budget in [0,8192]:
                        wl,wh=helper.bounds(ids,.01) if budget else (np.zeros(11),np.ones(11))
                        rejected=None
                        for cell in range(11):
                            src=cell if cell<10 else int(truthA[:-1,cell].argmax())
                            jt=w*truthA[:,cell]
                            truths=dict(W_cls=jt[src]/jt[:-1].sum(),W_unk=jt[-1]/jt.sum(),W_label=theta@truthA[:-1,cell]/truthA[:-1,cell].sum())
                            results=dict(W_cls=helper.interval(lo,hi,pl,ph,wl,wh,cell,src),W_unk=helper.interval(lo,hi,pl,ph,wl,wh,cell,None),W_label=helper.label_interval(lo[:-1],hi[:-1],tl,th,cell))
                            is_rejected=results['W_cls'][2]=='empty_set'
                            assert is_rejected==(results['W_unk'][2]=='empty_set')
                            if rejected is None:rejected=is_rejected
                            assert rejected==is_rejected
                            for target,(lower,upper,status) in results.items():
                                covered=bool(lower-1e-6<=truths[target]<=upper+1e-6)
                                records.append(dict(seed=seed,deployed=deployed,declared=declared,matched=deployed==declared,mode=mode,identities=budget,cell=cell,region='known_cells' if cell<10 else 'outlier',target=target,lower=lower,upper=upper,width=upper-lower,truth=truths[target],status=status,covers=covered,narrow_wrong=bool(status!='empty_set' and upper-lower<=.1 and not covered)))
                        runs.append(dict(seed=seed,deployed=deployed,declared=declared,mode=mode,identities=budget,rejected=rejected))
        pd.DataFrame(records).to_csv(HERE/'intervals.csv',index=False)
        print('Completed seed',seed,flush=True)
    df=pd.DataFrame(records);rf=pd.DataFrame(runs)
    rf.to_csv(HERE/'compatibility.csv',index=False)
    keys=['deployed','declared','mode','identities']
    rf.groupby(keys).rejected.agg(['mean','sum','count']).to_csv(HERE/'rejection_summary.csv')
    summaries=[]
    for key,g in df.groupby(keys+['target','region']):
        valid=g[g.status!='empty_set']
        summaries.append(dict(zip(keys+['target','region'],key),records=len(g),nonempty=len(valid),mean_width_nonempty=valid.width.mean(),coverage_nonempty=valid.covers.mean(),misses_nonempty=int((~valid.covers).sum()),narrow_wrong_nonempty=int(valid.narrow_wrong.sum())))
    summary=pd.DataFrame(summaries);summary.to_csv(HERE/'summary.csv',index=False)
    (HERE/'draws.json').write_text(json.dumps(draws))
    (HERE/'solver_retries.json').write_text(json.dumps(SOLVER_RETRIES,indent=2))
    (HERE/'config.json').write_text(json.dumps(dict(seeds=list(range(10)),laws={k:v.tolist() for k,v in laws.items()},weights=w.tolist(),theta=theta.tolist(),n=16000,known_references=1000,candidate_references=6500,target_labels=512,identities=[0,8192],alpha_per_family=.01,narrow_threshold=.1,seed_prefix=20261029,input_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest(),scope='Candidate mismatch only; known laws unchanged. Reliable external candidate identities refer to actual deployed candidate even when its declared law is wrong.'),indent=2))
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained',sharey=True)
    for i,deployed in enumerate(['fashion','blood']):
        for j,mode in enumerate(['estimated','exact']):
            ax=axes[i,j]
            for a,offset in [(0,-.18),(8192,.18)]:
                sub=rf[(rf.deployed==deployed)&(rf['mode']==mode)&(rf.identities==a)]
                vals=sub.groupby('declared').rejected.mean().reindex(['fashion','blood','emnist'])
                ax.bar(np.arange(3)+offset,vals,.36,label=f'{a} identities')
            ax.set(title=f'Deploy {deployed}; {mode} reference laws',xticks=range(3),xticklabels=['Fashion','Blood','EMNIST'],xlabel='Declared candidate law',ylabel='Fraction of seeds rejected',ylim=(0,1.05))
    axes[0,0].legend()
    fig.suptitle('Does candidate mismatch make the joint evidence incompatible?\n10 paired simulation seeds; matched laws included as controls')
    fig.savefig(HERE/'mismatch_rejection.png',dpi=180);fig.savefig(HERE/'mismatch_rejection.pdf')
    # Width and coverage together: rejected runs have no interval, not width zero.
    sub=summary[(summary.region=='known_cells')&(summary.target=='W_unk')]
    print(rf.groupby(keys).rejected.mean().to_string())
    print(sub.to_string(index=False))


if __name__=='__main__':main()
