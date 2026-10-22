"""Paired TV sensitivity experiment; preserves all coordinate-baseline evidence."""
from solver import interval as tv_interval, RETRIES
from pathlib import Path
import importlib.util
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'graded_mismatch_2026-10-22'
spec=importlib.util.spec_from_file_location('graded',PARENT/'run.py')
graded=importlib.util.module_from_spec(spec);spec.loader.exec_module(graded)
H=graded.H
DELTAS=[0,.01,.05,.1,.25,.5,1.]
CASES=[('matched','blood',0.),('blood_full','blood',1.),('emnist_partial','emnist',.4)]


def main():
    cfg=json.loads((PARENT/'config.json').read_text())
    draws=json.loads((PARENT/'draws.json').read_text())
    old=pd.read_csv(PARENT/'intervals.csv')
    A=np.array(cfg['laws']['fashion'])
    (HERE/'config.json').write_text(json.dumps(dict(deltas=DELTAS,cases=CASES,seeds=cfg['seeds'],identities=[0,8192],
        norm='Total variation to some normalized nominal candidate row within its confidence box',mode='estimated',alpha_per_family=.01,
        scope='Only candidate bounds expand; all other evidence reused. Delta grid fixed before run; truths used only for evaluation.',
        parent_config_sha256=hashlib.sha256((PARENT/'config.json').read_bytes()).hexdigest(),
        parent_draws_sha256=hashlib.sha256((PARENT/'draws.json').read_bytes()).hexdigest()),indent=2))
    records=[];runs=[];diagnostics=[]
    for seed in cfg['seeds']:
        for case,bad,t in CASES:
            d=next(v for v in draws if v['seed']==seed and v['bad']==bad and v['t']==t)
            kl,kh=H.bounds(np.array(d['known_counts']),.01)
            ql,qh=H.bounds(np.array(d['candidate_counts']),.01)
            pl,ph=H.bounds(np.array(d['log_counts']),.01)
            q=(1-t)*A[-1]+t*np.array(cfg['laws'][bad])[-1]
            # Evaluation diagnostics, never used to select delta.
            diagnostics.append(dict(seed=seed,case=case,true_tv_error=float(np.sum(np.abs(q-A[-1]))/2),
                minimum_box_expansion_for_true_row=float(max(0,np.max(ql-A[-1]),np.max(A[-1]-qh)))))
            for identities in [0,8192]:
                wl,wh=H.bounds(np.array(d['identity_counts']),.01) if identities else (np.zeros(11),np.ones(11))
                baseline=old[(old.seed==seed)&(old.bad==bad)&(old.t==t)&(old['mode']=='estimated')&(old.identities==identities)].set_index(['cell','target'])
                for delta in DELTAS:
                    lo=np.vstack([kl,ql]);hi=np.vstack([kh,qh])
                    rejection=[]
                    for cell in range(11):
                        src=cell if cell<10 else A[:-1,cell].argmax()
                        results=dict(W_cls=tv_interval(lo,hi,pl,ph,wl,wh,cell,src,delta),
                            W_unk=tv_interval(lo,hi,pl,ph,wl,wh,cell,None,delta),
                            W_label=tuple(baseline.loc[(cell,'W_label'),['lower','upper','status']]))
                        rejection.append(results['W_cls'][2]=='empty_set')
                        assert rejection[-1]==(results['W_unk'][2]=='empty_set')
                        for target,(lower,upper,status) in results.items():
                            truth=float(baseline.loc[(cell,target),'truth']);covered=bool(lower-1e-6<=truth<=upper+1e-6)
                            records.append(dict(seed=seed,case=case,identities=identities,delta=delta,cell=cell,
                                region='known_cells' if cell<10 else 'outlier',target=target,lower=lower,upper=upper,
                                width=upper-lower,status=status,truth=truth,covers=covered,
                                truth_distance=max(lower-truth,truth-upper,0) if status!='empty_set' else np.nan,
                                narrow_wrong=bool(status!='empty_set' and upper-lower<=.1 and not covered)))
                    assert len(set(rejection))==1
                    runs.append(dict(seed=seed,case=case,identities=identities,delta=delta,rejected=rejection[0]))
        pd.DataFrame(records).to_csv(HERE/'intervals.csv',index=False)
        print('Completed seed',seed,flush=True)
    pd.DataFrame(runs).to_csv(HERE/'compatibility.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(HERE/'oracle_diagnostics.csv',index=False)
    (HERE/'solver_retries.json').write_text(json.dumps(RETRIES,indent=2))
    summarize()


def summarize():
    df=pd.read_csv(HERE/'intervals.csv');rf=pd.read_csv(HERE/'compatibility.csv')
    keys=['case','identities','delta'];rows=[]
    for key,g in df.groupby(keys+['target','region']):
        v=g[g.status!='empty_set']
        rows.append(dict(zip(keys+['target','region'],key),records=len(g),nonempty=len(v),mean_width=v.width.mean(),
            coverage=v.covers.mean(),narrow_wrong=int(v.narrow_wrong.sum()),mean_truth_distance=v.truth_distance.mean()))
    s=pd.DataFrame(rows);s.to_csv(HERE/'summary.csv',index=False)
    r=rf.groupby(keys).rejected.mean().rename('rejection_rate').reset_index();r.to_csv(HERE/'rejection_summary.csv',index=False)
    fig,axes=plt.subplots(3,3,figsize=(15,11),layout='constrained')
    for i,(case,_,_) in enumerate(CASES):
        for identities,color in [(0,'#2166ac'),(8192,'#b2182b')]:
            ss=s[(s.case==case)&(s.identities==identities)&(s.target=='W_unk')&(s.region=='known_cells')].sort_values('delta')
            rr=r[(r.case==case)&(r.identities==identities)].sort_values('delta')
            for j,values in enumerate([rr.rejection_rate,ss.coverage,ss.mean_width]):
                axes[i,j].plot(range(len(DELTAS)),values,'o-',color=color,label=f'{identities} identities')
        for j,title in enumerate(['Rejection rate','Coverage among nonempty','Mean W_unk among nonempty']):
            axes[i,j].set(title=case+': '+title,xticks=range(len(DELTAS)),xticklabels=DELTAS,xlabel='Candidate error allowance delta (discrete grid)',ylim=(-.03,1.03))
            axes[i,j].grid(alpha=.2)
    axes[0,0].legend()
    fig.suptitle('TV candidate-law sensitivity: coverage versus informativeness\nFixed Fashion deployment; finite references; 10 paired seeds; missing = all rejected')
    fig.savefig(HERE/'sensitivity.png',dpi=180);fig.savefig(HERE/'sensitivity.pdf')
    print(s[(s.target=='W_unk')&(s.region=='known_cells')].to_string(index=False))


if __name__=='__main__':main()
