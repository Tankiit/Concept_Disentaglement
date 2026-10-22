"""Candidate audit versus direct estimation with disjoint cached image evidence."""
from pathlib import Path
import importlib.util,json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
H=module('identity',HERE.parent/'camouflage_identities_2026-10-21/run.py')
TV=module('tv_solver',HERE.parent/'candidate_tv_2026-10-22/solver.py')
H.linprog=TV.solve
BUDGETS=[32,128,512,2048]


def bounds(counts,alpha):
    counts=np.asarray(counts);n=counts.sum(-1,keepdims=True)
    half=np.sqrt(np.log(2*counts.size/alpha)/(2*n))
    return np.clip(counts/n-half,0,1),np.clip(counts/n+half,0,1)


def main():
    cells=pd.read_csv(HERE/'image_cells.csv')
    def hist(frame):return np.bincount(frame.cell,minlength=11)/len(frame)
    known=[cells[(cells.dataset=='mnist')&(cells.label==s)] for s in range(10)]
    fashion=cells[cells.dataset=='fashion'];blood=cells[cells.dataset=='blood']
    A=np.vstack([*[hist(x) for x in known],hist(fashion)])
    nk=np.floor(3200*np.arange(1,11)/55).astype(int);nk[-1]+=3200-nk.sum()
    w=np.r_[nk,800]/4000;theta=np.linspace(.1,.9,10)
    cfg=dict(date='2026-10-22',seeds=list(range(10)),audit_budgets=BUDGETS,nominal_refs=512,
        known_refs_per_digit=200,n=4000,source_log_counts=np.r_[nk,800].tolist(),weights=w.tolist(),laws=A.tolist(),
        audit_alpha=.01,other_alpha=.008,identity_budgets=[0,8192],label_budget=512,eta=.2,
        source_identity_control='8192 iid synthetic identity-only draws at designed weights; zero control included',
        truth='Full finite test-image population cell laws, used only for evaluation',
        sampling='Disjoint without-replacement per-source reference/audit/log/probe streams; stratified log with fixed source counts',
        image_cells_sha256=hashlib.sha256((HERE/'image_cells.csv').read_bytes()).hexdigest())
    (HERE/'config.json').write_text(json.dumps(cfg,indent=2))
    records=[];manifest=[];budgetrows=[];runrows=[];saved=[]
    for seed in cfg['seeds']:
        rng=np.random.default_rng(np.random.SeedSequence([20261030,seed]))
        def assign(frame,parts):
            shuffled=frame.iloc[rng.permutation(len(frame))];out={};offset=0
            for role,size in parts:
                sub=shuffled.iloc[offset:offset+size];assert len(sub)==size;offset+=size;out[role]=sub
                manifest.extend(dict(seed=seed,dataset=r.dataset,index=int(r.index),role=role,label=int(r.label),cell=int(r.cell)) for r in sub.itertuples(index=False))
            return out
        kc=[];log=[]
        for s,frame in enumerate(known):
            parts=assign(frame,[('known_reference',200),('deployment',int(nk[s])),('probe',100)])
            kc.append(np.bincount(parts['known_reference'].cell,minlength=11));log.extend(parts['deployment'].cell)
        fs=assign(fashion,[('nominal_reference',512),('candidate_audit',2048),('deployment',800),('probe',2000)])
        bs=assign(blood,[('nominal_reference',512),('probe',2000)])
        log.extend(fs['deployment'].cell);lc=np.bincount(log,minlength=11)
        kl,kh=bounds(np.array(kc),.008);pl,ph=bounds(lc,.008)
        ids=np.bincount(rng.choice(11,8192,p=w),minlength=11)
        labels=np.logical_xor(rng.random((10,512))<theta[:,None],rng.random((10,512))<.2)
        h=np.sqrt(np.log(20/.008)/(2*512));tl=np.clip((labels.mean(1)-h-.2)/.6,0,1);th=np.clip((labels.mean(1)+h-.2)/.6,0,1)
        for nominal,frame in [('fashion',fs['nominal_reference']),('blood',bs['nominal_reference'])]:
            rc=np.bincount(frame.cell,minlength=11);rl,rh=bounds(rc,.008)
            true_nominal=hist(fashion if nominal=='fashion' else blood)
            for budget in BUDGETS:
                ac=np.bincount(fs['candidate_audit'].iloc[:budget].cell,minlength=11);al,ah=bounds(ac,.01)
                upper=float(min(1,.5*np.maximum(ah-rl,rh-al).sum()))
                budgetrows.append(dict(seed=seed,nominal=nominal,budget=budget,tv_upper=upper,
                    true_tv=float(np.abs(true_nominal-A[-1]).sum()/2)))
                saved.append(dict(seed=seed,nominal=nominal,budget=budget,known_counts=np.array(kc).tolist(),nominal_counts=rc.tolist(),audit_counts=ac.tolist(),log_counts=lc.tolist(),identity_counts=ids.tolist(),target_positives=labels.sum(1).tolist()))
                for idbudget in [0,8192]:
                    wl,wh=bounds(ids,.008) if idbudget else (np.zeros(11),np.ones(11))
                    for method in ['nominal','audit_tv','direct']:
                        ql,qh=(al,ah) if method=='direct' else (rl,rh)
                        lo,hi=np.vstack([kl,ql]),np.vstack([kh,qh]);rejections=[]
                        for cell in range(11):
                            src=cell if cell<10 else int(A[:-1,cell].argmax());jt=w*A[:,cell]
                            truths=dict(W_cls=jt[src]/jt[:-1].sum(),W_unk=jt[-1]/jt.sum(),W_label=theta@A[:-1,cell]/A[:-1,cell].sum())
                            results={}
                            for target,source in [('W_cls',src),('W_unk',None)]:
                                results[target]=TV.interval(lo,hi,pl,ph,wl,wh,cell,source,upper) if method=='audit_tv' else H.interval(lo,hi,pl,ph,wl,wh,cell,source)
                            results['W_label']=H.label_interval(kl,kh,tl,th,cell)
                            rejections.append(results['W_cls'][2]=='empty_set')
                            assert rejections[-1]==(results['W_unk'][2]=='empty_set')
                            for target,(lower,higher,status) in results.items():
                                truth=truths[target]
                                records.append(dict(seed=seed,nominal=nominal,budget=budget,identities=idbudget,method=method,
                                    cell=cell,region='known_cells' if cell<10 else 'outlier',target=target,lower=lower,upper=higher,
                                    width=higher-lower,status=status,truth=truth,covers=bool(lower-1e-6<=truth<=higher+1e-6)))
                        assert len(set(rejections))==1
                        runrows.append(dict(seed=seed,nominal=nominal,budget=budget,identities=idbudget,method=method,rejected=rejections[0]))
        pd.DataFrame(records).to_csv(HERE/'intervals.csv',index=False)
        print('Completed seed',seed,flush=True)
    pd.DataFrame(manifest).to_csv(HERE/'image_assignments.csv',index=False)
    pd.DataFrame(budgetrows).to_csv(HERE/'audit_bounds.csv',index=False)
    pd.DataFrame(runrows).to_csv(HERE/'compatibility.csv',index=False)
    (HERE/'counts.json').write_text(json.dumps(saved));(HERE/'solver_retries.json').write_text(json.dumps(TV.RETRIES,indent=2))
    summarize()


def summarize():
    df=pd.read_csv(HERE/'intervals.csv');rows=[]
    keys=['nominal','budget','identities','method','target','region']
    for key,g in df.groupby(keys):
        v=g[g.status!='empty_set'];rows.append(dict(zip(keys,key),records=len(g),nonempty=len(v),width=v.width.mean(),coverage=v.covers.mean()))
    s=pd.DataFrame(rows);s.to_csv(HERE/'summary.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained',sharey=True)
    for i,nominal in enumerate(['fashion','blood']):
        for j,ids in enumerate([0,8192]):
            ax=axes[i,j]
            for method in ['nominal','audit_tv','direct']:
                g=s[(s.nominal==nominal)&(s.identities==ids)&(s.method==method)&(s.target=='W_unk')&(s.region=='known_cells')].sort_values('budget')
                ax.plot(range(4),g.width,'o-',label=method)
            ax.set(title=f'Nominal {nominal}; {ids} identity observations',xticks=range(4),xticklabels=BUDGETS,xlabel='Audited candidate images',ylabel='Mean W_unk among nonempty known-cell intervals',ylim=(-.02,1.02));ax.grid(alpha=.2)
    axes[0,0].legend();fig.suptitle('Same audit evidence: direct estimation versus audit-derived TV bound\nTen paired splits; image roles disjoint within seed; missing points mean rejection')
    fig.savefig(HERE/'audit_comparison.png',dpi=180);fig.savefig(HERE/'audit_comparison.pdf')
    print(s[(s.target=='W_unk')&(s.region=='known_cells')].to_string(index=False))


if __name__=='__main__':main()
