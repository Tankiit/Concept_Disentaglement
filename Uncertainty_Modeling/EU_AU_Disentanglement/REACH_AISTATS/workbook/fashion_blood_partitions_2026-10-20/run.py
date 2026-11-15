"""Frozen boundary partitions on Fashion/Blood: paired finite-pool acquisitions.

No retraining or threshold selection. Source distributions are class-balanced
empirical populations of existing test images. Independent with-replacement
reference and deployment draws quantify finite-pool sampling uncertainty only.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import roc_auc_score
import cvxpy as cp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BASE=ROOT/'workbook/mnist_reach_novelty_2026-10-19'
BOUNDARY=ROOT/'workbook/boundary_reach_2026-10-20'
sys.path[:0]=[str(ROOT),str(BASE),str(ROOT/'workbook/reach_interval_diagnostic_2026-10-19/followup')]
from run_reach import DigitNet, data, tensor
from real.head import LDAHead
from real.uncertainty_audit import mc_probs, uncertainty
from models.ball import embedding_radius
from investigate import bounds, joint_interval

DIMENSIONS=dict(prediction=10,margin=20,support=20,margin_support=40,original_lda=11)


def balanced_draw(labels, n, seed):
    rng=np.random.default_rng(seed)
    groups=[np.flatnonzero(labels==c) for c in np.unique(labels)]
    which=rng.integers(len(groups),size=n)
    ids=np.empty(n,dtype=int)
    for j,g in enumerate(groups):
        mask=which==j
        ids[mask]=rng.choice(g,size=mask.sum(),replace=True)
    return ids


def main():
    torch.set_num_threads(4);torch.manual_seed(20261021)
    cfg=json.loads((BOUNDARY/'config.json').read_text())
    net=DigitNet()
    net.load_state_dict(torch.load(BASE/'digit_net.pt',map_location='cpu',weights_only=True))
    net.eval().requires_grad_(False)
    head=nn.Sequential(nn.Dropout(.2),nn.Linear(64,10))
    head.load_state_dict(torch.load(BOUNDARY/'boundary_head.pt',map_location='cpu',weights_only=True))
    head.eval().requires_grad_(False)
    with np.load(BASE/'partition.npz') as z:lda=LDAHead(**{k:z[k] for k in z.files})
    w,b=head[1].weight.detach().numpy(),head[1].bias.detach().numpy()
    populations={};score_frames=[]
    for name in ('mnist','fashion','blood','letters'):
        images,labels=data(name,'test')
        if name=='letters':images=images.transpose(0,2,1)
        with torch.no_grad():
            x=np.concatenate([net.features(tensor(images[i:i+256],'cpu')).numpy() for i in range(0,len(images),256)])
        z=x@w.T+b;pred=z.argmax(1)
        sorted_z=np.sort(z,axis=1);margin=sorted_z[:,-1]-sorted_z[:,-2]
        dist=lda._distances(x).min(1)
        near=(margin<cfg['margin_cut']).astype(int);far=(dist>cfg['distance_cut']).astype(int)
        cells=dict(prediction=pred,margin=2*pred+near,support=2*pred+far,
                   margin_support=4*pred+2*near+far,original_lda=lda.cell(x))
        populations[name]=dict(labels=labels,cells=cells)
        # Scores are descriptive and are never used to select thresholds/partitions.
        if name!='letters':
            scores=[uncertainty(mc_probs(head,torch.from_numpy(x[i:i+1000]),100)) for i in range(0,len(x),1000)]
            eu=np.concatenate([s['eu'] for s in scores]);tu=np.concatenate([s['tu'] for s in scores])
            score_frames.append(pd.DataFrame(dict(dataset=name,index=np.arange(len(labels)),label=labels,
                prediction=pred,margin=margin,distance=dist,low_margin=near,far=far,eu=eu,tu=tu,
                quiet=(tu<=cfg['tu_cut'])&(eu<=cfg['eu_cut']),original_lda=cells['original_lda'])))
        print('Encoded',name,len(labels),flush=True)
    scores=pd.concat(score_frames,ignore_index=True);scores.to_csv(HERE/'scores.csv',index=False)
    # Audit reproduction explicitly: near-cutoff assignments can differ across
    # execution backends. Never silently substitute the old assignments.
    old=pd.read_csv(ROOT/'workbook/fashion_blood_reach_2026-10-19/evaluation_scores.csv')
    reproduction=[]
    for name in ('mnist','fashion','blood'):
        sub=old[old.dataset==name]
        current=populations[name]['cells']['original_lda'][sub['index']]
        bad=current!=sub.cell.to_numpy()
        reproduction.append(dict(dataset=name,n=len(sub),mismatches=int(bad.sum()),
                                 indices=sub.loc[bad,'index'].tolist()))
    (HERE/'historical_reproduction.json').write_text(json.dumps(reproduction,indent=2))
    np.savez_compressed(HERE/'cell_populations.npz',**{
        f'{name}_{key}':value for name,pop in populations.items()
        for key,value in dict(labels=pop['labels'],**pop['cells']).items()})
    descriptions=[]
    known=scores[scores.dataset=='mnist']
    for name in ('mnist','fashion','blood'):
        sub=scores[scores.dataset==name]
        row=dict(dataset=name,n=len(sub),low_margin_fraction=sub.low_margin.mean(),
                 far_fraction=sub.far.mean(),quiet_fraction=sub.quiet.mean(),
                 outlier_fraction=(sub.original_lda==10).mean(),mean_eu=sub.eu.mean())
        if name=='mnist':row['known_accuracy']=(sub.prediction==sub.label).mean()
        else:
            target=np.r_[np.zeros(len(known)),np.ones(len(sub))]
            for column,sign in [('margin',-1),('distance',1),('eu',1),('tu',1)]:
                row[column+'_auroc']=roc_auc_score(target,np.r_[sign*known[column],sign*sub[column]])
        descriptions.append(row)
    pd.DataFrame(descriptions).to_csv(HERE/'domain_diagnostics.csv',index=False)

    records=[];draw_manifest=[]
    known_y=populations['mnist']['labels']
    for seed in range(20):
        # Source weights and all streams fixed across budgets, domains and partitions.
        pi=np.random.default_rng(np.random.SeedSequence([seed,505])).dirichlet(np.ones(10))
        rng=np.random.default_rng(np.random.SeedSequence([seed,303]))
        ref_ids=[rng.choice(np.flatnonzero(known_y==c),size=1000) for c in range(10)]
        rng=np.random.default_rng(np.random.SeedSequence([seed,101]))
        sources=rng.choice(10,size=4000,p=pi)
        known_ids=np.empty(4000,dtype=int)
        for c in range(10):
            mask=sources==c
            known_ids[mask]=rng.choice(np.flatnonzero(known_y==c),size=mask.sum())
        novel_uniform=rng.random(4000)
        letter_ids=balanced_draw(populations['letters']['labels'],1300,np.random.SeedSequence([seed,605]))
        for domain in ('fashion','blood'):
            novel=populations[domain]
            q_ids=balanced_draw(novel['labels'],1300,np.random.SeedSequence([seed,606]))
            log_ids=balanced_draw(novel['labels'],4000,np.random.SeedSequence([seed,607]))
            draw_manifest.append(dict(domain=domain,seed=seed,known_weights=pi.tolist(),
                reference_sha256=hashlib.sha256(np.stack(ref_ids).tobytes()).hexdigest(),
                novel_reference_sha256=hashlib.sha256(q_ids.tobytes()).hexdigest(),
                log_sha256=hashlib.sha256(known_ids.tobytes()+log_ids.tobytes()+novel_uniform.tobytes()).hexdigest()))
            for delta in (0.,.05,.2):
                is_novel=novel_uniform<delta
                for n,m in ((1000,200),(4000,1000)):
                    for name,d in DIMENSIONS.items():
                        kc=populations['mnist']['cells'][name];nc=novel['cells'][name]
                        refs=np.stack([np.bincount(kc[ix[:m]],minlength=d) for ix in ref_ids])
                        log=np.where(is_novel[:n],nc[log_ids[:n]],kc[known_ids[:n]])
                        counts=np.bincount(log,minlength=d)
                        # Same total .05 failure budget for both count-region designs.
                        for mode in ('unrestricted','matching','letters_reference'):
                            if mode=='unrestricted':
                                lo,hi=bounds(refs,.025);plo,phi=bounds(counts,.025)
                                qlo,qhi=np.zeros(d),np.ones(d)
                            else:
                                lo,hi=bounds(refs,.0125);plo,phi=bounds(counts,.025)
                                qcells=nc[q_ids] if mode=='matching' else populations['letters']['cells'][name][letter_ids]
                                qlo,qhi=bounds(np.bincount(qcells,minlength=d),.0125)
                            low,high,status=joint_interval(np.vstack([lo,qlo]),np.vstack([hi,qhi]),plo,phi,None)
                            records.append(dict(domain=domain,delta=delta,n=n,m=m,seed=seed,partition=name,
                                mode=mode,lower=low,upper=high,width=high-low,status=status,
                                detects=bool(np.isfinite(low) and low>1e-7),
                                covers=bool(low-1e-7<=delta<=high+1e-7),realized_novel_fraction=float(is_novel[:n].mean())))
                        # Existing known-only MMD compatibility test, same partition and samples.
                        wt=cp.Variable(10,nonneg=True)
                        problem=cp.Problem(cp.Minimize(cp.norm((refs/m).T@wt-counts/n)),[cp.sum(wt)==1])
                        problem.solve(solver=cp.CLARABEL)
                        if problem.status!='optimal':raise RuntimeError(problem.status)
                        radius=embedding_radius(n,.025)+embedding_radius(m,.025/10)
                        records.append(dict(domain=domain,delta=delta,n=n,m=m,seed=seed,partition=name,
                            mode='mmd_known_only',lower=np.nan,upper=np.nan,width=np.nan,status='optimal',
                            detects=bool(problem.value>radius+1e-7),covers=np.nan,
                            residual=float(problem.value),radius=radius,realized_novel_fraction=float(is_novel[:n].mean())))
        pd.DataFrame(records).to_csv(HERE/'regions.csv',index=False)
        print('Completed sampling seed',seed,flush=True)
    frame=pd.DataFrame(records)
    summary=frame.groupby(['domain','delta','n','m','partition','mode']).agg(
        lower_mean=('lower','mean'),upper_mean=('upper','mean'),width_mean=('width','mean'),
        lower_min=('lower','min'),lower_max=('lower','max'),detection_rate=('detects','mean'),
        coverage=('covers','mean'),empty_fraction=('status',lambda v:(v=='empty_set').mean()))
    summary.to_csv(HERE/'summary.csv')
    # Paired differences: combined partition minus original LDA in each shared draw.
    piv=frame[frame['mode']=='matching'].pivot(index=['domain','delta','n','m','seed'],columns='partition',values='width')
    diff=piv.subtract(piv.original_lda,axis=0)
    diff.to_csv(HERE/'paired_width_differences.csv')
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [BASE/'digit_net.pt',BASE/'partition.npz',BOUNDARY/'boundary_head.pt',BOUNDARY/'config.json']}
    dataset_hashes={name:hashlib.sha256((ROOT/f'cache/mnist_reach/{name}_test.npz').read_bytes()).hexdigest()
                    for name in populations}
    (HERE/'config.json').write_text(json.dumps(dict(frozen_artifacts=hashes,datasets=dataset_hashes,
        thresholds={k:cfg[k] for k in ['margin_cut','distance_cut','tu_cut','eu_cut']},
        partitions=DIMENSIONS,seeds=list(range(20)),budgets=[[1000,200],[4000,1000]],novel_reference_draws=1300,
        delta=[0,.05,.2],alpha=.05,paired_across_partitions=True,nested_budgets=True,
        inference='independent with-replacement draws from fixed empirical test populations; novel classes balanced',
        limitations='one encoder and one head; no new independent images or population-level coverage claim',
        historical_lda_exact_reproduction=all(r['mismatches']==0 for r in reproduction),
        historical_reproduction=reproduction),indent=2))
    (HERE/'draw_manifest.json').write_text(json.dumps(draw_manifest,indent=2))
    assert np.isfinite(frame[frame['mode'].isin(['matching','unrestricted'])][['lower','upper']]).all().all()
    assert (frame.width.dropna()>=-1e-7).all()
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
    tab=summary.reset_index()
    for j,domain in enumerate(('fashion','blood')):
        for name in DIMENSIONS:
            s=tab[(tab.domain==domain)&(tab['partition']==name)&(tab.n==4000)&(tab['mode']=='matching')]
            axes[0,j].plot(s.delta,s.width_mean,'o-',label=name)
            s=tab[(tab.domain==domain)&(tab['partition']==name)&(tab.n==4000)&(tab['mode']=='unrestricted')]
            axes[1,j].plot(s.delta,s.lower_mean,'o-',label=name)
        axes[0,j].set(title=domain,ylabel='Matching-reference prevalence interval width',ylim=(0,1.05))
        axes[1,j].set(xlabel='True novelty prevalence',ylabel='Unrestricted residual lower bound',ylim=(-.005,.205))
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Frozen partitions; n=4000, m=1000/class; 20 paired sampling seeds')
    fig.tight_layout();fig.savefig(HERE/'partition_comparison.png',dpi=160)
    print('Completed',len(frame),'region evaluations.',flush=True)


if __name__=='__main__':main()
