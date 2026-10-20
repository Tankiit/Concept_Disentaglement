"""Boundary stress test on the saved MNIST encoder, with finite-pool REACH checks.

Head training and partition calibration are disjoint from the test population.
Reference/log draws for inference are independent multinomials from the clean/
perturbed finite test population. No claim of new-image population coverage.
"""
from pathlib import Path
import sys
import json
import hashlib
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
sys.path[:0]=[str(ROOT),str(BASE),str(ROOT/'workbook/reach_interval_diagnostic_2026-10-19/followup')]
from run_reach import DigitNet, data, tensor
from real.head import LDAHead
from real.uncertainty_audit import mc_probs, uncertainty, linear_boundary_path
from models.ball import embedding_radius
from investigate import bounds, joint_interval


def main():
    torch.set_num_threads(4)
    torch.manual_seed(20261020)
    rng=np.random.default_rng(20261020)
    net=DigitNet()
    net.load_state_dict(torch.load(BASE/'digit_net.pt',map_location='cpu',weights_only=True))
    net.eval().requires_grad_(False)
    def encode(images):
        with torch.no_grad():
            return np.concatenate([net.features(tensor(images[i:i+256],'cpu')).numpy()
                                   for i in range(0,len(images),256)])
    assignments=pd.read_csv(BASE/'pool_assignments.csv')
    train_x,train_y=data('mnist','train')
    fit=assignments[(assignments.dataset=='mnist')&(assignments.role=='cnn_train')]['index'].to_numpy()[:4000]
    dev=assignments[(assignments.dataset=='mnist')&(assignments.role=='partition_dev')]['index'].to_numpy()[:3000]
    assert not set(fit)&set(dev)
    xfit,xdev=encode(train_x[fit]),encode(train_x[dev])
    test_x,test_y=data('mnist','test')
    # Balanced test subset fixed before any boundary scores are observed.
    ids=np.concatenate([np.flatnonzero(test_y==c)[:100] for c in range(10)])
    xe,ye=encode(test_x[ids]),test_y[ids]
    head=nn.Sequential(nn.Dropout(.2),nn.Linear(64,10))
    opt=torch.optim.Adam(head.parameters(),lr=.003)
    fit_tensor=torch.from_numpy(xfit)
    labels=torch.from_numpy(train_y[fit])
    for _ in range(100):
        head.train()
        for batch in np.array_split(rng.permutation(len(fit)),16):
            opt.zero_grad()
            loss=nn.functional.cross_entropy(head(fit_tensor[batch]),labels[batch])
            loss.backward();opt.step()
    head.eval()
    w,b=head[1].weight.detach().numpy(),head[1].bias.detach().numpy()
    with np.load(BASE/'partition.npz') as z:
        lda=LDAHead(**{k:z[k] for k in z.files})
    zdev=xdev@w.T+b
    margin_dev=np.sort(zdev,axis=1)[:,-1]-np.sort(zdev,axis=1)[:,-2]
    distance_dev=lda._distances(xdev).min(1)
    margin_cut=float(np.quantile(margin_dev,.1))
    distance_cut=float(np.quantile(distance_dev,.95))
    cal=uncertainty(mc_probs(head,torch.from_numpy(xdev),T=100))
    tu_cut,eu_cut=float(np.quantile(cal['tu'],.9)),float(np.quantile(cal['eu'],.9))
    def partition(x):
        z=x@w.T+b
        pred=z.argmax(1)
        margin=np.sort(z,axis=1)[:,-1]-np.sort(z,axis=1)[:,-2]
        dist=lda._distances(x).min(1)
        near=(margin<margin_cut).astype(int)
        far=(dist>distance_cut).astype(int)
        return dict(prediction=pred,margin=2*pred+near,support=2*pred+far,
                    margin_support=4*pred+2*near+far,original_lda=lda.cell(x)),margin,dist
    dimensions=dict(prediction=10,margin=20,support=20,margin_support=40,original_lda=11)
    clean_cells,_,_=partition(xe)
    summaries,rows,interval_rows=[],[],[]
    clean_u=None
    for depth in (0.,.5,1.,2.,4.):
        xa,info=linear_boundary_path(w,b,xe,ye,depth)
        xa=xa.astype(np.float32)
        cells,margin,dist=partition(xa)
        pred=(xa@w.T+b).argmax(1)
        u=uncertainty(mc_probs(head,torch.from_numpy(xa),T=100))
        if clean_u is None: clean_u=u
        correct=info['correct']  # all depth comparisons use the same initially correct rows
        wrong=pred!=ye
        quiet=(u['tu']<=tu_cut)&(u['eu']<=eu_cut)
        summary=dict(depth=depth,n=len(ye),initially_correct=int(correct.sum()),
                     accuracy=float((pred==ye).mean()),accuracy_initially_correct=float((pred[correct]==ye[correct]).mean()),
                     mean_eu=float(u['eu'][correct].mean()),mean_tu=float(u['tu'][correct].mean()),
                     mean_margin=float(margin[correct].mean()),mean_distance=float(dist[correct].mean()),
                     quiet_wrong_fraction=float((wrong&quiet)[correct].mean()),
                     eu_auroc_vs_clean=float(roc_auc_score(np.r_[np.zeros(correct.sum()),np.ones(correct.sum())],
                          np.r_[clean_u['eu'][correct],u['eu'][correct]])))
        summaries.append(summary)
        for i in range(len(ye)):
            rows.append(dict(index=int(ids[i]),label=int(ye[i]),depth=depth,prediction=int(pred[i]),
                             initially_correct=bool(correct[i]),eu=float(u['eu'][i]),tu=float(u['tu'][i]),
                             margin=float(margin[i]),distance=float(dist[i]),quiet_wrong=bool(wrong[i]&quiet[i])))
        # Use the SAME random row draws across partitions and depths. Source proportions
        # are uniform and fixed. Original clean mistakes stay unchanged at all depths.
        for n,m in ((1000,200),(4000,1000)):
            for seed in range(10):
                draw=np.random.default_rng(np.random.SeedSequence([seed,n,m,92]))
                reference_ids=[draw.choice(np.flatnonzero(ye==c),size=m) for c in range(10)]
                log_ids=draw.integers(len(ye),size=n)
                for name,d in dimensions.items():
                    refs=np.stack([np.bincount(clean_cells[name][ix],minlength=d) for ix in reference_ids])
                    logs=np.bincount(cells[name][log_ids],minlength=d)
                    lo,hi=bounds(refs,.025)
                    plo,phi=bounds(logs,.025)
                    # Final source is arbitrary residual mass, not a semantic novelty label.
                    low,high,status=joint_interval(np.vstack([lo,np.zeros(d)]),
                                                  np.vstack([hi,np.ones(d)]),plo,phi,None)
                    # Existing REACH MMD-family test with clean references only.
                    wt=cp.Variable(10,nonneg=True)
                    prob=cp.Problem(cp.Minimize(cp.norm((refs/m).T@wt-logs/n)),[cp.sum(wt)==1])
                    prob.solve(solver=cp.CLARABEL)
                    if prob.status!='optimal': raise RuntimeError(prob.status)
                    radius=embedding_radius(n,.025)+embedding_radius(m,.025/10)
                    interval_rows.append(dict(depth=depth,n=n,m=m,seed=seed,partition=name,
                        residual_lower=low,residual_upper=high,status=status,
                        joint_flags_shift=bool(low>1e-7),mmd_residual=float(prob.value),mmd_radius=radius,
                        mmd_flags_shift=bool(prob.value>radius+1e-7)))
        print('Completed depth',depth,summary,flush=True)
    pd.DataFrame(rows).to_csv(HERE/'scores.csv',index=False)
    result=pd.DataFrame(summaries);result.to_csv(HERE/'boundary_summary.csv',index=False)
    regions=pd.DataFrame(interval_rows);regions.to_csv(HERE/'regions.csv',index=False)
    regions.groupby(['depth','n','m','partition']).agg(
        lower_mean=('residual_lower','mean'),lower_min=('residual_lower','min'),lower_max=('residual_lower','max'),
        joint_detection=('joint_flags_shift','mean'),mmd_detection=('mmd_flags_shift','mean')).to_csv(HERE/'region_summary.csv')
    np.savez(HERE/'experiment_inputs.npz',fit_ids=fit,calibration_ids=dev,test_ids=ids,features=xe,
             labels=ye,weight=w,bias=b,margin_cut=margin_cut,distance_cut=distance_cut)
    torch.save(head.state_dict(),HERE/'boundary_head.pt')
    (HERE/'config.json').write_text(json.dumps(dict(seed=20261020,head='input dropout .2 + linear 64 -> 10',
        head_epochs=100,mc_passes=100,fit_size=len(fit),calibration_size=len(dev),test_size=len(ids),
        model_sha256=hashlib.sha256((BASE/'digit_net.pt').read_bytes()).hexdigest(),
        margin_cut=margin_cut,distance_cut=distance_cut,tu_cut=tu_cut,eu_cut=eu_cut,
        budgets=[[1000,200],[4000,1000]],seeds=list(range(10)),
        inference='independent finite-pool reference/log draws, paired across partitions and depths',
        interpretation='unexplained mass under stable clean class conditionals; NOT semantic novelty'),indent=2))
    fig,ax=plt.subplots(1,3,figsize=(13,4))
    ax[0].plot(result.depth,result.accuracy_initially_correct,'o-',label='Accuracy')
    ax[0].plot(result.depth,result.quiet_wrong_fraction,'o-',label='Confident wrong, low EU')
    ax[0].set(xlabel='Boundary-distance multiple',ylim=(-.02,1.02));ax[0].legend(fontsize=8)
    ax[1].plot(result.depth,result.mean_eu,'o-');ax[1].set(xlabel='Boundary-distance multiple',ylabel='Mean dropout EU (nats)')
    sub=regions[regions.n==4000].groupby(['partition','depth']).residual_lower.mean()
    for name in dimensions:
        v=sub.loc[name];ax[2].plot(v.index,v.values,'o-',label=name)
    ax[2].set(xlabel='Boundary-distance multiple',ylabel='Lower bound on unexplained mass')
    ax[2].legend(fontsize=7);fig.tight_layout();fig.savefig(HERE/'boundary_comparison.png',dpi=160)


if __name__=='__main__':main()
