"""Same boundary trajectories: dropout disagreement and fixed-reference REACH."""
from pathlib import Path
import sys,json,hashlib,importlib.util
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
OLD=HERE.parent/'boundary_reach_2026-10-20'
BASE=HERE.parent/'mnist_reach_novelty_2026-10-19'
sys.path.insert(0,str(ROOT))
from real.head import LDAHead
from real.uncertainty_audit import linear_boundary_path,mc_probs,uncertainty
spec=importlib.util.spec_from_file_location('digits',HERE.parent/'digit_sources_2026-10-21/run.py')
digits=importlib.util.module_from_spec(spec);spec.loader.exec_module(digits)
bounds,label_interval=digits.bounds,digits.label_interval
DEPTHS=[0.,.5,1.,2.,4.]


class Region:
    def __init__(self,lo,hi,pl,ph):
        self.k,self.d=lo.shape; size=lo.size;rows=[];rhs=[]
        for s in range(self.k):
            row=np.zeros(size);row[s*self.d:(s+1)*self.d]=1
            for c in range(self.d):
                e=np.zeros(size);e[s*self.d+c]=1
                rows.extend([e-hi[s,c]*row,lo[s,c]*row-e]);rhs.extend([0,0])
        for c in range(self.d):
            e=np.zeros(size);e[c::self.d]=1
            rows.extend([e,-e]);rhs.extend([ph[c],-pl[c]])
        self.A=csr_matrix(np.array(rows));self.b=np.array(rhs)
        self.hom=csr_matrix(np.column_stack([np.array(rows),-self.b]))
        self.total=np.ones((1,size))
        fit=linprog(np.zeros(size),A_ub=self.A,b_ub=self.b,A_eq=self.total,b_eq=[1],bounds=(0,None))
        assert fit.status in (0,2),fit.message
        self.feasible=fit.success

    def interval(self,cell,source):
        if not self.feasible:return np.nan,np.nan,'empty_set'
        size=self.k*self.d;den=np.zeros(size)
        if cell is None:den[:]=1
        else:den[cell::self.d]=1
        mass=linprog(den,A_ub=self.A,b_ub=self.b,A_eq=self.total,b_eq=[1],bounds=(0,None))
        assert mass.success,mass.message
        if mass.fun<=1e-10:return 0.,1.,'zero_mass_policy'
        num=np.zeros(size)
        if cell is None:num[source*self.d:(source+1)*self.d]=1
        else:num[source*self.d+cell]=1
        eq=csr_matrix(np.stack([np.r_[np.ones(size),-1],np.r_[den,0]]))
        values=[]
        for sign in [1,-1]:
            fit=linprog(np.r_[sign*num,0],A_ub=self.hom,b_ub=np.zeros(len(self.b)),A_eq=eq,b_eq=[0,1],bounds=(0,None))
            assert fit.success,fit.message
            assert np.max(self.hom@fit.x)<1e-6
            assert np.max(np.abs(eq@fit.x-[0,1]))<1e-6
            values.append(float(np.clip(sign*fit.fun,0,1)))
        return *values,'optimal'


def main():
    torch.set_num_threads(4)
    z=np.load(OLD/'experiment_inputs.npz');x,y=z['features'],z['labels']
    w,b=z['weight'],z['bias'];ids=z['test_ids']
    h=np.load(BASE/'partition.npz');lda=LDAHead(**{k:h[k] for k in h.files})
    saved=pd.read_csv(OLD/'scores.csv')
    config=json.loads((OLD/'config.json').read_text())
    head=nn.Sequential(nn.Dropout(.2),nn.Linear(64,10))
    head.load_state_dict(torch.load(OLD/'boundary_head.pt',map_location='cpu',weights_only=True));head.eval()
    def partition(x):
        logits=x@w.T+b;pred=logits.argmax(1)
        margin=np.sort(logits,axis=1)[:,-1]-np.sort(logits,axis=1)[:,-2]
        near=margin<float(z['margin_cut']);far=lda._distances(x).min(1)>float(z['distance_cut'])
        return {'prediction':pred,'margin_support':4*pred+2*near+far}
    clean=partition(x);trajectories={};points=[];score_summary=[];quiet_masks={};prediction_checks=[]
    for depth in DEPTHS:
        xa,info=linear_boundary_path(w,b,x,y,depth);xa=xa.astype(np.float32)
        cells=partition(xa);trajectories[depth]=cells
        prior=saved[saved.depth==depth].set_index('index').loc[ids]
        prediction=(xa@w.T+b).argmax(1)
        mismatch=int((prediction!=prior.prediction.to_numpy()).sum())
        if depth!=1:assert mismatch==0
        prediction_checks.append(dict(depth=depth,historical_prediction_differences=mismatch))
        np.testing.assert_array_equal(info['correct'],prior.initially_correct)
        # Recompute disagreement on these exact arrays. At depth=1 floating-point
        # boundary ties can change argmax versus the historical execution.
        torch.manual_seed(20261024)
        u=uncertainty(mc_probs(head,torch.from_numpy(xa),T=100))
        quiet_wrong=(prediction!=y)&(u['tu']<=config['tu_cut'])&(u['eu']<=config['eu_cut'])
        quiet_masks[depth]=quiet_wrong
        keep=info['correct']
        score_summary.append(dict(depth=depth,accuracy_initially_correct=float((prediction[keep]==y[keep]).mean()),
            quiet_wrong_fraction=float(quiet_wrong[keep].mean()),mean_eu=float(u['eu'][keep].mean()),mean_tu=float(u['tu'][keep].mean())))
        for name,assign in cells.items():
            for i in range(len(y)):
                points.append(dict(depth=depth,partition=name,index=int(ids[i]),label=int(y[i]),cell=int(assign[i]),
                    initially_correct=bool(info['correct'][i]),prediction=int(prediction[i]),
                    eu=float(u['eu'][i]),tu=float(u['tu'][i]),quiet_wrong=bool(quiet_wrong[i])))
    pd.DataFrame(points).to_csv(HERE/'point_scores.csv',index=False)
    pd.DataFrame(score_summary).to_csv(HERE/'score_summary.csv',index=False)
    pd.DataFrame(prediction_checks).to_csv(HERE/'historical_prediction_check.csv',index=False)
    initially=(x@w.T+b).argmax(1)==y
    rows=[];runs=[];draws=[]
    theta=np.linspace(.1,.9,10)
    for seed in range(10):
        rng=np.random.default_rng(np.random.SeedSequence([20261023,seed]))
        refs=[rng.choice(np.flatnonzero(y==c),1000) for c in range(10)]
        log=rng.integers(len(y),size=4000)
        labels=np.logical_xor(rng.random((10,512))<theta[:,None],rng.random((10,512))<.2)
        hw=np.sqrt(np.log(20/.0125)/(2*512))
        tl=np.clip((labels.mean(1)-hw-.2)/.6,0,1);th=np.clip((labels.mean(1)+hw-.2)/.6,0,1)
        draws.append(dict(seed=seed,reference_indices=[v.tolist() for v in refs],log_indices=log.tolist()))
        for name,d in [('prediction',10),('margin_support',40)]:
            kc=np.stack([np.bincount(clean[name][v],minlength=d) for v in refs])
            lo,hi=bounds(kc,.025)
            clean_A=np.stack([np.bincount(clean[name][y==c],minlength=d)/sum(y==c) for c in range(10)])
            for depth in DEPTHS:
                cellids=trajectories[depth][name]
                lc=np.bincount(cellids[log],minlength=d);pl,ph=bounds(lc,.0125)
                known=Region(lo,hi,pl,ph)
                residual=Region(np.vstack([lo,np.zeros(d)]),np.vstack([hi,np.ones(d)]),pl,ph)
                rl,rh,rs=residual.interval(None,10)
                runs.append(dict(seed=seed,partition=name,depth=depth,known_only_rejected=not known.feasible,
                                 residual_lower=rl,residual_upper=rh,residual_status=rs))
                truth_A=np.stack([np.bincount(cellids[y==c],minlength=d)/sum(y==c) for c in range(10)])
                for cell in range(d):
                    source=cell if name=='prediction' else cell//4
                    observed=(cellids==cell)&initially
                    quiet=observed & quiet_masks[depth]
                    total=truth_A[:,cell].sum();ct=clean_A[:,cell].sum()
                    true_cls=truth_A[source,cell]/total if total>0 else np.nan
                    true_label=theta@clean_A[:,cell]/ct if ct>0 else np.nan
                    results={'W_cls':known.interval(cell,source),
                             'W_label':label_interval(lo,hi,tl,th,cell),
                             'W_residual':residual.interval(cell,10)}
                    for target,(lower,upper,status) in results.items():
                        truth=true_cls if target=='W_cls' else true_label if target=='W_label' else np.nan
                        rows.append(dict(seed=seed,partition=name,depth=depth,cell=cell,target=target,
                            lower=lower,upper=upper,width=upper-lower,status=status,truth=truth,
                            retains_target=bool(lower-1e-6<=truth<=upper+1e-6) if np.isfinite(truth) else np.nan,
                            evaluated_rows=int(observed.sum()),quiet_wrong_rows=int(quiet.sum()),
                            empirical_deployment_mass=total/10))
        print('Completed seed',seed,flush=True)
        pd.DataFrame(rows).to_csv(HERE/'intervals.csv',index=False)
    df=pd.DataFrame(rows);rd=pd.DataFrame(runs)
    rd.to_csv(HERE/'run_diagnostics.csv',index=False)
    (HERE/'draws.json').write_text(json.dumps(draws))
    summaries=[]
    for (part,depth,target,seed),g in df.groupby(['partition','depth','target','seed']):
        selected=g[g.evaluated_rows>0];valid=selected.width.notna()
        weights=selected.evaluated_rows.to_numpy()
        def mean(column):
            return float(np.average(selected.loc[valid,column],weights=selected.loc[valid,'evaluated_rows'])) if valid.any() else np.nan
        summaries.append(dict(partition=part,depth=depth,target=target,seed=seed,
            width=mean('width'),lower=mean('lower'),upper=mean('upper'),
            empty_fraction=float(np.average(~valid,weights=weights)),
            undefined_target_fraction=float(np.average(selected.truth.isna(),weights=weights)) if target!='W_residual' else np.nan,
            retained_fraction=float(np.average(selected.loc[selected.truth.notna(),'retains_target'].astype(bool),weights=selected.loc[selected.truth.notna(),'evaluated_rows'])) if target!='W_residual' and selected.truth.notna().any() else np.nan))
    ss=pd.DataFrame(summaries);ss.to_csv(HERE/'per_seed_summary.csv',index=False)
    summary=ss.groupby(['partition','depth','target']).agg(width=('width','mean'),lower=('lower','mean'),upper=('upper','mean'),
         empty_fraction=('empty_fraction','mean'),retained_fraction=('retained_fraction','mean'))
    summary.to_csv(HERE/'summary.csv')
    rd.groupby(['partition','depth']).agg(rejection=('known_only_rejected','mean'),residual_lower=('residual_lower','mean')).to_csv(HERE/'compatibility_summary.csv')
    config.update(dict(seeds=list(range(10)),n=4000,r_per_class=1000,m_target=512,alpha_split=dict(known=.025,log=.0125,label=.0125),
        source_input_sha256=hashlib.sha256((OLD/'experiment_inputs.npz').read_bytes()).hexdigest(),
        scores_sha256=hashlib.sha256((OLD/'scores.csv').read_bytes()).hexdigest(),
        comparison='Saved MNIST model/trajectory inputs, fresh paired MC scores (seed 20261024). Not a CLIP notebook rerun. W_cls is known-only; W_label is clean-reference anchored; W_residual is arbitrary unexplained mass, not semantic novelty.',
        averaging='Observed-cell weights over the same 973 initially correct rows; raw per-cell values also saved. Empty known-only sets remain NaN widths and count as target non-retention.'))
    (HERE/'config.json').write_text(json.dumps(config,indent=2))
    finite=df[df.status!='empty_set']
    assert np.isfinite(finite[['lower','upper']]).all().all()
    assert (finite.width>=-1e-6).all()
    assert df[df.target=='W_label'].groupby(['seed','partition','cell']).width.agg(lambda x:x.max()-x.min()).max()<1e-12
    (HERE/'validation.json').write_text(json.dumps(dict(rows=len(df),historical_predictions_match_away_from_boundary=True,
        label_width_invariant_per_cell=True,finite_nonempty_intervals=True,
        empty_intervals=int((df.status=='empty_set').sum())),indent=2))
    old_summary=pd.DataFrame(score_summary)
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    axes[0,0].plot(old_summary.depth,old_summary.accuracy_initially_correct,'o-',label='Accuracy')
    axes[0,0].plot(old_summary.depth,old_summary.quiet_wrong_fraction,'o-',label='Quiet wrong')
    axes[0,0].set_title('Same 973 initially correct examples');axes[0,0].legend()
    axes[0,1].plot(old_summary.depth,old_summary.mean_eu,'o-');axes[0,1].set_title('Dropout disagreement (nats)')
    for part in ['prediction','margin_support']:
        v=rd[rd.partition==part].groupby('depth')
        axes[0,2].plot(DEPTHS,v.known_only_rejected.mean(),'o-',label=part)
        axes[1,0].plot(DEPTHS,v.residual_lower.mean(),'o-',label=part)
        s=summary.loc[(part,slice(None),'W_cls'),:]
        axes[1,1].plot(DEPTHS,s.retained_fraction,'o-',label=part)
        axes[1,2].plot(DEPTHS,s.width,'o-',label=part)
    axes[0,2].set_title('Known-only rejection rate');axes[0,2].legend(fontsize=8)
    axes[1,0].set_title('Global unexplained-mass lower bound')
    axes[1,1].set_title('W_cls target retention (empty = failure)')
    axes[1,2].set_title('W_cls width, feasible runs only')
    for ax in axes.flat:ax.set_xlabel('Boundary-distance multiple');ax.grid(alpha=.2)
    fig.suptitle('Boundary stress: disagreement versus fixed-reference REACH\nFeature-space trajectories; class-conditional shift is deliberate; no semantic novelty claim',fontsize=14)
    fig.savefig(HERE/'boundary_epistemic.png',dpi=180);fig.savefig(HERE/'boundary_epistemic.pdf')
    print(summary.to_string());print(rd.groupby(['partition','depth']).known_only_rejected.mean().to_string())


if __name__=='__main__':main()
