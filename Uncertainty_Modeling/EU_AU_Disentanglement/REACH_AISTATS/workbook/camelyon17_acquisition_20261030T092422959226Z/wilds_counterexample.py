"""Real-record near-match search; negative results are first-class outputs."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import beta
from real.beta_acquisition import moments, action_values, epistemic_mi, update


def target_interval(shapes, alpha=.05):
    """Posterior credible enclosure, NOT an identification/support interval.

    Bonferroni over three independent Beta parameters; multilinear target
    extrema occur at box vertices. Posterior content >=1-alpha under model.
    """
    s = np.asarray(shapes)
    lo = beta.ppf(alpha/6, s[:,0], s[:,1])
    hi = beta.ppf(1-alpha/6, s[:,0], s[:,1])
    values = [(1-w)*t0+w*t1 for w in [lo[0],hi[0]] for t0 in [lo[1],hi[1]] for t1 in [lo[2],hi[2]]]
    return min(values), max(values)


def run_counterexample(records, output, seed=20261030, state_repeats=20, repeats=2000):
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    protocol = dict(seed=seed, identity_budgets=[0,2,8,32,128,512], label_budgets=[2,8,32,128,512],
                    state_repeats=state_repeats, mi_relative_tolerance=.01, mean_tolerance=.02,
                    variance_relative_tolerance=.10, action_margin=2, validation_repeats=repeats,
                    budgets=[0,1,2,4,8,16,32,64,128], target='same binary population prevalence in both states',
                    selection='smallest relative MI difference; then state IDs; never use evaluation outcomes',
                    interpretation='finite empirical-population resampling; no population-level iid-patch inference')
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    train = records[(records.role=='fit') & (records.source<2) & ~records.temporal_ood].copy()
    test = records[(records.role=='evaluation') & (records.source<2) & ~records.temporal_ood].copy()
    # Camelyon: tumor=1. FMoW: most frequent training category, deterministic tie-break.
    labels = sorted(train.target.unique())
    event = 1 if labels == [0,1] else int(train.target.value_counts().rename_axis('label').reset_index(name='n').sort_values(['n','label'],ascending=[False,True]).iloc[0].label)
    train['binary_target'] = (train.target==event).astype(int)
    test['binary_target'] = (test.target==event).astype(int)
    (out/'target_definition.json').write_text(json.dumps(dict(positive_class=event,selection='tumor=1 if binary, else modal training class; frozen before heldout labels'),indent=2))
    rng = np.random.default_rng(seed)
    pools = [np.flatnonzero(train.source.to_numpy()==s) for s in [0,1]]
    if min(map(len,pools)) == 0 or test.groupby('source').size().shape[0] != 2:
        raise ValueError('Both known sources must exist in train and evaluation')
    rows=[]; observations={}
    for ni in protocol['identity_budgets']:
        for nl in protocol['label_budgets']:
            for _ in range(state_repeats):
                idx = rng.integers(len(train),size=ni)
                lab = [rng.choice(pool,nl) for pool in pools]
                k = int(train.source.iloc[idx].sum())
                successes = [int(train.binary_target.iloc[ix].sum()) for ix in lab]
                shape = np.array([[1+k,1+ni-k]]+[[1+j,1+nl-j] for j in successes],float)
                mean,var = moments(shape); vi,vl=action_values(shape)
                sid=len(rows)
                rows.append(dict(state_id=sid,n_identity=ni,n_labels_per_source=nl,mean=float(mean),variance=float(var),
                                 mi=epistemic_mi(shape),value_identity=float(vi),value_label=float(vl),shapes=json.dumps(shape.tolist())))
                observations[sid]=dict(identity=train.record_id.iloc[idx].tolist(),labels=[train.record_id.iloc[ix].tolist() for ix in lab])
    states=pd.DataFrame(rows);states.to_csv(out/'candidate_states.csv',index=False)
    (out/'initial_observation_ids.json').write_text(json.dumps(observations))
    pairs=[]
    for a in states[states.value_identity>=2*states.value_label].itertuples():
        for b in states[states.value_label>=2*states.value_identity].itertuples():
            md=abs(a.mi-b.mi)/max((a.mi+b.mi)/2,1e-15)
            vd=abs(a.variance-b.variance)/max((a.variance+b.variance)/2,1e-15)
            if md<=.01 and vd<=.1 and abs(a.mean-b.mean)<=.02:
                pairs.append(dict(identity_state=a.state_id,label_state=b.state_id,mi_relative_difference=md,
                                  mean_difference=abs(a.mean-b.mean),variance_relative_difference=vd))
    pairs=pd.DataFrame(pairs,columns=['identity_state','label_state','mi_relative_difference','mean_difference','variance_relative_difference'])
    pairs=pairs.sort_values(['mi_relative_difference','identity_state','label_state'])
    pairs.to_csv(out/'qualifying_pairs.csv',index=False)
    if pairs.empty:
        result=dict(status='no_match',candidates=len(states),matches=0,confirmed=False)
        (out/'RESULTS.md').write_text('# Counterexample search\n\nNo pair met the prespecified criteria. No thresholds were relaxed. This is not evidence of an empirical counterexample.\n')
        (out/'result.json').write_text(json.dumps(result,indent=2));return result
    match=pairs.iloc[0]
    chosen=states.set_index('state_id').loc[[int(match.identity_state),int(match.label_state)]].reset_index()
    chosen['role']=['identity_preferred','label_preferred']
    chosen['mi_order48']=[epistemic_mi(np.array(json.loads(s)),48) for s in chosen.shapes]
    if np.max(abs(chosen.mi-chosen.mi_order48))>1e-7:
        raise RuntimeError('Quadrature convergence failed; no claim allowed')
    chosen.to_csv(out/'selected_states.csv',index=False)
    truth=float(test.binary_target.mean());w=float(test.source.mean())
    rates=[float(test[test.source==s].binary_target.mean()) for s in [0,1]]
    pd.DataFrame([dict(split=name,source=s,n=len(f[f.source==s]),target_rate=f[f.source==s].binary_target.mean())
                  for name,f in [('train',train),('heldout',test)] for s in [0,1]]).to_csv(out/'population_rates.csv',index=False)
    exact=[]; intervals=[];checks=[]
    import torch
    from probly.representation.distribution.torch_categorical import TorchProbabilityCategoricalDistribution,TorchCategoricalDistributionSample
    from probly.quantification import quantify
    for row in chosen.itertuples():
        shape=np.array(json.loads(row.shapes));mu,var=moments(shape)
        lo,hi=target_interval(shape)
        intervals.append(dict(state_id=row.state_id,lower=lo,upper=hi,width=hi-lo,kind='95% posterior credible enclosure; support remains [0,1]'))
        draws=rng.beta(shape[:,0],shape[:,1],size=(100000,3));t=(1-draws[:,0])*draws[:,1]+draws[:,0]*draws[:,2]
        rep=TorchCategoricalDistributionSample(tensor=TorchProbabilityCategoricalDistribution(torch.tensor(np.c_[1-t,t])),sample_dim=0)
        checks.append(dict(state_id=row.state_id,quadrature=row.mi,probly=float(quantify(rep).epistemic)))
        for action in ['source_id','reference_label']:
            groups=[(None,1,w)] if action=='source_id' else [(0,.5,rates[0]),(1,.5,rates[1])]
            err=0
            for source,weight,rate in groups:
                for y,p in [(0,1-rate),(1,rate)]:
                    mean,_=moments(update(shape,action,y,source));err+=weight*p*(mean-truth)**2
            exact.append(dict(state_id=row.state_id,role=row.role,action=action,expected_squared_error=float(err),initial_squared_error=float((mu-truth)**2)))
    pd.DataFrame(intervals).to_csv(out/'target_credible_enclosures.csv',index=False)
    pd.DataFrame(checks).to_csv(out/'probly_check.csv',index=False)
    exact=pd.DataFrame(exact);exact.to_csv(out/'heldout_exact_one_step.csv',index=False)
    pivot=exact.pivot(index='role',columns='action',values='expected_squared_error')
    advantages=[float(pivot.loc['identity_preferred','reference_label']-pivot.loc['identity_preferred','source_id']),
                float(pivot.loc['label_preferred','source_id']-pivot.loc['label_preferred','reference_label'])]
    # Paired real-record streams; standard errors describe Monte Carlo variation only.
    rg=np.random.default_rng(seed+9000);size=(repeats,128)
    ix=rg.integers(len(test),size=size);ss=rg.integers(2,size=size);lx=np.empty(size,int)
    for s in [0,1]:lx[ss==s]=rg.choice(np.flatnonzero(test.source.to_numpy()==s),int((ss==s).sum()))
    identities=test.source.to_numpy()[ix];labels=test.binary_target.to_numpy()[lx]
    np.savez_compressed(out/'acquisition_record_ids.npz',identity=test.record_id.to_numpy()[ix].astype(str),
                        label=test.record_id.to_numpy()[lx].astype(str),label_sources=ss,identity_outcomes=identities,label_outcomes=labels)
    trajectory=[]
    for row in chosen.itertuples():
        for action in ['source_id','reference_label']:
            shape=np.broadcast_to(np.array(json.loads(row.shapes)),(repeats,3,2)).copy()
            for n in range(129):
                if n in protocol['budgets']:
                    mu,var=moments(shape)
                    trajectory.append(pd.DataFrame(dict(state_id=row.state_id,role=row.role,action=action,n=n,replicate=np.arange(repeats),variance=var,error=(mu-truth)**2)))
                if n<128:
                    component=np.zeros(repeats,int) if action=='source_id' else 1+ss[:,n]
                    outcome=identities[:,n] if action=='source_id' else labels[:,n]
                    shape[np.arange(repeats),component,1-outcome]+=1
    tr=pd.concat(trajectory);tr.to_csv(out/'trajectories.csv.gz',index=False)
    summary=tr.groupby(['role','action','n'],as_index=False).agg(variance=('variance','mean'),mse=('error','mean'),sd=('error','std'))
    summary['mc_se']=summary.sd/np.sqrt(repeats);summary.to_csv(out/'acquisition_summary.csv',index=False)
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for i,role in enumerate(chosen.role):
        for action,g in summary[summary.role==role].groupby('action'):
            axes[0,i].plot(g.n,g.variance,'o-',label=action)
            axes[1,i].errorbar(g.n,g.mse,yerr=1.96*g.mc_se,fmt='o-',label=action)
        for ax in axes[:,i]:ax.set(xscale='symlog',xlabel='Acquired fields',title=role);ax.legend()
        axes[0,i].set_ylabel('Posterior variance');axes[1,i].set_ylabel('Held-out prevalence MSE')
    fig.tight_layout();fig.savefig(out/'acquisition_curves.png',dpi=160);plt.close(fig)
    result=dict(status='evaluated',candidates=len(states),matches=len(pairs),mi_relative_difference=float(match.mi_relative_difference),
                identity_advantage=advantages[0],label_advantage=advantages[1],confirmed=all(a>0 for a in advantages))
    (out/'result.json').write_text(json.dumps(result,indent=2))
    (out/'RESULTS.md').write_text('# Real-record counterexample\n\n'+json.dumps(result,indent=2)+'\n\n'
        'A selected near-match under a declared Beta model, not an exact equality proof. Both held-out advantages must be positive for confirmation. '
        'Target is population prevalence, not per-image prediction. Images are not conditioned on in this counterexample. '
        'Acquisitions resample genuine fields from fixed empirical populations; uncertainty bars are not patient-population confidence intervals. '
        'Neither DAPPr failure nor sufficiency of static probability bounds follows.\n')
    return result
