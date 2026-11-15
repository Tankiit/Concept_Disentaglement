"""Frozen visual representation, matched CE/DAPPr heads, and acquisition bounds.

This is a declared pilot, not an official WILDS or original DAPPr reproduction.
"""
from pathlib import Path
import copy
import hashlib
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from real.wilds_records import load_images
from real.source_acquisition import frequency_box, membership_bounds, polytope, ratio_interval
from real.probability_bounds import DropoutEnvelopeRepresenter, envelope_from_samples, evaluate_probability_ood, ProbabilityBounds


def extract_features(records, weights, output, device='cuda:0', batch_size=128):
    from torchvision.models import resnet18, ResNet18_Weights
    weights=Path(weights)
    if not weights.exists():raise FileNotFoundError(f'ImageNet ResNet18 weights required: {weights}')
    model=resnet18(weights=None)
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True))
    model.fc=nn.Identity();model.eval().to(device)
    transform=ResNet18_Weights.IMAGENET1K_V1.transforms()
    result=np.empty((len(records),512),np.float32);batch=[];positions=[]
    def flush():
        with torch.inference_mode():
            result[positions]=model(torch.stack(batch).to(device)).cpu().numpy()
        batch.clear();positions.clear()
    for i,im in load_images(records):
        batch.append(transform(im));positions.append(i)
        if len(batch)>=batch_size:flush()
    if batch:flush()
    np.savez_compressed(Path(output)/'features.npz',features=result,record_ids=records.record_id.to_numpy().astype(str))
    (Path(output)/'encoder.json').write_text(json.dumps(dict(architecture='ResNet18',pretraining='ImageNet1K V1',
        frozen=True,weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),device=device),indent=2))
    return result


def dappr_loss(logits, labels, coefficient):
    alpha=nn.functional.softplus(logits)+1
    y=nn.functional.one_hot(labels,logits.shape[-1]).to(alpha.dtype)
    adjusted=alpha-y+1e-8
    p=(adjusted/adjusted.sum(-1,keepdim=True)).detach()
    mass=alpha.sum(-1)
    fit=mass*mass.log()+(alpha*(p/alpha).log()).sum(-1)
    return fit.mean()+coefficient*(alpha*(1-y)).square().sum(-1).mean()


def fit_predict(records, features, output, seed=20261030, epochs=30, device='cuda:0', mc_passes=30):
    from probly.method import dropout
    from probly.representer.sampler import Sampler
    from probly.representation.distribution.torch_categorical import TorchLogitCategoricalDistribution
    from probly.representation.sample import create_sample
    from probly.quantification import quantify
    out=Path(output);torch.manual_seed(seed)
    fit=(records.role=='fit') & (records.source<2) & ~records.temporal_ood
    evaluation=(records.role=='evaluation') & ~records.temporal_ood
    # Fixed epochs avoid spending reference labels on model selection.
    k=int(records.target.max())+1
    x=torch.tensor(features,device=device);y=torch.tensor(records.target.to_numpy(),device=device)
    mean=x[fit.to_numpy()].mean(0);std=x[fit.to_numpy()].std(0).clamp_min(.01)
    x=(x-mean)/std
    torch.save(dict(mean=mean.cpu(),std=std.cpu()),out/'feature_scaler.pt')
    initial=dropout(nn.Sequential(nn.Linear(512,128),nn.ReLU(),nn.Linear(128,k)),p=.25,predictor_type='logit_classifier')
    history=[];scores=[];models={};predictions={}
    ix=np.flatnonzero(fit);ev=np.flatnonzero(evaluation)
    for method in ['ce','dappr']:
        torch.manual_seed(seed);rng=np.random.default_rng(seed)
        model=copy.deepcopy(initial).to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        for epoch in range(epochs):
            model.train();total=0
            for batch in np.array_split(rng.permutation(ix),max(1,int(np.ceil(len(ix)/128)))):
                opt.zero_grad();logits=model(x[batch])
                loss=nn.functional.cross_entropy(logits,y[batch]) if method=='ce' else dappr_loss(logits,y[batch],2e-4*min(1,(epoch+1)/10))
                loss.backward();opt.step();total+=float(loss.detach())*len(batch)
            history.append(dict(method=method,epoch=epoch+1,loss=total/len(ix)))
        model.eval()
        with torch.no_grad():predictions[method]=torch.cat([model(b).cpu() for b in x.split(512)])
        torch.save(model.state_dict(),out/f'{method}_head.pt');models[method]=model
    pd.DataFrame(history).to_csv(out/'training_history.csv',index=False)
    def sample_factory(logits,sample_axis=-1):
        return create_sample([TorchLogitCategoricalDistribution(tensor=z) for z in logits],sample_axis=sample_axis)
    representer=DropoutEnvelopeRepresenter(Sampler(models['ce'],num_samples=mc_passes,sample_factory=sample_factory))
    torch.manual_seed(seed+100)
    draws=torch.cat([representer.represent(batch).samples.cpu() for batch in x[ev].split(256)])
    rep=envelope_from_samples(draws)
    entropy=lambda p:-(p*p.clamp_min(1e-30).log()).sum(-1)
    mi=entropy(draws.mean(1))-entropy(draws).mean(1)
    alpha=nn.functional.softplus(predictions['dappr'][ev])+1
    score_table=records.iloc[ev][['record_id','source','target','group']].copy()
    score_table['mc_mi']=mi.numpy();score_table['mc_width']=quantify(rep).epistemic.numpy()
    score_table['dappr_eu']=(k/alpha.sum(-1)).numpy()
    score_table['negative_max_softmax']=-predictions['ce'][ev].softmax(-1).max(-1).values.numpy()
    score_table.to_csv(out/'image_scores.csv',index=False)
    np.savez_compressed(out/'mc_representation.npz',samples=draws.numpy(),lower=rep.lower.numpy(),upper=rep.upper.numpy(),
                        record_ids=score_table.record_id.to_numpy().astype(str))
    for name in ['mc_mi','mc_width','dappr_eu','negative_max_softmax']:
        scores.append(dict(method=name,**evaluate_probability_ood(score_table.loc[score_table.source<2,name],score_table.loc[score_table.source==2,name])))
    pd.DataFrame(scores).to_csv(out/'ood_scores.csv',index=False)
    pd.DataFrame([dict(method=m,source=s,n=int(sum(records.iloc[ev].source==s)),accuracy=float((p[ev].argmax(-1).numpy()[records.iloc[ev].source==s]==records.iloc[ev].target.to_numpy()[records.iloc[ev].source==s]).mean()))
                  for m,p in predictions.items() for s in [0,1,2]]).to_csv(out/'accuracy.csv',index=False)
    return score_table


def label_bounds(rowlo,rowhi,thetalo,thetahi,cell):
    """Source-and-cell conditional label rates; no Y independent of X assumption."""
    p=polytope(rowlo[:2],rowhi[:2],weights=(np.array([.5,.5]),np.array([.5,.5])))
    den=np.zeros_like(rowlo[:2]);den[:,cell]=1
    a=ratio_interval(p,den*thetalo,den);b=ratio_interval(p,den*thetahi,den)
    if a['status']=='empty' or b['status']=='empty':return dict(lower=np.nan,upper=np.nan,width=np.nan,status='empty',residual=np.nan)
    return dict(lower=a['lower'],upper=b['upper'],width=b['upper']-a['lower'],status=a['status'],residual=max(a['residual'],b['residual']))


def acquisition_bounds(records,features,output,seed=20261030,n_cells=4,repeats=5,budgets=(0,32,128,512,2048)):
    from sklearn.cluster import KMeans
    out=Path(output)
    fit=(records.role=='fit') & (records.source<2) & ~records.temporal_ood
    km=KMeans(n_clusters=n_cells,n_init=10,random_state=seed).fit(features[fit])
    cells=km.predict(features);np.save(out/'cell_centers.npy',km.cluster_centers_)
    ref=records[(records.role=='reference') & ~records.temporal_ood].copy()
    ev=records[(records.role=='evaluation') & ~records.temporal_ood].copy()
    ref['cell']=cells[ref.index];ev['cell']=cells[ev.index]
    event=1 if records.target.nunique()==2 else int(records.loc[fit,'target'].value_counts().sort_index().idxmax())
    ref['binary_target']=(ref.target==event).astype(int)
    ref.to_csv(out/'reference_cells.csv',index=False);ev.to_csv(out/'evaluation_cells.csv',index=False)
    # Union allocation across all constructed boxes in this run. Conditional
    # empirical resampling, not an iid-patient population guarantee.
    alpha=.05/(repeats*len(budgets)*4*(3+2*n_cells+2))
    rows=[];audit=[];representations=[]
    for repeat in range(repeats):
        rng=np.random.default_rng(seed+repeat)
        bmax=max(budgets)
        source_streams=[rng.choice(ref.index[ref.source==s],128+bmax) for s in range(3)]
        input_stream=rng.choice(ev.index,128+bmax);id_stream=rng.choice(ev.index,32+bmax)
        label_streams={}
        for s in [0,1]:
            for c in range(n_cells):
                pool=ref.index[(ref.source==s)&(ref.cell==c)]
                label_streams[s,c]=rng.choice(pool,16+bmax) if len(pool) else np.array([],int)
        audit.append(dict(repeat=repeat,reference=[[str(records.loc[i,'record_id']) for i in a] for a in source_streams],
                          inputs=records.loc[input_stream,'record_id'].tolist(),identities=records.loc[id_stream,'record_id'].tolist(),
                          labels={f'{s}:{c}':records.loc[ix,'record_id'].tolist() for (s,c),ix in label_streams.items()}))
        for action in ['inputs','source_id','reference_inputs','target_labels']:
            for budget in budgets:
                nref=128+(budget//3 if action=='reference_inputs' else 0)
                lo=[];hi=[]
                for stream in source_streams:
                    a,b=frequency_box(np.bincount(cells[stream[:nref]],minlength=n_cells),alpha);lo.append(a);hi.append(b)
                lo,hi=np.array(lo),np.array(hi)
                nin=128+(budget if action=='inputs' else 0)
                marginal=frequency_box(np.bincount(cells[input_stream[:nin]],minlength=n_cells),alpha)
                nid=32+(budget if action=='source_id' else 0)
                weights=frequency_box(np.bincount(records.loc[id_stream[:nid],'source'],minlength=3),alpha)
                tl=np.zeros((2,n_cells));th=np.ones((2,n_cells))
                nl=16+(budget//(2*n_cells) if action=='target_labels' else 0)
                for (s,c),stream in label_streams.items():
                    outcomes=ref.loc[stream[:nl],'binary_target'].to_numpy()
                    l,h=frequency_box(np.bincount(outcomes,minlength=2),alpha);tl[s,c]=l[1];th[s,c]=h[1]
                for c in range(n_cells):
                    intervals=membership_bounds(lo,hi,marginal,weights,c)
                    intervals['reference_label']=label_bounds(lo,hi,tl,th,c)
                    for target,v in intervals.items():
                        rows.append(dict(repeat=repeat,action=action,budget=budget,cell=c,target=target,**v))
                if repeat==0 and action=='inputs' and budget==0:
                    # Same Probly representation interface, with source-event semantics.
                    from real.probability_bounds import SourceMixtureBoundsRepresenter
                    wrapper=SourceMixtureBoundsRepresenter(None,lo,hi,marginal,weights)
                    rep=wrapper.represent_cells(ev.cell.to_numpy())
                    np.savez_compressed(out/'source_representation.npz',lower=rep.lower.numpy(),upper=rep.upper.numpy(),
                        candidate_lower=rep.candidate_lower.numpy(),candidate_upper=rep.candidate_upper.numpy(),record_ids=ev.record_id.to_numpy().astype(str))
    result=pd.DataFrame(rows);result.to_csv(out/'acquisition_intervals.csv',index=False)
    (out/'acquisition_record_ids.json').write_text(json.dumps(audit))
    summary=result.groupby(['action','budget','target'],as_index=False).agg(mean_width=('width','mean'),
        empty_fraction=('status',lambda s:float((s=='empty').mean())),undefined_fraction=('status',lambda s:float((s=='undefined_denominator_policy').mean())))
    summary.to_csv(out/'acquisition_summary.csv',index=False)
    # Empirical cell laws are a model-check diagnostic, never assumed invariant.
    law=[]
    for role,f in [('reference',ref),('evaluation',ev)]:
        for s in range(3):
            p=np.bincount(f.loc[f.source==s,'cell'],minlength=n_cells)/sum(f.source==s)
            law.extend(dict(role=role,source=s,cell=c,probability=float(p[c])) for c in range(n_cells))
    pd.DataFrame(law).to_csv(out/'source_cell_law_audit.csv',index=False)
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,(target,f) in zip(axes,summary.groupby('target')):
        for action,g in f.groupby('action'):ax.plot(g.budget,g.mean_width,'o-',label=action)
        ax.set(title=target,xscale='symlog',xlabel='Additional field budget',ylabel='Mean probability width',ylim=(0,1.02));ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'acquisition_bounds.png',dpi=160);plt.close(fig)
    (out/'bounds_definition.json').write_text(json.dumps(dict(event_class=event,n_cells=n_cells,alpha_per_box=alpha,
        candidate='specified third source with estimated reference cell law, not arbitrary unknown OOD',
        source_events='P(S=0 | C, S in {0,1}) and P(S=2 | C)',
        label_event='P(Y=event | C) under equal known-source reference weights and cell-conditional rates',
        guarantee='conditional empirical-resampling box coverage; transfer requires stable source cell laws',
        budgets='inputs/IDs: n fields; reference inputs: 3*floor(n/3); labels: 2*C*floor(n/(2*C)); fixed initial information',
        missing_cell='unobserved conditional label rate uses [0,1]'),indent=2))
    return summary
