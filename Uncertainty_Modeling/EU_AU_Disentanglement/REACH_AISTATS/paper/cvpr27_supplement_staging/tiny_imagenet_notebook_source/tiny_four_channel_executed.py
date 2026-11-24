#!/usr/bin/env python3
"""Tiny ImageNet extension of REACH finite-cell four-channel acquisition study.

This is a CONTROLLED image-derived categorical world, NOT a real-record evaluation
of four-channel acquisition. The image classifier and feature cells are frozen.
Known sources: clean, blur; candidate: noise. For the confounded setting,
A_candidate = 0.5*(A_clean+A_blur) is a DECLARED synthetic cell law, not a
claim about real noisy images. Images in model-fit and evaluation splits are disjoint.

Targets at a probe cell c:
  known_membership = w0*A0c/(w0*A0c+w1*A1c)
  candidate_membership = w2*A2c/sum_s(w_s*A_sc)
  anchored_label = (A0c*theta0c + A1c*theta1c)/(A0c+A1c)
where theta_s,c is P(frozen classifier makes a top-1 error | S=s,C=c).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import hashlib
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import linprog

SOURCES = ("clean", "blur", "noise")
ARMS = ("inputs", "references", "identities", "labels")
TARGETS = ("known_membership", "anchored_label", "candidate_membership")


def seed_for(master, *parts):
    h=hashlib.sha256((str(master)+"|"+"|".join(map(str,parts))).encode()).digest()
    return int.from_bytes(h[:8],"little")


def frequency_box(counts, delta):
    counts=np.asarray(counts,dtype=float)
    n=counts.sum()
    if not (0<delta<1): raise ValueError('delta must be in (0,1)')
    if n==0: return np.zeros_like(counts),np.ones_like(counts)
    radius=np.sqrt(np.log(2*len(counts)/delta)/(2*n))
    return np.clip(counts/n-radius,0,1),np.clip(counts/n+radius,0,1)


def joint_constraints(row_lo,row_hi,p_lo=None,p_hi=None,w_lo=None,w_hi=None):
    """Gz<=h, sum(z)=1, z>=0; z[s,c] are joint source-cell masses."""
    row_lo=np.asarray(row_lo,dtype=float);row_hi=np.asarray(row_hi,dtype=float)
    if row_lo.shape!=row_hi.shape:raise ValueError('row shape mismatch')
    S,D=row_lo.shape; N=S*D
    if np.any(row_hi<row_lo):raise ValueError('Inconsistent row bounds')
    mat=[];rhs=[]
    for s in range(S):
        row=np.zeros(N);row[s*D:(s+1)*D]=1.
        for c in range(D):
            basis=np.zeros(N);basis[s*D+c]=1
            mat.append(row_lo[s,c]*row-basis);rhs.append(0.)
            mat.append(basis-row_hi[s,c]*row);rhs.append(0.)
    if p_lo is not None:
        for c in range(D):
            col=np.zeros(N);col[c::D]=1
            mat.extend([-col,col]);rhs.extend([-float(p_lo[c]),float(p_hi[c])])
    if w_lo is not None:
        for s in range(S):
            row=np.zeros(N);row[s*D:(s+1)*D]=1
            mat.extend([-row,row]);rhs.extend([-float(w_lo[s]),float(w_hi[s])])
    return np.stack(mat), np.asarray(rhs),np.ones((1,N)),np.asarray([1.])


def feasible_model(G,h,E,f):
    return linprog(np.zeros(E.shape[1]),A_ub=G,b_ub=h,A_eq=E,b_eq=f,
                   bounds=(0,None),method='highs')


def sharp_ratio(G,h,E,f,num,den,eps=1e-9):
    """Charnes-Cooper LP; conservative [0,1] if feasible denominator may vanish."""
    check=feasible_model(G,h,E,f)
    if check.status !=0: return (np.nan,np.nan,'empty_polytope')
    dres=linprog(den,A_ub=G,b_ub=h,A_eq=E,b_eq=f,
                 bounds=(0,None),method='highs')
    if dres.status!=0:return (np.nan,np.nan,'denominator_lp_failure')
    if dres.fun <=eps:return (0.,1.,'undefined_denominator_policy')
    n=len(num)
    Gnew=np.column_stack([G,-h]);Enew=np.vstack([np.r_[E[0],-f[0]],np.r_[den,0.]])
    Fnew=np.array([0.,1.])
    r=[]
    for sign in (1.,-1.):
        opt=linprog(np.r_[sign*num,0.],A_ub=Gnew,b_ub=np.zeros(len(G)),
                    A_eq=Enew,b_eq=Fnew,bounds=(0,None),method='highs')
        if opt.status!=0: return (np.nan,np.nan,f'ratio_lp_failure_{opt.status}')
        r.append(sign*opt.fun)
    return (float(np.clip(r[0],0,1)),float(np.clip(r[1],0,1)),'ok')


def tightened_coordinate(lo,hi,c):
    """Sharp coordinate range for a categorical probability box and sum=1."""
    low=max(float(lo[c]),1.-float(sum(hi)-hi[c]))
    high=min(float(hi[c]),1.-float(sum(lo)-lo[c]))
    return max(0,low),min(1,high)


def anchored_bounds(row_lo,row_hi,label_lo,label_hi,c):
    a0lo,a0hi=tightened_coordinate(row_lo[0],row_hi[0],c)
    a1lo,a1hi=tightened_coordinate(row_lo[1],row_hi[1],c)
    if a0lo>a0hi+1e-10 or a1lo>a1hi+1e-10:return (np.nan,np.nan,'empty_reference_box')
    if a0lo+a1lo<=1e-9:return (0.,1.,'undefined_denominator_policy')
    corners=[(a,b) for a in (a0lo,a0hi) for b in (a1lo,a1hi)]
    low=min((a*label_lo[0]+b*label_lo[1])/(a+b) for a,b in corners)
    high=max((a*label_hi[0]+b*label_hi[1])/(a+b) for a,b in corners)
    return float(low),float(high),'ok'


def intervals_for_state(input_counts,ref_counts,identity_counts,label_counts,candidate_row,c,delta=0.05):
    """Same fixed-setting union allocation as original four-dataset construction.

    Boxes: input alpha/4, identity alpha/4, known references alpha/8 each,
    and known-source probe-cell labels alpha/8 each. Candidate row declared exact.
    """
    D=len(candidate_row)
    p_lo,p_hi=frequency_box(input_counts,delta/4)
    w_lo,w_hi=frequency_box(identity_counts,delta/4)
    rows=[frequency_box(ref_counts[s],delta/8) for s in range(2)]
    row_lo=np.stack([rows[0][0],rows[1][0],candidate_row])
    row_hi=np.stack([rows[0][1],rows[1][1],candidate_row])
    label_lo=np.zeros(2);label_hi=np.ones(2)
    # Label count vector [negative, positive], so positive coordinate is index 1.
    for s in range(2):
        qlo,qhi=frequency_box(label_counts[s],delta/8)
        label_lo[s],label_hi[s]=qlo[1],qhi[1]
    G,h,E,f=joint_constraints(row_lo,row_hi,p_lo,p_hi,w_lo,w_hi)
    nums=[];dens=[]
    for kind in ('known_membership','candidate_membership'):
        num=np.zeros(3*D);den=np.zeros(3*D)
        if kind=='known_membership':
            num[c]=1;den[c]=1;den[D+c]=1
        else:
            num[2*D+c]=1;den[c::D]=1
        nums.append(num);dens.append(den)
    known=sharp_ratio(G,h,E,f,nums[0],dens[0])
    candidate=sharp_ratio(G,h,E,f,nums[1],dens[1])
    anchored=anchored_bounds(row_lo,row_hi,label_lo,label_hi,c)
    return {'known_membership':known,'anchored_label':anchored,
            'candidate_membership':candidate}


def true_targets(A,theta,w,c):
    p=w@A
    x=w[0]*A[0,c]+w[1]*A[1,c]
    y=A[0,c]+A[1,c]
    if min(x,y,p[c])<=0: raise ValueError('Target conditioning probability must be positive')
    return {'known_membership':float(w[0]*A[0,c]/x),
            'candidate_membership':float(w[2]*A[2,c]/p[c]),
            'anchored_label':float((A[0,c]*theta[0,c]+A[1,c]*theta[1,c])/y)}


def build_world_from_cache(cache, cells, seed, smoothing=.02):
    from sklearn.cluster import MiniBatchKMeans
    d=np.load(cache,allow_pickle=False)
    for key in ('fit_features','fit_source','fit_error','eval_features','eval_source','eval_error'):
        if key not in d: raise KeyError(f'Cached data missing: {key}')
    sf=np.asarray(d['fit_source'],dtype=int);se=np.asarray(d['eval_source'],dtype=int)
    km=MiniBatchKMeans(n_clusters=cells,batch_size=1024,n_init=5,random_state=seed)
    km.fit(d['fit_features'][sf!=2])  # Only known-source fit records define the cell representation.
    cf=km.predict(d['fit_features']);ce=km.predict(d['eval_features'])
    ef=np.asarray(d['fit_error'],dtype=int);ee=np.asarray(d['eval_error'],dtype=int)
    count=np.zeros((3,cells),dtype=int);err=np.zeros_like(count)
    count_eval=np.zeros_like(count);err_eval=np.zeros_like(count)
    for s in range(3):
        for c in range(cells):
            m=(sf==s)&(cf==c); count[s,c]=m.sum();err[s,c]=ef[m].sum()
            m=(se==s)&(ce==c);count_eval[s,c]=m.sum();err_eval[s,c]=ee[m].sum()
    if np.any(count[:2]<0):
        raise ValueError(f'Known-source/cell support <20: {count.tolist()}. Try 2 cells or different feature seed.')
    A=(1-smoothing)*count/count.sum(axis=1,keepdims=True)+smoothing/cells
    theta=(err+.5)/(count+1.)
    Ae=np.divide(count_eval,count_eval.sum(axis=1,keepdims=True),
                 out=np.zeros_like(count_eval,dtype=float),where=count_eval.sum(axis=1,keepdims=True)>0)
    te=np.divide(err_eval,count_eval,out=np.full_like(err_eval,np.nan,dtype=float),where=count_eval>0)
    audit={
      'fit_source_cell_count':count.tolist(),'fit_source_cell_errors':err.tolist(),
      'eval_source_cell_count':count_eval.tolist(),'eval_source_cell_errors':err_eval.tolist(),
      'fit_source_laws':A.tolist(),'eval_source_laws':Ae.tolist(),
      'fit_error_rates':theta.tolist(),'eval_error_rates':te.tolist(),
      'per_source_reference_to_eval_TV':[float(np.abs(A[s]-Ae[s]).sum()/2) for s in range(3)],
      'classifier_accuracy_fit':float(1-ef.mean()),'classifier_accuracy_eval':float(1-ee.mean()),
      'centroids_fit_only':True,'centroids_known_sources_only':True,
      'eval_labels_used_to_build_world':False,
      'source_names':SOURCES,
      'source_generation':'fixed per-image corruption from clean/blur/noise, NOT recorded native domain',
      'event':'classifier top-1 error derived from ground-truth Tiny ImageNet class',
      'smoothed_source_law_implementation':f'(1-{smoothing})*empirical+{smoothing}/cells',
      'smoothed_event_rate_implementation':'Jeffreys (positive + .5)/(source-cell-count + 1)'
    }
    return A,theta,Ae,te,audit


def run_experiment(cache,output,cells=4,repeats=6,budgets=(0,100,1000,10000,100000),
                   w=(.45,.25,.30),seed=20261029,delta=.05,
                   baseline_inputs=500,baseline_references=200,baseline_labels=100):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    budgets=tuple(sorted(set(int(x) for x in budgets)))
    if budgets[0]!=0 or any(x<0 for x in budgets):raise ValueError('budgets must include 0 and be >=0')
    w=np.asarray(w,dtype=float);w=w/w.sum()
    if len(w)!=3 or (w<=0).any():raise ValueError('3 positive mixture weights required')
    A,theta,Aeval,thetaeval,audit=build_world_from_cache(cache,cells,seed)
    c=int(np.argmax((A[0]+A[1])/2)) # same ex-ante probe as original controlled experiment
    worlds={'image_derived':A.copy(),'confounded':np.vstack([A[:2],(A[0]+A[1])/2])}
    frames=[];bench=[]
    for regime,AA in worlds.items():
        p=w@AA
        true=true_targets(AA,theta,w,c)
        B=np.column_stack([AA[0]-AA[2],AA[1]-AA[2]])
        sv=np.linalg.svd(B,compute_uv=False)
        # Exact-input benchmark; exact rows and exact cell distribution, unknown w.
        G,h,E,f=joint_constraints(AA,AA,p,p,None,None)
        d=AA.shape[1]
        for tar in ('known_membership','candidate_membership'):
            num=np.zeros(3*d);den=np.zeros(3*d)
            if tar=='known_membership':num[c]=1;den[c]=1;den[d+c]=1
            else:num[2*d+c]=1;den[c::d]=1
            l,u,st=sharp_ratio(G,h,E,f,num,den)
            if st!='ok':warnings.warn(f'Exact benchmark {regime}/{tar}: {st}')
            bench.append({'regime':regime,'target':tar,'lower':l,'upper':u,'width':u-l,
                          'status':st,'min_source_geometry_singular_value':float(sv[-1])})
        for rep in range(repeats):
            rng=np.random.default_rng(seed_for(seed,regime,rep,'baseline'))
            base={'inputs':rng.multinomial(baseline_inputs,p),
                  'identities':rng.multinomial(0,w),
                  'references':np.stack([rng.multinomial(baseline_references,AA[s]) for s in range(2)]),
                  'labels':np.stack([np.array([baseline_labels-k,k]) for k in
                            [rng.binomial(baseline_labels,theta[s,c]) for s in range(2)]])}
            for arm in ARMS:
                state={k:v.copy() for k,v in base.items()}
                rg=np.random.default_rng(seed_for(seed,regime,rep,arm,'stream'))
                prev=0
                for b in budgets:
                    n=b-prev
                    if n>0:
                        if arm=='inputs': state['inputs']+=rg.multinomial(n,p)
                        elif arm=='identities':state['identities']+=rg.multinomial(n,w)
                        elif arm=='references':
                            for s in range(2):state['references'][s]+=rg.multinomial(n,AA[s])
                        elif arm=='labels':
                            for s in range(2):
                                k=rg.binomial(n,theta[s,c]);state['labels'][s]+=np.array([n-k,k])
                    prev=b
                    result=intervals_for_state(state['inputs'],state['references'],
                         state['identities'],state['labels'],AA[2],c,delta)
                    for target,(lo,hi,status) in result.items():
                        frames.append({'regime':regime,'repeat':rep,'arm':arm,'budget':b,
                            'actual_extra_fields':2*b if arm in ('references','labels') else b,
                            'target':target,'lower':lo,'upper':hi,'width':hi-lo,
                            'truth':true[target], 'contains_truth': bool(lo-1e-8<=true[target]<=hi+1e-8)
                            if np.isfinite(lo) and np.isfinite(hi) else False,
                            'status':status,'probe_cell':c,'n_cells':cells})
    df=pd.DataFrame(frames)
    df.to_csv(output/'intervals.csv',index=False)
    benchmarks=pd.DataFrame(bench);benchmarks.to_csv(output/'exact_identification_benchmarks.csv',index=False)
    summary=(df.groupby(['regime','target','arm','budget'],as_index=False)
             .agg(mean_width=('width','mean'),min_width=('width','min'),max_width=('width','max'),
                  containment_fraction=('contains_truth','mean'),n=('repeat','count')))
    summary.to_csv(output/'summary.csv',index=False)
    # Descriptive held-out source-law / error-rate audit. The constructed confounded
    # population intentionally has no corresponding unmodified real-image eval pool.
    held=[]
    for s in range(3):
        for k in range(cells):
            held.append({'source':SOURCES[s],'cell':k,'fit_cell_prob':float(A[s,k]),
               'eval_cell_prob':float(Aeval[s,k]),'fit_error_rate':float(theta[s,k]),
               'eval_error_rate':float(thetaeval[s,k])})
    pd.DataFrame(held).to_csv(output/'reference_eval_transfer.csv',index=False)
    audit.update({'generating_mixture_weights':w.tolist(),'cells':cells,'probe_cell':c,
        'repeats':repeats,'budgets':list(budgets),'alpha_fixed_setting':delta,
        'confidence_allocation':'inputs delta/4, identities delta/4, 2 references delta/8 each, 2 labels delta/8 each',
        'observations':'synthetic nested multinomial/Bernoulli draws WITH replacement from fixed image-derived finite laws',
        'candidate_law':'Specified EXACT in the compatibility calculation; confounded A2=(A0+A1)/2 by construction',
        'baseline':{'inputs':baseline_inputs,'references_per_known_source':baseline_references,
            'labels_per_known_source_at_probe':baseline_labels,'identities':0},
        'confidence_coverage':'per fixed setting only, not across all budgets/repeats',
        'positive_denominator_cutoff':1e-9,
        'selection':'probe chosen from known-source fit laws, before sampling; never on held-out',
        'empty_polytope_n':int((df.status=='empty_polytope').sum()),
        'zero_denominator_n':int((df.status=='undefined_denominator_policy').sum()),
        'synthetic_world_target_inclusion_n':int(df.contains_truth.sum()),
        'total_target_intervals':len(df),
        'no_real_image_paired_acquisition_test':True})
    (output/'audit.json').write_text(json.dumps(audit,indent=2,allow_nan=True))
    fig,axes=plt.subplots(2,3,figsize=(15,7.3),sharex=True,sharey=True)
    colors={'inputs':'#2563eb','references':'#0f766e','identities':'#c2410c','labels':'#9333ea'}
    for row,regime in enumerate(worlds):
        for col,target in enumerate(TARGETS):
            ax=axes[row,col]
            for arm in ARMS:
                ss=summary[(summary.regime==regime)&(summary.target==target)&(summary.arm==arm)].sort_values('budget')
                xs=ss.budget.to_numpy();m=ss.mean_width.to_numpy()
                ax.plot(xs,m,marker='o',ms=3.7,color=colors[arm],label=arm.replace('_',' '))
                ax.fill_between(xs,ss.min_width,ss.max_width,color=colors[arm],alpha=.10)
            ss=benchmarks[(benchmarks.regime==regime)&(benchmarks.target==target)]
            if not ss.empty and np.isfinite(ss.iloc[0]['width']):
                ax.axhline(ss.iloc[0]['width'],color='black',linestyle='--',lw=1,
                           label='exact input-only ID width' if row==0 and col==0 else None)
            ax.set_title(f'{regime.replace("_"," ")} | {target.replace("_"," ")}')
            ax.set_xscale('symlog',linthresh=100)
            ax.set_ylim(-.025,1.025);ax.grid(alpha=.2)
            if col==0:ax.set_ylabel('Compatible interval width')
            if row==1:ax.set_xlabel('Added budget b (ref/label = 2b fields)')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=5,loc='upper center',bbox_to_anchor=(.5,1.02),frameon=False)
    fig.suptitle('Tiny ImageNet — controlled four-channel acquisition',y=1.075,fontsize=14)
    fig.tight_layout()
    fig.savefig(output/'four_channel_acquisition.png',dpi=180,bbox_inches='tight')
    fig.savefig(output/'four_channel_acquisition.pdf',bbox_inches='tight')
    plt.close(fig)
    print('Saved four-channel Tiny ImageNet outputs:',output)
    print('Probe:',c,'| intervals:',len(df),'| inclusion:',int(df.contains_truth.sum()),'/',len(df))
    print('Exact input-only floors:')
    print(benchmarks[['regime','target','width','min_source_geometry_singular_value']].to_string(index=False))
    return summary,df


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',required=True)
    p.add_argument('--output',default='runs/tiny_four_channel')
    p.add_argument('--cells',type=int,default=4)
    p.add_argument('--repeats',type=int,default=6)
    p.add_argument('--budgets',default='0,100,1000,10000,100000')
    p.add_argument('--weights',default='.45,.25,.30')
    p.add_argument('--seed',type=int,default=20261029)
    p.add_argument('--alpha',type=float,default=.05)
    p.add_argument('--baseline-inputs',type=int,default=500)
    p.add_argument('--baseline-references',type=int,default=200)
    p.add_argument('--baseline-labels',type=int,default=100)
    args=p.parse_args()
    run_experiment(args.cache,args.output,cells=args.cells,repeats=args.repeats,
         budgets=tuple(int(b) for b in args.budgets.split(',')),
         w=tuple(float(x) for x in args.weights.split(',')),seed=args.seed,
         delta=args.alpha,baseline_inputs=args.baseline_inputs,
         baseline_references=args.baseline_references,baseline_labels=args.baseline_labels)


if __name__=='__main__':main()