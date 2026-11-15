"""Paired iid finite-pool acquisitions; frozen source laws and pilot weights."""
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import beta
import matplotlib.pyplot as plt
from run import HERE, BASE, coherent, original, bounds, MMDBall, embedding_radius

OUT = HERE / 'repeated'
BUDGETS = [(1000,200,1300), (4000,1000,5200)]
SEEDS = 100


def main():
    OUT.mkdir(exist_ok=True)
    path = BASE/'single_run_inputs.npz'
    z = np.load(path)
    truth, weights = z['truth_conditionals'], z['true_weights']
    saved = pd.read_csv(BASE/'single_run.csv').set_index('cell')
    marginal = weights@truth
    records, draws = [], []
    for seed in range(SEEDS):
        rng = np.random.default_rng(np.random.SeedSequence([20261021,seed]))
        refs = [rng.choice(11,size=1000,p=row) for row in truth[:-1]]
        novel = rng.choice(11,size=5200,p=truth[-1])
        log = rng.choice(11,size=4000,p=marginal)
        for n,m,q in BUDGETS:
            kc = np.stack([np.bincount(x[:m],minlength=11) for x in refs])
            qc = np.bincount(novel[:q],minlength=11)
            lc = np.bincount(log[:n],minlength=11)
            draws.append(dict(seed=seed,n=n,m=m,q=q,known_counts=kc.tolist(),novel_counts=qc.tolist(),log_counts=lc.tolist()))
            a = np.vstack([kc/m,qc/q])
            kl,kh = bounds(kc,.0125)
            ql,qh = bounds(qc,.0125)
            pl,ph = bounds(lc,.025)
            lo,hi = np.vstack([kl,ql]),np.vstack([kh,qh])
            qr = embedding_radius(q,.05/33)
            rr = embedding_radius(m,.05/33)
            entry = np.sqrt(np.log(2/(.05/(3*10*11)))/(2*m))
            ball = MMDBall(a,lc/n,embedding_radius(n,.05/3),max(rr,qr),'l2')
            for cell in range(11):
                yhat = int(saved.loc[cell,'yhat'])
                jt = weights*truth[:,cell]
                for target,source in [('novelty',None),('known_class',yhat)]:
                    value = jt[-1]/jt.sum() if source is None else jt[source]/jt[:-1].sum()
                    cases = {
                        'original': original(ball,np.clip(a[:-1]-entry,0,1),np.clip(a[:-1]+entry,0,1),qr,cell,source),
                        'revised': coherent(lo,hi,pl,ph,cell,source),
                    }
                    for method,(lower,upper,status) in cases.items():
                        records.append(dict(seed=seed,n=n,m=m,q=q,cell=cell,
                            region='known_cells' if cell<10 else 'outlier',target=target,method=method,
                            lower=lower,upper=upper,width=upper-lower,truth=value,status=status,
                            covers=bool(lower-1e-6<=value<=upper+1e-6),
                            empty=status=='empty_set',zero_mass='zero_mass_policy' in status))
        print('Finished seed',seed,flush=True)
        pd.DataFrame(records).to_csv(OUT/'intervals.csv',index=False)
    df = pd.DataFrame(records)
    assert len(df)==SEEDS*2*11*2*2
    finite = df[~df['empty']]
    assert np.isfinite(finite[['lower','upper']]).all().all()
    assert (finite.width>=-1e-6).all() and (finite.width<=1+1e-6).all()
    (OUT/'draws.json').write_text(json.dumps(draws))
    (OUT/'config.json').write_text(json.dumps(dict(seeds=list(range(SEEDS)),budgets=BUDGETS,
        delta=.05,weights=weights.tolist(),input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        alpha=.05,alpha_split=dict(log=.025,known=.0125,novel=.0125),
        scope='iid draws from saved deployment-pool source laws, fixed mixture and partition; nested budgets and paired methods; no train/test shift; pooled iid novelty, not stratified letter acquisition'),indent=2))
    keys = ['n','m','q','method','target']
    per_cell = df.groupby(keys+['cell']).agg(mean_width=('width','mean'),
        median_width=('width','median'),min_width=('width','min'),max_width=('width','max'),
        coverage=('covers','mean'),empty_rate=('empty','mean'),zero_mass_rate=('zero_mass','mean'),
        near_vacuous_rate=('width',lambda v: (v>=.99).mean()))
    per_cell.to_csv(OUT/'per_cell_summary.csv')
    per_seed = df.groupby(keys+['region','seed']).agg(mean_width=('width','mean'),
        all_cover=('covers','all'),empty_rate=('empty','mean'),zero_mass_rate=('zero_mass','mean'))
    per_seed.to_csv(OUT/'per_seed_summary.csv')
    summary = per_seed.groupby(level=keys+['region']).agg(mean_width=('mean_width','mean'),
        min_seed_width=('mean_width','min'),max_seed_width=('mean_width','max'),
        simultaneous_coverage=('all_cover','mean'),empty_rate=('empty_rate','mean'),zero_mass_rate=('zero_mass_rate','mean'))
    summary.to_csv(OUT/'summary.csv')
    joint = df.groupby(keys+['seed']).covers.all().groupby(level=keys).agg(['sum','count'])
    joint['coverage'] = joint['sum']/joint['count']
    joint['CP95_lower'] = [0 if s==0 else beta.ppf(.025,s,n-s+1) for s,n in zip(joint['sum'],joint['count'])]
    joint['CP95_upper'] = [1 if s==n else beta.ppf(.975,s+1,n-s) for s,n in zip(joint['sum'],joint['count'])]
    joint.to_csv(OUT/'simultaneous_coverage.csv')
    paired = df.pivot(index=['seed','n','m','q','cell','target'],columns='method',values='width')
    paired['revised_minus_original'] = paired.revised-paired.original
    paired.to_csv(OUT/'paired_width_differences.csv')
    fig,axes = plt.subplots(2,2,figsize=(14,8),layout='constrained',sharey=True)
    for i,target in enumerate(['novelty','known_class']):
        for j,(n,m,q) in enumerate(BUDGETS):
            ax = axes[i,j]
            for method,color in [('original','#b2182b'),('revised','#2166ac')]:
                sub = df[(df.target==target)&(df.n==n)&(df.method==method)]
                grouped = sub.groupby('cell').width
                mean = grouped.mean()
                ax.plot(mean.index,mean.values,'o-',label=method,color=color)
                ax.fill_between(mean.index,grouped.min(),grouped.max(),color=color,alpha=.14)
            ax.set(title=f'{target.replace("_"," ").title()} • n={n}, m={m}, q={q}',
                   xticks=range(11),xticklabels=[str(c) for c in range(10)]+['Outlier'],
                   ylim=(-.025,1.025),xlabel='Cell',ylabel='Interval width')
            ax.grid(alpha=.2)
    axes[0,0].legend()
    fig.suptitle(f'MNIST + 5% EMNIST • {SEEDS} paired finite-pool acquisitions\nLines: mean; bands: seed min–max, not confidence intervals',fontsize=14)
    fig.savefig(OUT/'repeated_widths.png',dpi=180)
    fig.savefig(OUT/'repeated_widths.pdf')
    print(summary.to_string())
    print(joint.to_string())


if __name__=='__main__':
    main()
