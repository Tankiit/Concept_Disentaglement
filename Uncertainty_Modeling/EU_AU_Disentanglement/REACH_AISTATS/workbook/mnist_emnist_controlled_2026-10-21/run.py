"""Paired all-cell diagnosis of the saved MNIST/EMNIST pilot, seed 0."""
from pathlib import Path
import sys
import json
import hashlib
import numpy as np
import pandas as pd
from scipy.optimize import linprog
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = HERE.parent / 'mnist_reach_novelty_2026-10-19'
sys.path[:0] = [str(ROOT), str(HERE.parent / 'reach_interval_diagnostic_2026-10-19/followup')]
from investigate import bounds
from models.ball import MMDBall, embedding_radius


def coherent(lo, hi, plo, phi, cell, source):
    """Joint-mass LP; source=None targets novelty, otherwise known class."""
    k, d = lo.shape
    size = k*d
    rows = []
    for s in range(k):
        total = np.zeros(size); total[s*d:(s+1)*d] = 1
        for c in range(d):
            unit = np.zeros(size); unit[s*d+c] = 1
            rows.extend([np.r_[unit-hi[s,c]*total, 0.],
                         np.r_[lo[s,c]*total-unit, 0.]])
    for c in range(d):
        col = np.zeros(size); col[c::d] = 1
        rows.extend([np.r_[col, -phi[c]], np.r_[-col, plo[c]]])
    a = np.array(rows)
    den = np.zeros(size)
    den[cell::d] = 1
    if source is not None:
        den[(k-1)*d+cell] = 0
    num = np.zeros(size)
    num[(k-1 if source is None else source)*d+cell] = 1
    mass = linprog(den, A_ub=a[:,:-1], b_ub=-a[:,-1],
                   A_eq=np.ones((1,size)), b_eq=[1.], bounds=(0,None))
    if mass.status == 2:
        return np.nan, np.nan, 'empty_set'
    assert mass.success, mass.message
    if mass.fun <= 1e-10:
        return 0., 1., 'zero_mass_policy'
    eq = np.stack([np.r_[np.ones(size), -1.], np.r_[den, 0.]])
    values = []
    for sign in [1., -1.]:
        cost = np.r_[num, 0.]*sign
        fit = linprog(cost, A_ub=a, b_ub=np.zeros(len(a)), A_eq=eq,
                      b_eq=[0.,1.], bounds=(0,None))
        assert fit.success, fit.message
        assert np.max(a@fit.x) < 1e-6
        assert np.max(np.abs(eq@fit.x-[0.,1.])) < 1e-6
        values.append(float(np.clip(sign*fit.fun, 0, 1)))
    return *values, 'optimal'


def original(ball, lo, hi, qr, cell, source):
    col = ball.conditionals[:,cell]
    if source is None:
        qlo, qhi = max(col[-1]-qr,0), min(col[-1]+qr,1)
        low = np.zeros(len(col)); low[-1] = qlo
        high = np.zeros(len(col)); high[-1] = qhi
        l = ball.ratio_interval(low,low,np.r_[hi[:,cell],qlo]) if qlo else {'lower':0.}
        h = ball.ratio_interval(high,high,np.r_[lo[:,cell],qhi]) if qhi else {'upper':0.}
    else:
        others = np.arange(len(col)-1) != source
        dl = np.r_[np.where(others, hi[:,cell],lo[source,cell]),0.]
        dh = np.r_[np.where(others, lo[:,cell],hi[source,cell]),0.]
        low = np.zeros(len(col)); low[source] = lo[source,cell]
        high = np.zeros(len(col)); high[source] = hi[source,cell]
        l = ball.ratio_interval(low,low,dl)
        h = ball.ratio_interval(high,high,dh)
    return l['lower'], h['upper'], l.get('status','entry_endpoint')+' / '+h.get('status','entry_endpoint')


def main():
    z = np.load(BASE/'single_run_inputs.npz')
    a = np.vstack([z['known_conditionals'],z['novel_conditional']])
    truth, weights, log = z['truth_conditionals'], z['true_weights'], z['log_cells']
    marginal = weights@truth
    empirical = np.bincount(log,minlength=11)/len(log)
    saved = pd.read_csv(BASE/'single_run.csv').set_index('cell')
    counts = np.rint(a*np.r_[np.full(10,200),1300][:,None]).astype(int)
    np.testing.assert_allclose(counts/counts.sum(1)[:,None],a,atol=1e-12)
    kl, kh = bounds(counts[:-1],.0125)
    # Letter references were balanced across 26 strata, not iid draws from
    # the pooled letter distribution. Hoeffding allows independent bounded
    # non-identically distributed draws; pooled binomial CP is inappropriate.
    qeps = np.sqrt(np.log(2*11/.0125)/(2*1300))
    ql, qh = np.clip(a[-1]-qeps,0,1), np.clip(a[-1]+qeps,0,1)
    lo, hi = np.vstack([kl,ql]), np.vstack([kh,qh])
    pl, ph = bounds(np.bincount(log,minlength=11),.025)
    qr = embedding_radius(1300,.05/33)
    ref, entry = float(z['reference_radius']),float(z['entry_radius'])
    ball = MMDBall(a,empirical,embedding_radius(len(log),.05/3),max(ref,qr),'l2')
    oracle = MMDBall(truth,empirical,embedding_radius(len(log),.05/3),0.,'l2')
    records = []
    for cell in range(11):
        yhat = int(saved.loc[cell,'yhat'])
        for target, source in [('novelty',None),('known_class',yhat)]:
            jt = weights*truth[:,cell]
            value = jt[-1]/jt.sum() if source is None else jt[source]/jt[:-1].sum()
            cases = {
                'original': original(ball,np.clip(a[:-1]-entry,0,1),np.clip(a[:-1]+entry,0,1),qr,cell,source),
                'exact_references': original(oracle,truth[:-1],truth[:-1],0.,cell,source),
                'exact_references_and_log': coherent(truth,truth,marginal,marginal,cell,source),
                'revised_joint_counts': coherent(lo,hi,pl,ph,cell,source),
                'exact_references_CP_log': coherent(truth,truth,pl,ph,cell,source),
            }
            prefix = 'unk_joint' if source is None else 'joint'
            np.testing.assert_allclose(cases['original'][:2],saved.loc[cell,[prefix+'_lower',prefix+'_upper']].to_numpy(float),atol=2e-5)
            for case,(lower,upper,status) in cases.items():
                records.append(dict(cell=cell,region='known_cells' if cell<10 else 'outlier',target=target,
                                    case=case,lower=lower,upper=upper,width=upper-lower,truth=value,
                                    covers=lower-1e-6<=value<=upper+1e-6,status=status))
        print('Completed cell',cell,flush=True)
    df = pd.DataFrame(records)
    assert np.isfinite(df[['lower','upper']]).all().all()
    assert (df.width >= -1e-6).all()
    exact = df[df.case=='exact_references_and_log']
    assert exact.covers.all()
    df.to_csv(HERE/'intervals.csv',index=False)
    summary = df.groupby(['target','case','region']).agg(width=('width','mean'),
                  coverage=('covers','mean'),min_width=('width','min'),max_width=('width','max'))
    summary.to_csv(HERE/'summary.csv')
    config = dict(input_sha256=hashlib.sha256((BASE/'single_run_inputs.npz').read_bytes()).hexdigest(),
                  n=1000,m=200,r_per_letter=50,delta=.05,seed=0,alpha=.05,
                  reference_matrix_rank=int(np.linalg.matrix_rank(truth)),
                  reference_matrix_condition_number=float(np.linalg.cond(truth)),
                  singular_values=np.linalg.svd(truth,compute_uv=False).tolist(),
                  original_endpoints_reproduced=True,
                  revised_bounds='CP known references and deployment; stratification-compatible Hoeffding letter bounds; alpha split .0125/.025/.0125',
                  scope='One saved finite-pool run; all 11 cells. Oracle inputs are diagnostic, not deployable. No population coverage claim.')
    (HERE/'config.json').write_text(json.dumps(config,indent=2))
    order = ['original','exact_references','exact_references_and_log','revised_joint_counts','exact_references_CP_log']
    labels = ['Original','Exact references\n(original log ball)','Exact references\n+ exact log','Revised joint counts','Exact references\n(CP log bounds)']
    fig,axes = plt.subplots(2,1,figsize=(13,8),layout='constrained')
    for ax,target in zip(axes,['novelty','known_class']):
        values = np.array([[df.query('target == @target and case == @case and cell == @cell').width.iloc[0]
                            for cell in range(11)] for case in order])
        im = ax.imshow(values,vmin=0,vmax=1,cmap='YlOrRd',aspect='auto')
        ax.set(yticks=range(5),yticklabels=labels,xticks=range(11),
               xticklabels=[str(c) for c in range(10)]+['Outlier'],title=target.replace('_',' ').title()+' probability: interval width',xlabel='Fixed original partition cell')
        for (i,j),v in np.ndenumerate(values):
            ax.text(j,i,f'{v:.3f}',ha='center',va='center',color='white' if v>.78 else 'black')
    fig.colorbar(im,ax=axes,label='Width (1 = full probability range)',shrink=.8)
    fig.suptitle('MNIST + 5% EMNIST: same encoder, partition, references and deployment draw\nSeed 0; n = 1000; 200 references/digit; 1300 letter references',fontsize=14)
    fig.savefig(HERE/'controlled_widths.png',dpi=180)
    fig.savefig(HERE/'controlled_widths.pdf')
    print(summary.to_string())


if __name__ == '__main__':
    main()
