"""Paired oracle diagnostics using saved evidence only; no acquisition claims."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from run import HERE, ALPHA, bounds, ratio, solver, H

MODES = ['baseline', 'exact_sources', 'exact_weights', 'exact_sources_weights',
         'exact_log', 'exact_sources_log', 'fixed_setting_alpha']

def main():
    cfg = json.loads((HERE/'config.json').read_text())
    A, w = np.array(cfg['laws']), np.array(cfg['weights'])
    theta = np.array(cfg['theta'])
    evidence = json.loads((HERE/'evidence.json').read_text())
    baseline = pd.read_csv(HERE/'intervals.csv')
    records = []
    for item in evidence:
        for mode in MODES:
            alpha = .05/8 if mode == 'fixed_setting_alpha' else ALPHA
            rc, ac = np.array(item['reference_counts']), np.array(item['audit_counts'])
            lo, hi = bounds(rc, alpha)
            al, ah = bounds(ac, alpha)
            lo, hi = np.maximum(lo, al), np.minimum(hi, ah)
            wl, wh = bounds(ac.sum(1), alpha)
            jl, jh = bounds(ac.ravel(), alpha)
            jl, jh = jl.reshape(11,11), jh.reshape(11,11)
            pl, ph = bounds(np.array(item['log_counts']), alpha)
            bl, bh = bounds(np.array(item['anchor_counts']).ravel(), alpha)
            bl, bh = bl.reshape(2,11), bh.reshape(2,11)
            if mode in ['exact_sources', 'exact_sources_weights', 'exact_sources_log']:
                lo, hi = A.copy(), A.copy()
            if mode in ['exact_weights', 'exact_sources_weights']:
                wl, wh = w.copy(), w.copy()
            if mode in ['exact_log', 'exact_sources_log']:
                pl, ph = w@A, w@A
            for cell in range(11):
                src = cell if cell < 10 else 0
                for target, source in [('W_cls', src), ('W_unk', None), ('W_label', None)]:
                    diagnostic = {}
                    if target == 'W_label':
                        lower, upper, status = H.label_interval(lo[:10], hi[:10], theta, theta, cell)
                        dl, du, _ = ratio(bl[:,cell], bh[:,cell], [False,True])
                        lower, upper = max(lower,dl), min(upper,du)
                        # Exact minimum anchor cell mass over normalized independent row boxes.
                        row_min = np.maximum(lo[:10,cell], 1-hi[:10].sum(1)+hi[:10,cell])
                        diagnostic['minimum_mass'] = row_min.mean()
                        truth = theta@A[:10,cell]/A[:10,cell].sum()
                    else:
                        lower, upper, status = solver.interval(lo,hi,pl,ph,wl,wh,cell,source,jl,jh,diagnostic)
                        joint = w*A[:,cell]
                        truth = joint[src]/joint[:10].sum() if target == 'W_cls' else joint[-1]/joint.sum()
                    records.append(dict(arm=item['arm'],budget=item['budget'],mode=mode,
                                        target=target,cell=cell,lower=lower,upper=upper,
                                        width=max(0.,upper-lower),status=status,truth=truth,
                                        covers=lower-1e-6<=truth<=upper+1e-6,**diagnostic))
        print(item['arm'],item['budget'],'diagnosed',flush=True)
    df = pd.DataFrame(records)
    df.to_csv(HERE/'diagnostic_intervals.csv',index=False)
    keys = ['arm','budget','target','cell']
    paired = df[df['mode']=='baseline'].merge(baseline[baseline.method=='reach_direct_law'],on=keys,suffixes=('_new','_old'))
    assert len(paired)==528
    assert np.allclose(paired[['lower_new','upper_new']],paired[['lower_old','upper_old']],atol=1e-7)
    assert (paired.status_new==paired.status_old).all()
    reference = df[df['mode']=='baseline'][keys+['lower','upper']]
    for mode in MODES[1:]:
        pair = df[df['mode']==mode].merge(reference,on=keys,suffixes=('_mode','_base'))
        assert (pair.lower_mode >= pair.lower_base-1e-6).all()
        assert (pair.upper_mode <= pair.upper_base+1e-6).all()
    exact = df[df['mode']=='exact_sources_weights']
    assert (exact.width.abs()<1e-6).all()
    assert (df[df['mode']=='exact_sources_log'].width.abs()<1e-6).all()
    assert (df.minimum_mass >= -1e-7).all()
    assert df.covers.all() and not (df.status=='empty_set').any()
    summary = df.groupby(['arm','budget','mode','target']).agg(
        width=('width','mean'),coverage=('covers','mean'),
        zero_mass=('status',lambda v:int((v=='zero_mass_policy').sum())),
        minimum_mass=('minimum_mass','min'))
    summary.to_csv(HERE/'diagnostic_summary.csv')
    fig, axes = plt.subplots(1,3,figsize=(14,5),layout='constrained',sharey=True)
    for ax,target in zip(axes,['W_cls','W_label','W_unk']):
        for budget in [0,256]:
            s=summary.xs(('audit',budget,target),level=['arm','budget','target']).reindex(MODES)
            ax.plot(range(len(MODES)),s.width,'o-',label=f'{budget} audits')
        ax.set(title=target,xticks=range(len(MODES)),xticklabels=[m.replace('_','\n') for m in MODES],ylim=(-.02,1.02))
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Mean width over eleven cells');axes[-1].legend()
    fig.suptitle('Saved smoke evidence: oracle controls and confidence-accounting diagnostic\nFixed-setting alpha does not retain the simultaneous-across-settings guarantee')
    fig.savefig(HERE/'diagnostic_widths.png',dpi=160)
    validation = dict(records=len(df),baseline_records_reproduced=len(paired),
                      controls_nested_in_baseline=True,exact_sources_weights_point_identified=True,
                      exact_sources_log_point_identified=True,
                      source_law_rank=int(np.linalg.matrix_rank(A)),
                      smallest_singular_value=float(np.linalg.svd(A,compute_uv=False)[-1]),
                      all_recorded_truths_covered=True,rejected_records=0,
                      warning='One split; oracle controls are not deployable baselines; alpha control changes guarantee scope')
    (HERE/'diagnostic_validation.json').write_text(json.dumps(validation,indent=2))
    print(summary.xs(('audit',256),level=['arm','budget']).to_string())

if __name__ == '__main__':
    main()
