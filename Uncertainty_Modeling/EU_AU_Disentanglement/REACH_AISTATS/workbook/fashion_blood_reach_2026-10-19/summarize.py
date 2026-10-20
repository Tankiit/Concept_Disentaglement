from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
BASE=ROOT/'workbook/mnist_reach_novelty_2026-10-19'

def main():
 d=pd.read_csv(HERE/'reach_rows.csv');pool=pd.read_csv(HERE/'pool_assignments.csv')
 assert len(d)==924 and not d.duplicated(['scenario','setting','seed','cell']).any()
 assert not pool.duplicated(['dataset','split','index']).any()
 for prefix in ['cls','joint','unk','unk_joint']:
  lo,hi=d[prefix+'_lower'],d[prefix+'_upper'];good=np.isfinite(lo)&np.isfinite(hi)
  assert ((lo[good]>=-1e-6)&(hi[good]<=1+1e-6)&(lo[good]<=hi[good]+1e-6)).all()
  assert np.allclose(d.loc[good,'W_'+prefix],hi[good]-lo[good],atol=1e-6)
 runs=d.drop_duplicates(['scenario','setting','seed']);assert len(runs)==84
 cols=['W_cls','W_label','W_joint','W_unk','W_unk_joint','joint_covers','unk_joint_covers','incompatible']
 d.groupby(['scenario','setting','region'])[cols].mean().to_csv(HERE/'summary.csv')
 d.groupby(['scenario','setting','seed','region'])[cols].mean().to_csv(HERE/'per_seed_summary.csv')
 d.groupby(['scenario','setting','cls_status']).size().rename('rows').to_csv(HERE/'solver_status.csv')
 d.groupby(['scenario','setting','region'])[['W_joint','W_unk_joint']].count().to_csv(HERE/'finite_counts.csv')
 pool.groupby(['dataset','role']).size().rename('rows').to_csv(HERE/'pool_summary.csv')
 diag=json.loads((HERE/'diagnostics.json').read_text());old=json.loads((BASE/'diagnostics.json').read_text())
 comparisons=[]
 for name,source,key in [('EMNIST Letters',old,'letters'),('KMNIST',old,'kmnist'),('Fashion-MNIST',diag,'fashion'),('BloodMNIST grayscale',diag,'blood')]:
  item=source[key];comparisons.append(dict(dataset=name,entropy_auroc=item['entropy_auroc'],distance_auroc=item['distance_auroc'],outlier_fraction=item.get('outlier_fraction',item.get('outlier_cell_fraction'))))
 pd.DataFrame(comparisons).to_csv(HERE/'domain_comparison.csv',index=False)
 plt.rcdefaults()
 fig,axes=plt.subplots(1,2,figsize=(11,4))
 ev=pd.read_csv(HERE/'evaluation_scores.csv')
 for name in ['mnist','fashion','blood']:
  v=ev[ev.dataset==name]
  axes[0].hist(v.entropy,bins=np.linspace(0,np.log(10),31),density=True,histtype='step',label=name)
  axes[1].hist(np.log10(v.distance.clip(lower=1e-8)),bins=35,density=True,histtype='step',label=name)
 axes[0].set(xlabel='Predictive entropy (nats)',ylabel='Density');axes[1].set(xlabel='log10 nearest-class Mahalanobis distance',ylabel='Density')
 for ax in axes:ax.legend()
 fig.tight_layout();fig.savefig(HERE/'domain_scores.png',dpi=150);plt.close(fig)
 fig,axes=plt.subplots(3,8,figsize=(11,4.5))
 for row,name in enumerate(['mnist','fashion','blood']):
  with np.load(ROOT/f'cache/mnist_reach/{name}_test.npz') as a:
   for c in range(8):
    i=np.flatnonzero(a['y']==c)[0];axes[row,c].imshow(a['X'][i],cmap='gray',vmin=0,vmax=255);axes[row,c].axis('off');axes[row,c].set_title(f'{name}: {c}',fontsize=8)
 fig.tight_layout();fig.savefig(HERE/'input_examples.png',dpi=140);plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(13,4))
 on=d[d.region=='on_support']
 for ax,(prefix,values,title) in zip(axes,[('n',[250,1000,4000],'Deployment observations'),('m',[50,200,1000],'Digit references/class'),('r',[10,50,200],'Letter references/class')]):
  for scenario in sorted(d.scenario.unique()):
   means=[on[(on.scenario==scenario)&(on.setting==s)].W_unk_joint.mean() for s in [prefix+'_low','baseline',prefix+'_high']]
   ax.plot(values,means,'o-',alpha=.7,label=scenario)
  ax.set(xscale='log',xlabel=title,ylabel='Mean joint novelty width',ylim=(0,1.05))
 axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(HERE/'acquisition_response.png',dpi=150);plt.close(fig)
 table='\n'.join(f"| {r['dataset']} | {r['entropy_auroc']:.3f} | {r['distance_auroc']:.5f} | {r['outlier_fraction']:.1%} |" for r in comparisons)
 report=f'''# Results — Fashion-MNIST and BloodMNIST

Completed 84 REACH calls (924 cell records), with the previous frozen model,
partition and references. MNIST cell assignments and entropy reproduce the
previous run; known-digit accuracy remains {diag['known_accuracy']:.2%}.
Input pool IDs are disjoint within each dataset. Finite bounds, ordering and
width identities passed checks. Class status counts: {d.cls_status.value_counts().to_dict()}.
Captured warnings: {len(json.loads((HERE/'warnings.json').read_text()))}.
Missing joint novelty bounds: {int(d.W_unk_joint.isna().sum())}.

## OOD diagnostics (separate from REACH)

| Novel domain | Entropy AUROC | Distance AUROC | Outlier cell fraction |
| --- | ---: | ---: | ---: |
{table}

The first two rows are prior-run results with the same frozen model. Distance
is the nearest-class Mahalanobis distance used by the fixed LDA partition.
Blood images are almost perfectly ranked above digits by this score. This
can reflect background/texture as well as content; it is not evidence of
medical understanding. Fashion entropy separates well, while most fashion
images still fall below the fixed outlier threshold. AUROC measures ranking,
so good AUROC need not imply high detection at that particular threshold.

## REACH result

At the baseline (n=1000, m=200, r=50), mean joint novelty width is approximately
1 in all four new scenarios. The outlier cell also has width 1. No declared
mixture incompatibility is flagged in any of the {len(runs)} runs.

With m increased to 1000, mean known-cell joint novelty widths at 5% novelty
are about 0.9792 (Fashion) and 0.9829 (Blood). At 20% novelty they remain about
1. W_label decreases from about 0.6404 to 0.4251, as in the previous pilot.
Increasing n or r alone does not materially resolve the broad joint intervals.

Thus changing to more visually distinct datasets did not solve the precision
problem at these budgets. Strong feature-distance OOD ranking coexists with
uninformative REACH bounds. High observed interval coverage is largely due
to wide intervals and does not validate the mismatched EMNIST novelty model.

These widths describe cell probabilities, not per-image medical diagnoses.
The source references are still EMNIST, so Fashion and Blood are deliberately
undeclared novelty. Obtaining matching references would be a different
information-access condition and has not been tested here. Likewise an RGB
medical condition, intensity-matched controls, more informative reference
budgets, and oracle-conditional diagnostics remain untested.

Only three simulation seeds and one trained encoder are used. No claim of
population coverage or general OOD superiority is made from this pilot.
'''
 (HERE/'RESULTS.md').write_text(report)
 manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'run_extension.py',HERE/'prepare_data.py',ROOT/'real/audit.py',ROOT/'real/head.py',ROOT/'models/ball.py']}
 (HERE/'source_checksums.json').write_text(json.dumps(manifest,indent=2))
 print(report)

if __name__=='__main__':main()
