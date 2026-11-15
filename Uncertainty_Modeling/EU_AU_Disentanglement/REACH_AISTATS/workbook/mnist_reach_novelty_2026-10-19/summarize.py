from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def main():
 df=pd.read_csv(HERE/'reach_rows.csv');pool=pd.read_csv(HERE/'pool_assignments.csv')
 assert len(df)==1386 and not df.duplicated(['scenario','setting','seed','cell']).any()
 assert not pool.duplicated(['dataset','split','index']).any()
 for name in ['cls','joint','unk','unk_joint']:
  lo,hi=df[name+'_lower'],df[name+'_upper']
  assert np.isfinite(lo).all() and np.isfinite(hi).all()
  assert (lo>=-1e-6).all() and (hi<=1+1e-6).all() and (hi>=lo-1e-6).all()
  width='W_'+name
  assert np.allclose(df[width],hi-lo,atol=1e-6)
 # Two controls and four novelty arms; no repeated baseline conditions counted twice.
 run=df.drop_duplicates(['scenario','setting','seed'])
 assert len(run)==126
 pool.groupby(['dataset','split','role']).size().rename('rows').to_csv(HERE/'pool_summary.csv')
 on=df[df.region=='on_support']
 cols=['W_cls','W_label','W_joint','W_unk_joint','joint_covers','unk_joint_covers']
 on.groupby(['scenario','setting','seed'])[cols].mean().to_csv(HERE/'per_seed_summary.csv')
 diag=json.loads((HERE/'diagnostics.json').read_text())
 sample=pd.read_csv(HERE/'single_run.csv');first=sample.iloc[0]
 with np.load(HERE/'single_run_inputs.npz') as inp:
  rr=float(inp['reference_radius']);p=inp['true_weights']*inp['truth_conditionals'][:,0]
  truth=p[-1]/p.sum()
 assert np.isclose(truth,first.true_unk)
 baseline=on[on.setting=='baseline']
 high=on[(on.scenario=='letters_05')&(on.setting=='m_high')]
 report=f'''# Pilot results — 2026-10-19

All 126 original REACH-engine calls completed, producing 1,386 finite cell
records. Bounds, interval ordering, width identities, and pool assignments
passed checks. No warnings were captured in these calls. Class interval
statuses: {df.cls_status.value_counts().to_dict()}.

## Frozen-model checks

Known-digit CNN accuracy on held-out evaluation images: {diag['cnn_known_accuracy']:.2%}.
Entropy AUROC: letters {diag['letters']['entropy_auroc']:.3f}, KMNIST {diag['kmnist']['entropy_auroc']:.3f}.
Distance AUROC: letters {diag['letters']['distance_auroc']:.3f}, KMNIST {diag['kmnist']['distance_auroc']:.3f}.
These are ordinary OOD diagnostics, not REACH guarantees or MC-dropout results.
The LDA outlier cell receives {diag['letters']['outlier_cell_fraction']:.1%} of letters and
{diag['kmnist']['outlier_cell_fraction']:.1%} of KMNIST evaluation images. Many novel
images therefore still land in known-class cells.

## REACH outcome

At the baseline budgets, mean on-support W_joint and W_unk_joint are approximately
1 in all six scenarios. These intervals contain almost the whole probability
range. Coverage is high but precision is poor. No declared-family incompatibility
was flagged in any of the 126 runs, including KMNIST deployment. This does not
establish that the undeclared novelty was modeled correctly.

Increasing digit references from 200 to 1000 per class reduces mean W_label
from about 0.6404 to 0.4251. In the letters-5% scenario, mean W_unk_joint
changes from 1 to {high.W_unk_joint.mean():.4f}. This is only modest tightening.
Increasing log size or letter reference size alone generally leaves the joint
novelty intervals near 1 at the baseline digit-reference budget.

## Worked REACH call

Scenario: 5% EMNIST novelty; seed 0; n=1000; m=200 per digit;
r=50 per letter class (1300 letter references total).

The radius is eps_log + max(reference radii). In this call:

- eps_log = {first.eps_n:.4f}.
- Known-reference radius = {rr:.4f}.
- Total radius = {first.radius:.4f}.

The known-reference term dominates, so adding only letter references cannot
reduce the maximum reference-radius term once it is already smaller. Adding
log data reduces its own term but leaves the known-reference term in place.

At cell 0, the true novel posterior is {truth:.4f}, although global novel mass
is only 0.05. A conditional probability at a cell can differ substantially
from the global mixture weight. The true value is computed by multiplying
each source weight by its probability of that cell and normalizing.
The joint novelty interval is [{first.unk_joint_lower:.4f}, {first.unk_joint_upper:.4f}].
The width reflects lack of precision; it does not estimate the novel posterior
as 1. Inputs and all cell intervals are in single_run_inputs.npz and single_run.csv.

## Interpretation and next decision

This pilot supports neither precise novelty attribution nor reliable detection
of a mismatched novelty family at these budgets. It does show why reference
uncertainty matters even when the encoder classifies digits well. Smaller
posterior intervals require investigating acquisition bottlenecks and cell
resolution, not merely adding more deployment observations.

A next experiment could increase digit references and log budget together,
then compare to an explicitly labelled oracle-conditional diagnostic to separate
finite-reference conservatism from non-identification. That experiment has not
been run. Fashion-MNIST and medical-image extensions also remain pending.

There is only one trained encoder and three simulation seeds. Cell averages
are descriptive; cells sharing a run are not independent replications.
Declared-letter and undeclared-KMNIST coverage is measured against finite
empirical deployment-pool truth; the latter is outside the assumed family
unless its cell law happens to be representable. No population theorem is
established by this pilot.
'''
 (HERE/'RESULTS.md').write_text(report)
 plt.rcdefaults();fig,axes=plt.subplots(3,3,figsize=(13,10))
 for j,(prefix,values,title) in enumerate([('n',[250,1000,4000],'Deployment observations'),('m',[50,200,1000],'Digit references/class'),('r',[10,50,200],'Letter references/class')]):
  for i,width in enumerate(['W_cls','W_label','W_unk_joint']):
   ax=axes[i,j]
   for scenario in ['letters_05','kmnist_05','letters_20','kmnist_20']:
    means=[on[(on.scenario==scenario)&(on.setting==s)][width].mean() for s in [prefix+'_low','baseline',prefix+'_high']]
    ax.plot(values,means,'o-',label=scenario,alpha=.7)
   ax.set(xscale='log',xlabel=title,ylabel=width,ylim=(0,1.05))
 axes[0,0].legend(fontsize=8);fig.tight_layout();fig.savefig(HERE/'width_response_grid.png',dpi=140)
 manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'real/audit.py',ROOT/'models/ball.py',ROOT/'real/head.py',HERE/'run_reach.py',HERE/'prepare_data.py']}
 (HERE/'source_checksums.json').write_text(json.dumps(manifest,indent=2))
 print(report)

if __name__=='__main__':main()
