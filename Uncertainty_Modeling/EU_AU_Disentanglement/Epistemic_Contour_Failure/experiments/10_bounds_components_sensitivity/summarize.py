"""Summarize completed matched sensitivity runs, retaining paired differences."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
rows=list(csv.DictReader((OUT/'rows.csv').open()));training=list(csv.DictReader((OUT/'training.csv').open()))
assert len(rows)==120 and len(training)==400
configs=['narrow3','wide3','narrow5','wide5'];summary=[];paired=[]
for config in configs:
 for scenario in ['clean','shared_mixture']:
  for n in [1000,100000]:
   for truth in ([0] if scenario=='clean' else [0,1]):
    selected=[x for x in rows if (x['config'],x['scenario'],int(x['n']),int(x['truth']))==(config,scenario,n,truth)]
    record=dict(config=config,scenario=scenario,n=n,truth=truth,reps=len(selected))
    for metric in ['mi_nats','contour_error','member_radius']:
     v=np.array([float(x[metric]) for x in selected]);record[metric+'_mean']=v.mean();record[metric+'_min']=v.min();record[metric+'_max']=v.max()
    record['member_radius_hits']=sum(int(x['inside_member_radius']) for x in selected)
    summary.append(record)
for before,after in [('narrow3','wide3'),('narrow5','wide5'),('narrow3','narrow5'),('wide3','wide5')]:
 a={(x['dataset'],x['truth']):x for x in rows if x['config']==before}
 for x in rows:
  if x['config']!=after:continue
  y=a[(x['dataset'],x['truth'])]
  paired.append(dict(comparison=after+' minus '+before,dataset=x['dataset'],truth=x['truth'],
      error_change=float(x['contour_error'])-float(y['contour_error']),
      mi_change=float(x['mi_nats'])-float(y['mi_nats']),radius_change=float(x['member_radius'])-float(y['member_radius'])))
for name,data in [('summary.csv',summary),('paired_differences.csv',paired)]:
 with (OUT/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
checks={}
for c in configs:
 t=[x for x in training if x['config']==c]
 checks[c]=dict(members=len(t),members_near_bound=sum(int(x['near_bound'])>0 for x in t),
    members_near_bound_with_weight_ge_01=sum(int(x['near_bound_weight_ge_01'])>0 for x in t),
    gradients_above_1e_6=sum(float(x['gradient'])>1e-6 for x in t),
    max_gradient=max(float(x['gradient']) for x in t),max_continuation_l1=max(float(x['continuation_l1']) for x in t),
    max_continuation_nll_improvement=max(float(x['continuation_improvement']) for x in t))
(OUT/'diagnostic_summary.json').write_text(json.dumps(checks,indent=2)+'\n')
fig,axes=plt.subplots(2,2,figsize=(10,7))
for col,scenario in enumerate(['clean','shared_mixture']):
 for config in configs:
  s=[x for x in summary if x['config']==config and x['scenario']==scenario and x['truth']==0]
  axes[0,col].loglog([x['n'] for x in s],[x['contour_error_mean'] for x in s],'o-',label=config)
  axes[1,col].loglog([x['n'] for x in s],[x['mi_nats_mean'] for x in s],'o-',label=config)
 axes[0,col].set_title(scenario+' (left clean truth)')
 for row in [0,1]:axes[row,col].set_xlabel('Sample size');axes[row,col].legend(fontsize=8)
axes[0,0].set_ylabel('Mean clean-contour error');axes[1,0].set_ylabel('Mean cell MI (nats)')
fig.tight_layout();fig.savefig(OUT/'sensitivity.png',dpi=160)
for x in summary:
 if x['truth']==0:print(x['config'],x['scenario'],x['n'],'error',round(x['contour_error_mean'],5),'MI',x['mi_nats_mean'],'radius',round(x['member_radius_mean'],5))
print(json.dumps(checks,indent=2))
