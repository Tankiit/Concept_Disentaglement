from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
tv=pd.read_csv(HERE/'intervals.csv')
box=pd.read_csv(HERE.parent/'candidate_sensitivity_2026-10-22/intervals.csv')
keys=['seed','case','identities','delta','cell','target']
m=tv.merge(box,on=keys,suffixes=('_tv','_box'),validate='one_to_one')
v=m[(m.status_tv!='empty_set')&(m.status_box!='empty_set')].copy()
v['width_reduction']=v.width_box-v.width_tv
groups=['case','identities','delta','target']
v.groupby(groups).agg(mean_reduction=('width_reduction','mean'),max_reduction=('width_reduction','max')).to_csv(HERE/'width_comparison.csv')
out=dict(paired_records=len(m),paired_nonempty=len(v),max_width_reduction=float(v.width_reduction.max()),
    mean_width_reduction=float(v.width_reduction.mean()),different_rejection_records=int(((m.status_tv=='empty_set')!=(m.status_box=='empty_set')).sum()),
    different_coverage_records=int((m.covers_tv!=m.covers_box).sum()),
    intervals_reduced_by_more_than_001=int((v.width_reduction>.01).sum()))
(HERE/'comparison.json').write_text(json.dumps(out,indent=2));print(out)
fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
for identities,color in [(0,'#2166ac'),(8192,'#b2182b')]:
    s=v[(v.target=='W_unk')&(v.cell<10)&(v.identities==identities)]
    axes[0].scatter(s.width_box,s.width_tv,s=9,alpha=.3,color=color,label=f'{identities} identities')
    g=s.groupby('delta').width_reduction.mean()
    axes[1].plot(range(len(g)),g,'o-',color=color,label=f'{identities} identities')
axes[0].plot([0,1],[0,1],'k:',label='Equal widths')
axes[0].set(xlabel='Coordinate-box W_unk',ylabel='TV W_unk',title='Paired nonempty known-cell intervals')
axes[0].legend(fontsize=8)
axes[1].set(xticks=range(7),xticklabels=[0,.01,.05,.1,.25,.5,1],xlabel='Sensitivity budget (discrete grid)',ylabel='Mean box width minus TV width',title='Average reduction across cases and seeds')
fig.savefig(HERE/'comparison.png',dpi=180);fig.savefig(HERE/'comparison.pdf')
