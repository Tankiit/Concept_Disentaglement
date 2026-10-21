"""Show actual endpoints, not only widths, for preselected seed zero."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
df=pd.read_csv(HERE/'intervals.csv')
fig,axes=plt.subplots(1,3,figsize=(14,5),layout='constrained',sharex=True,sharey=True)
for ax,declared in zip(axes,['fashion','blood','emnist']):
    sub=df[(df.seed==0)&(df.deployed=='fashion')&(df.declared==declared)&(df['mode']=='estimated')&(df.identities==0)&(df.target=='W_unk')&(df.cell<10)].sort_values('cell')
    for row in sub.itertuples():
        color='#2166ac' if row.covers else '#b2182b'
        ax.plot([row.lower,row.upper],[row.cell,row.cell],color=color,linewidth=4)
    ax.scatter(sub.truth,sub.cell,color='black',marker='x',label='Actual Fashion membership',zorder=3)
    ax.set(title=f'Declared candidate: {declared}',xlabel='Candidate membership probability',ylabel='Known-class cell',yticks=range(10),xlim=(-.02,1.02))
    ax.grid(alpha=.2)
axes[0].legend(fontsize=8)
fig.suptitle('Fashion deployed; finite references; no identities; seed 0\nBlue intervals cover; red intervals miss; black crosses show truth')
fig.savefig(HERE/'mismatch_endpoints.png',dpi=180)
fig.savefig(HERE/'mismatch_endpoints.pdf')
