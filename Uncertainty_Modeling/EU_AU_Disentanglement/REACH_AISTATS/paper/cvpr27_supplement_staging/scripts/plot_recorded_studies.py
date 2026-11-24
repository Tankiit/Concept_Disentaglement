"""Redraw paper figures from bundled archived CSVs; no training or new sampling.
Run: python scripts/plot_recorded_studies.py
Dependencies: matplotlib, numpy. Outputs: vector PDF and 220-dpi PNG.
"""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/cvpr2027-matplotlib')
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter, NullLocator

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / 'figure_data', ROOT / 'figures'
OUT.mkdir(exist_ok=True)
for entry in json.loads((DATA/'source_manifest.json').read_text()):
    assert hashlib.sha256((DATA/entry['file']).read_bytes()).hexdigest() == entry['sha256'], entry['file']

plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':8,
    'axes.titlesize':8.5, 'axes.labelsize':8, 'xtick.labelsize':7,
    'ytick.labelsize':7, 'legend.fontsize':8, 'axes.linewidth':0.6,
    'lines.linewidth':1.5, 'lines.markersize':3.2, 'pdf.fonttype':42,
    'ps.fonttype':42, 'savefig.dpi':220, 'axes.spines.top':False,
    'axes.spines.right':False, 'axes.axisbelow':True})
COLORS = {'inputs':'#0072B2', 'references':'#009E73',
          'identities':'#D55E00', 'labels':'#8B59A3'}
STYLES = {'inputs':('o','-'), 'references':('s','--'),
          'identities':('^','-'), 'labels':('D',':')}
LABELS = {'inputs':'Deployment inputs', 'references':'Reference inputs',
          'identities':'Source identities', 'labels':'Target labels'}
DATASETS=[('mnist','MNIST'),('emnist','EMNIST'),('fashion','Fashion-MNIST'),('blood','BloodMNIST')]
TARGETS=[('candidate_membership','Candidate membership'),('known_membership','Known membership'),('anchored_target','Anchored label')]

def read(name):
    with (DATA/name).open() as f: return list(csv.DictReader(f))

def select(rows, **kwargs):
    return [r for r in rows if all(r[k]==v for k,v in kwargs.items())]

def values(rows, key): return np.array([float(r[key]) for r in rows])

def legend(fig, kinds, y=1, floor=False):
    handles=[Line2D([],[],color=COLORS[k], marker=STYLES[k][0],
        linestyle=STYLES[k][1],label=LABELS[k]) for k in kinds]
    if floor: handles.append(Line2D([],[],color='#424242',ls='--',lw=1.2,label='Exact-input identification width'))
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,y),
        ncol=len(handles),frameon=False,columnspacing=1.3,handlelength=2)

def axis(ax, mode='controlled'):
    ax.grid(axis='y',color='#dce1e5',lw=.5)
    threshold={'controlled':100,'pilot':32,'trajectory':1}[mode]
    ticks={'controlled':[0,100,1000,10000,100000], 'pilot':[0,32,128,512,2048],
           'trajectory':[0,1,4,16,128]}[mode]
    labels={'controlled':['0',r'$10^2$',r'$10^3$',r'$10^4$',r'$10^5$'],
            'pilot':['0','32','128','512','2048'],
            'trajectory':['0','1','4','16','128']}[mode]
    ax.set_xscale('symlog',linthresh=threshold)
    ax.set_xticks(ticks,labels)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(length=2.5,pad=2)
    ax.margins(x=.035)
    if mode != 'trajectory':
        ax.set_ylim(-.025,1.04)
        ax.set_yticks([0,.25,.5,.75,1])

def line(ax, x, y, kind, **kw):
    marker,ls=STYLES[kind]
    return ax.plot(x,y,color=COLORS[kind],marker=marker,ls=ls,**kw)

def save(fig,name):
    # Preserve the designed full-width dimensions instead of cropping each figure differently.
    fig.savefig(OUT/(name+'.pdf'),metadata={'Creator':'Matplotlib', 'CreationDate':None})
    fig.savefig(OUT/(name+'.png'))
    plt.close(fig)

summary=read('controlled_summary.csv'); oracle=read('controlled_oracle.csv')
# Figure 2: same-scale panels retain both regimes; only matched-cost channels shown.
fig,axs=plt.subplots(2,4,figsize=(7,3.65),sharex=True,sharey=True)
fig.subplots_adjust(left=.09,right=.985,bottom=.15,top=.81,wspace=.12,hspace=.26)
legend(fig,['inputs','identities'],y=.995,floor=True)
for row,regime in enumerate(['image_derived','confounded']):
    for col,(ds,title) in enumerate(DATASETS):
        ax=axs[row,col];axis(ax)
        for arm,kind in [('inputs','inputs'),('source_ids','identities')]:
            rs=sorted(select(summary,dataset=ds,regime=regime,target='candidate_membership',arm=arm),key=lambda r:int(r['budget']))
            assert len(rs)==5
            x=values(rs,'budget');line(ax,x,values(rs,'mean_width'),kind)
            ax.fill_between(x,values(rs,'min_width'),values(rs,'max_width'),color=COLORS[kind],alpha=.12,lw=0)
        r=select(oracle,dataset=ds,regime=regime,target='candidate_membership',information='exact_inputs')
        assert len(r)==1
        ax.axhline(float(r[0]['width']),ls='--',color='#424242',lw=1.2,zorder=1)
        if row==0:ax.set_title(title,pad=6,weight='bold')
        if col==0:ax.set_ylabel(('Image-derived\n' if row==0 else 'Confounded\n')+'interval width')
fig.supxlabel('Added observations (inputs or identities)',y=.025,fontsize=8.5)
save(fig,'fig2_controlled_identification')

# Supplement: all three targets and all four arms, preserving per-source budgets.
for regime in ['image_derived','confounded']:
    fig,axs=plt.subplots(3,4,figsize=(7,5),sharex=True,sharey=True)
    fig.subplots_adjust(left=.1,right=.985,bottom=.12,top=.85,wspace=.12,hspace=.16)
    legend(fig,['inputs','references','identities','labels'],y=.99)
    for row,(target,label) in enumerate(TARGETS):
        for col,(ds,title) in enumerate(DATASETS):
            ax=axs[row,col];axis(ax)
            for arm,kind in [('inputs','inputs'),('references','references'),('source_ids','identities'),('target_labels','labels')]:
                rs=sorted(select(summary,dataset=ds,regime=regime,target=target,arm=arm),key=lambda r:int(r['budget']))
                assert len(rs)==5
                x=values(rs,'budget');line(ax,x,values(rs,'mean_width'),kind)
                ax.fill_between(x,values(rs,'min_width'),values(rs,'max_width'),color=COLORS[kind],alpha=.07,lw=0)
            if row==0:ax.set_title(title,weight='bold')
            if col==0:ax.set_ylabel(label+'\ninterval width')
    fig.supxlabel('Added budget b (references and labels: b per known source)',y=.025,fontsize=8.5)
    save(fig,'supp_controlled_'+regime)

# Figure 3: actual pilot field totals; no synthetic confidence bands.
fig,axs=plt.subplots(2,3,figsize=(7,3.65),sharex=True,sharey=True)
fig.subplots_adjust(left=.095,right=.985,bottom=.15,top=.81,wspace=.15,hspace=.24)
legend(fig,['inputs','references','identities','labels'],y=.995)
for row,(ds,title) in enumerate([('camelyon17','CAMELYON17'),('fmow','FMoW')]):
    rows=read(ds+'_pilot_summary.csv')
    for col,(target,label) in enumerate([('candidate_membership','Candidate membership'),('known_membership','Known membership'),('reference_label','Anchored label')]):
        ax=axs[row,col];axis(ax,'pilot')
        for arm,kind in [('inputs','inputs'),('reference_inputs','references'),('source_id','identities'),('target_labels','labels')]:
            rs=sorted(select(rows,target=target,action=arm),key=lambda r:int(r['budget']))
            assert len(rs)==5
            line(ax,values(rs,'budget'),values(rs,'mean_width'),kind)
        if row==0:ax.set_title(label,weight='bold')
        if col==0:ax.set_ylabel(title+'\ninterval width')
fig.supxlabel('Added field budget (allocated total)',y=.025,fontsize=8.5)
save(fig,'fig3_visual_targets')

# Figure 4: original (unnormalized) variance/MSE, per-panel scientific units.
fig,axs=plt.subplots(2,4,figsize=(7,4.15),sharex=True)
fig.subplots_adjust(left=.09,right=.985,bottom=.14,top=.73,wspace=.46,hspace=.38)
legend(fig,['identities','labels'],y=.995)
fig.text(.30,.875,'CAMELYON17: ordering confirmed',ha='center',weight='bold',fontsize=9)
fig.text(.765,.875,'FMoW: confirmation fails',ha='center',weight='bold',fontsize=9)
for di,ds in enumerate(['camelyon17','fmow']):
    rows=read(ds+'_counter_summary.csv');states=read(ds+'_states.csv')
    for ri,role in enumerate(['identity_preferred','label_preferred']):
        col=2*di+ri
        state=select(states,role=role)[0]
        axs[0,col].set_title('State '+state['state_id']+'\nModel prefers '+('identities' if ri==0 else 'labels'),fontsize=8,pad=20)
        for row,metric in enumerate(['variance','mse']):
            ax=axs[row,col];axis(ax,'trajectory')
            for arm,kind in [('source_id','identities'),('reference_label','labels')]:
                rs=sorted(select(rows,role=role,action=arm),key=lambda r:int(r['n']))
                assert len(rs)==9
                x,y=values(rs,'n'),values(rs,metric)
                line(ax,x,y,kind)
                if metric=='mse':
                    ax.errorbar(x,y,yerr=1.96*values(rs,'mc_se'),fmt='none',ecolor=COLORS[kind],elinewidth=.8,capsize=1.5)
            formatter=ScalarFormatter(useMathText=True);formatter.set_powerlimits((0,0));formatter.set_useOffset(False)
            ax.yaxis.set_major_formatter(formatter)
            ax.tick_params(axis='x',labelsize=6.5)
            ax.locator_params(axis='y',nbins=4)
            if col==0:ax.set_ylabel('Posterior variance' if row==0 else 'Held-out MSE')
fig.supxlabel('Added fields (one channel per trajectory)',y=.025,fontsize=8.5)
save(fig,'fig4_heldout_acquisition')
