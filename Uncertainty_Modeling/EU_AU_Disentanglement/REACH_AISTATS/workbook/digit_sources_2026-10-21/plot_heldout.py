from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
df=pd.read_csv(HERE/'intervals.csv')
df=df[df.world=='heldout_6_9']
modes=['exact','estimated_200','estimated_1000','unrestricted_1000']
labels=['Exact candidate + known laws','Estimated: 200 / 1300 refs','Estimated: 1000 / 6500 refs','No candidate law; 1000 known refs']
fig,axes=plt.subplots(2,2,figsize=(13,7),layout='constrained')
for i,delta in enumerate([0.,.2]):
    for j,target in enumerate(['W_cls','W_unk']):
        ax=axes[i,j]
        values=np.array([[df[(df.delta==delta)&(df.target==target)&(df['mode']==mode)&(df.cell==cell)].width.mean() for cell in range(7)] for mode in modes])
        im=ax.imshow(values,vmin=0,vmax=1,cmap='YlOrRd',aspect='auto')
        ax.set(yticks=range(4),yticklabels=labels,xticks=range(7),xticklabels=['0','1','2','3','4','5','Outlier'],
               title=f'{target}: {delta:.0%} held-out digits',xlabel='Cell')
        for (r,c),v in np.ndenumerate(values):
            ax.text(c,r,f'{v:.2f}',ha='center',va='center',color='white' if v>.78 else 'black',fontsize=9)
fig.colorbar(im,ax=axes,label='Mean width across 20 seeds',shrink=.7)
fig.suptitle('Trained on 0–5; held-out digits 6–9\nSame finite deployment log; candidate information changes',fontsize=14)
fig.savefig(HERE/'heldout_membership.png',dpi=180)
fig.savefig(HERE/'heldout_membership.pdf')
pred=pd.read_csv(HERE/'heldout_predictions.csv')
pred['correct']=pred.prediction==pred.label
pred['outlier']=pred.cell==6
pred['confident']=pred.confidence>.9
pred.groupby('label').agg(images=('label','size'),accuracy=('correct','mean'),outlier_fraction=('outlier','mean'),mean_confidence=('confidence','mean'),confident_fraction=('confident','mean')).to_csv(HERE/'per_digit_classifier.csv')
