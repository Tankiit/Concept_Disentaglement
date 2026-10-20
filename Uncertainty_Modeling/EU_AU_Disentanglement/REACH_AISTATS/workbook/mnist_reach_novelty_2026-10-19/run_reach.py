"""Exploratory frozen MNIST encoder with original real.audit REACH engine."""
from pathlib import Path
import sys, json, hashlib, warnings
import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from real.head import LDAHead
from real.audit import run_cell, _group_hist, hull_distance
import real.audit as audit
CACHE=ROOT/'cache/mnist_reach'

class DigitNet(nn.Module):
 def __init__(self):
  super().__init__()
  self.conv=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Flatten())
  self.emb=nn.Sequential(nn.Linear(32*7*7,64),nn.ReLU())
  self.output=nn.Linear(64,10)
 def features(self,x): return self.emb(self.conv(x))
 def forward(self,x): return self.output(self.features(x))

def data(name,split):
 with np.load(CACHE/f'{name}_{split}.npz') as d: return d['X'],d['y']

def tensor(X,device): return torch.from_numpy(np.ascontiguousarray(X)).to(device=device,dtype=torch.float32)[:,None]/255.

def main():
 torch.set_num_threads(4);torch.manual_seed(0)
 device='cuda' if torch.cuda.is_available() else 'cpu'
 config=dict(date='2026-10-19',scope='exploratory pilot, not historical reproduction',
  seeds=[0,1,2],n=[250,1000,4000],m=[50,200,1000],novel_ref_per_class=[10,50,200],
  baseline=dict(n=1000,m=200,r=50),delta=[0,.05,.2],epochs=5,
  cnn_train=30000,partition_dev=10000,mnist_reference=20000,
  cell_head='original PCA-16/LDA, 99th percentile distance outlier cell',
  input='28x28 grayscale /255; EMNIST orientation corrected by transpose',
  n_boot=20,bootstrap_members=5,eta=0.,device=device)
 (HERE/'config.json').write_text(json.dumps(config,indent=2))
 X,y=data('mnist','train');rng=np.random.default_rng(0);perm=rng.permutation(len(y))
 fit,dev,ref=perm[:30000],perm[30000:40000],perm[40000:]
 assert not set(fit)&set(dev) and not set(fit)&set(ref) and not set(dev)&set(ref)
 assignments=[dict(dataset='mnist',split='train',index=int(i),role=role,label=int(y[i])) for role,ids in [('cnn_train',fit),('partition_dev',dev),('reference',ref)] for i in ids]
 net=DigitNet().to(device);optimizer=torch.optim.Adam(net.parameters(),lr=.001)
 loss_rows=[]
 for epoch in range(5):
  net.train();order=fit[rng.permutation(len(fit))];total=0
  for start in range(0,len(order),256):
   ix=order[start:start+256];optimizer.zero_grad()
   loss=nn.functional.cross_entropy(net(tensor(X[ix],device)),torch.from_numpy(y[ix]).to(device))
   loss.backward();optimizer.step();total+=loss.item()*len(ix)
  loss_rows.append(dict(epoch=epoch+1,loss=total/len(fit)));print(loss_rows[-1],flush=True)
 pd.DataFrame(loss_rows).to_csv(HERE/'training.csv',index=False)
 net.eval();net.requires_grad_(False);torch.save(net.cpu().state_dict(),HERE/'digit_net.pt');net.to(device)
 def encode(images):
  with torch.no_grad(): return np.concatenate([net.features(tensor(images[s:s+256],device)).cpu().numpy() for s in range(0,len(images),256)])
 Fdev=encode(X[dev]);head=LDAHead.fit(Fdev,y[dev],np.arange(10),dim=16)
 np.savez(HERE/'partition.npz',**head.__dict__)
 known_ref=head.cell(encode(X[ref]));ref_pools={ (c,):[known_ref[y[ref]==c]] for c in range(10)}
 LX,Ly=data('letters','train');LX=LX.transpose(0,2,1)
 letter_ids=np.concatenate([rng.permutation(np.flatnonzero(Ly==c))[:200] for c in range(26)])
 letter_cells=head.cell(encode(LX[letter_ids]));letter_ref=[letter_cells[Ly[letter_ids]==c] for c in range(26)]
 assignments += [dict(dataset='letters',split='train',index=int(i),role='novel_reference',label=int(Ly[i])) for i in letter_ids]
 pools={};eval_rows=[]
 for name in ['mnist','letters','kmnist']:
  xx,yy=data(name,'test')
  if name=='letters': xx=xx.transpose(0,2,1)
  feats=encode(xx);cells=head.cell(feats)
  # Stratified, seeded and disjoint log/eval halves; pools retain all classes.
  logids=[];evalids=[]
  for c in np.unique(yy):
   ids=np.flatnonzero(yy==c);ids=np.random.default_rng(100+int(c)).permutation(ids)
   logids.extend(ids[:len(ids)//2]);evalids.extend(ids[len(ids)//2:])
  logids=np.array(logids);evalids=np.array(evalids)
  assert not set(logids)&set(evalids)
  pools[name]=[cells[logids][yy[logids]==c] for c in np.unique(yy)]
  dist=head._distances(feats[evalids]).min(1)
  with torch.no_grad():
   logits=net.output(torch.from_numpy(feats[evalids]).to(device)).cpu()
   prob=logits.softmax(1);ent=-(prob*prob.clamp_min(1e-12).log()).sum(1)
  for j,i in enumerate(evalids):
   eval_rows.append(dict(dataset=name,index=int(i),label=int(yy[i]),cell=int(cells[i]),cnn_prediction=int(logits[j].argmax()),entropy=float(ent[j]),distance=float(dist[j])))
  for role,ids in [('log_pool',logids),('evaluation',evalids)]:
   assignments += [dict(dataset=name,split='test',index=int(i),role=role,label=int(yy[i])) for i in ids]
 pd.DataFrame(assignments).to_csv(HERE/'pool_assignments.csv',index=False)
 ev=pd.DataFrame(eval_rows);ev.to_csv(HERE/'evaluation_scores.csv',index=False)
 diagnostics={'cnn_known_accuracy':float((ev[ev.dataset=='mnist'].label==ev[ev.dataset=='mnist'].cnn_prediction).mean())}
 groups=[(c,) for c in range(10)];D=head.cells
 for name in ['letters','kmnist']:
  sub=ev[ev.dataset.isin(['mnist',name])]
  diagnostics[name]={ 'entropy_auroc':roc_auc_score(sub.dataset!= 'mnist',sub.entropy), 'distance_auroc':roc_auc_score(sub.dataset!='mnist',sub.distance),
    'outlier_cell_fraction':float((ev[ev.dataset==name].cell==10).mean()),
    'cell_hull_distance':hull_distance(_group_hist(pools[name],D),np.stack([_group_hist([p],D) for p in pools['mnist']]))}
 (HERE/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2));print('Diagnostics',diagnostics,flush=True)
 # Distribution shift controlled by source weights. Novel references are always letters.
 configs=[('baseline',1000,200,50),('n_low',250,200,50),('n_high',4000,200,50),('m_low',1000,50,50),('m_high',1000,1000,50),('r_low',1000,200,10),('r_high',1000,200,200)]
 scenarios=[('digits_uniform','letters',0.,'uniform'),('digits_shift','letters',0.,'dirichlet'),('letters_05','letters',.05,'dirichlet'),('letters_20','letters',.2,'dirichlet'),('kmnist_05','kmnist',.05,'dirichlet'),('kmnist_20','kmnist',.2,'dirichlet')]
 records=[];warning_rows=[]
 for scenario,novel,delta,prev in scenarios:
  for setting,n,m,r in configs:
   ctx=dict(arm='held_out',label=scenario,known=np.arange(10),novel=np.arange(26),groups=groups,grouping='classes',d=D,set_kind='l2',cell_group=np.r_[np.arange(10),-1],
     prevalence='dirichlet',dirichlet_alpha=1.,delta=delta,
     ref_cells={**ref_pools,'novel':[p[:r] for p in letter_ref]},
     dep_cells={**{g:[pools['mnist'][j]] for j,g in enumerate(groups)},'novel':pools[novel]})
   ctx['truth_conditionals']=np.vstack([_group_hist(ctx['dep_cells'][g],D) for g in groups+['novel']])
   # Uniform control uses the same original engine with an explicit weight override.
   original=audit.true_weights
   if prev=='uniform':
    audit.true_weights=lambda context,seed:np.r_[np.full(10,(1-context['delta'])/10),context['delta']]
   try:
    for seed in config['seeds']:
     with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter('always')
      out=run_cell(ctx,n,m,seed,'oracle',n_boot=20,members=5)
     records += [dict(row,scenario=scenario,setting=setting,r=r) for row in out]
     warning_rows += [dict(scenario=scenario,setting=setting,seed=seed,message=str(w.message)) for w in caught]
     if scenario=='letters_05' and setting=='baseline' and seed==0:
      # Explicit inputs to a representative original REACH call.
      a,centre,rr,entry=audit._reference(ctx,m,seed)
      q=_group_hist(ctx['ref_cells']['novel'],D)
      weights=audit.true_weights(ctx,seed)
      log=audit._draw_log(ctx,max(audit.GRIDS['n']+(n,)),seed,weights)[:n]
      np.savez(HERE/'single_run_inputs.npz',known_conditionals=a,novel_conditional=q,true_weights=weights,truth_conditionals=ctx['truth_conditionals'],log_cells=log,reference_radius=rr,entry_radius=entry)
      pd.DataFrame(out).to_csv(HERE/'single_run.csv',index=False)
   finally: audit.true_weights=original
   pd.DataFrame(records).to_csv(HERE/'reach_rows.csv',index=False)
   print('Finished',scenario,setting,'records',len(records),flush=True)
 (HERE/'warnings.json').write_text(json.dumps(warning_rows,indent=2))
 df=pd.DataFrame(records);on=df[df.region=='on_support']
 cols=['W_cls','W_label','W_joint','W_unk','W_unk_joint','joint_covers','unk_joint_covers','incompatible']
 for c in cols: on=on.assign(**{c:pd.to_numeric(on[c],errors='coerce')})
 on.groupby(['scenario','setting'])[cols].mean().to_csv(HERE/'summary.csv')
 df.groupby(['scenario','setting','cls_status']).size().rename('rows').to_csv(HERE/'solver_status.csv')
 plt.rcdefaults();fig,axes=plt.subplots(1,3,figsize=(14,4))
 for ax,(prefix,values,title) in zip(axes,[('n',[250,1000,4000],'Deployment observations'),('m',[50,200,1000],'Digit references/class'),('r',[10,50,200],'Letter references/class')]):
  for scenario in ['letters_20','kmnist_20']:
   means=[on[(on.scenario==scenario)&(on.setting==s)].W_unk_joint.mean() for s in [prefix+'_low','baseline',prefix+'_high']]
   ax.plot(values,means,'o-',label=scenario)
  ax.set(xscale='log',xlabel=title,ylabel='Mean joint novelty width',ylim=(0,1.05));ax.legend()
 fig.tight_layout();fig.savefig(HERE/'acquisition_response.png',dpi=150)
 print('Finished pilot.',flush=True)

if __name__=='__main__':main()
