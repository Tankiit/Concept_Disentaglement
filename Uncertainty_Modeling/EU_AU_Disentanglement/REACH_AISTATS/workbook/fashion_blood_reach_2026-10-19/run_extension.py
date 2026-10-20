"""Extend the frozen MNIST REACH pilot; never retrain or recalibrate."""
from pathlib import Path
import sys,json,hashlib,warnings
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BASE=ROOT/'workbook/mnist_reach_novelty_2026-10-19'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(BASE))
from run_reach import DigitNet,tensor,data
from real.head import LDAHead
from real.audit import run_cell,_group_hist,hull_distance

def main():
 torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
 config=dict(seeds=[0,1,2],delta=[.05,.2],baseline=dict(n=1000,m=200,r=50),
  n=[250,1000,4000],m=[50,200,1000],r=[10,50,200],n_boot=20,members=5,
  model_retrained=False,partition_refitted=False,novel_reference='EMNIST letters only',
  base=str(BASE.relative_to(ROOT)),checkpoint_sha256=hashlib.sha256((BASE/'digit_net.pt').read_bytes()).hexdigest(),
  partition_sha256=hashlib.sha256((BASE/'partition.npz').read_bytes()).hexdigest())
 (HERE/'config.json').write_text(json.dumps(config,indent=2))
 net=DigitNet().to(device);net.load_state_dict(torch.load(BASE/'digit_net.pt',map_location=device,weights_only=True));net.eval();net.requires_grad_(False)
 with np.load(BASE/'partition.npz') as a: head=LDAHead(**{k:a[k] for k in a.files})
 def encode(images):
  with torch.no_grad():return np.concatenate([net.features(tensor(images[s:s+256],device)).cpu().numpy() for s in range(0,len(images),256)])
 assignments=pd.read_csv(BASE/'pool_assignments.csv')
 refs={};letter_ref=[]
 for name,role in [('mnist','reference'),('letters','novel_reference')]:
  ids=assignments[(assignments.dataset==name)&(assignments.role==role)]['index'].to_numpy()
  X,y=data(name,'train')
  if name=='letters':X=X.transpose(0,2,1)
  cells=head.cell(encode(X[ids]));labels=y[ids]
  if name=='mnist':refs={(c,):[cells[labels==c]] for c in range(10)}
  else:letter_ref=[cells[labels==c] for c in range(26)]
 pools={};evrows=[];newassign=[]
 for name in ['mnist','fashion','blood']:
  X,y=data(name,'test');features=encode(X);cells=head.cell(features)
  if name=='mnist':
   ids=assignments[(assignments.dataset==name)&(assignments.role=='log_pool')]['index'].to_numpy()
   eids=assignments[(assignments.dataset==name)&(assignments.role=='evaluation')]['index'].to_numpy()
  else:
   ids=[];eids=[]
   for c in np.unique(y):
    ix=np.flatnonzero(y==c);ix=np.random.default_rng(100+int(c)).permutation(ix)
    ids.extend(ix[:len(ix)//2]);eids.extend(ix[len(ix)//2:])
   ids=np.array(ids);eids=np.array(eids)
  assert not set(ids)&set(eids) and len(ids)+len(eids)==len(y)
  pools[name]=[cells[ids][y[ids]==c] for c in np.unique(y)]
  dist=head._distances(features[eids]).min(1)
  with torch.no_grad():
   logits=net.output(torch.from_numpy(features[eids]).to(device)).cpu();p=logits.softmax(-1)
   entropy=-(p*p.clamp_min(1e-12).log()).sum(-1)
  evrows.extend(dict(dataset=name,index=int(i),label=int(y[i]),cell=int(cells[i]),cnn_prediction=int(logits[j].argmax()),entropy=float(entropy[j]),distance=float(dist[j])) for j,i in enumerate(eids))
  for role,idx in [('log_pool',ids),('evaluation',eids)]:
   newassign.extend(dict(dataset=name,split='test',index=int(i),label=int(y[i]),role=role) for i in idx)
 pd.DataFrame(newassign).to_csv(HERE/'pool_assignments.csv',index=False)
 ev=pd.DataFrame(evrows);ev.to_csv(HERE/'evaluation_scores.csv',index=False)
 old=pd.read_csv(BASE/'evaluation_scores.csv');known=ev[ev.dataset=='mnist'].sort_values('index');oldknown=old[old.dataset=='mnist'].sort_values('index')
 assert np.array_equal(known.cell,oldknown.cell) and np.allclose(known.entropy,oldknown.entropy,atol=1e-6)
 diagnostics={'known_reproduction_verified':True,'known_accuracy':float((known.label==known.cnn_prediction).mean())}
 for name in ['fashion','blood']:
  sub=ev[ev.dataset.isin(['mnist',name])]
  diagnostics[name]=dict(entropy_auroc=roc_auc_score(sub.dataset!='mnist',sub.entropy),distance_auroc=roc_auc_score(sub.dataset!='mnist',sub.distance),outlier_fraction=float((ev[ev.dataset==name].cell==10).mean()),cell_hull_distance=hull_distance(_group_hist(pools[name],11),np.stack([_group_hist([p],11) for p in pools['mnist']])))
 (HERE/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2));print(diagnostics,flush=True)
 configs=[('baseline',1000,200,50),('n_low',250,200,50),('n_high',4000,200,50),('m_low',1000,50,50),('m_high',1000,1000,50),('r_low',1000,200,10),('r_high',1000,200,200)]
 records=[];warn=[];groups=[(c,) for c in range(10)]
 for novel in ['fashion','blood']:
  for delta in [.05,.2]:
   scenario=f'{novel}_{int(delta*100):02d}'
   for setting,n,m,r in configs:
    ctx=dict(arm='held_out',label=scenario,known=np.arange(10),novel=np.arange(26),groups=groups,grouping='classes',d=11,set_kind='l2',cell_group=np.r_[np.arange(10),-1],prevalence='dirichlet',dirichlet_alpha=1.,delta=delta,
     ref_cells={**refs,'novel':[p[:r] for p in letter_ref]},dep_cells={**{g:[pools['mnist'][j]] for j,g in enumerate(groups)},'novel':pools[novel]})
    ctx['truth_conditionals']=np.vstack([_group_hist(ctx['dep_cells'][g],11) for g in groups+['novel']])
    for seed in [0,1,2]:
     with warnings.catch_warnings(record=True) as ws:
      warnings.simplefilter('always');out=run_cell(ctx,n,m,seed,'oracle',n_boot=20,members=5)
     records.extend(dict(row,scenario=scenario,setting=setting,r=r) for row in out)
     warn.extend(dict(scenario=scenario,setting=setting,seed=seed,message=str(w.message)) for w in ws)
    pd.DataFrame(records).to_csv(HERE/'reach_rows.csv',index=False)
    print('Finished',scenario,setting,len(records),flush=True)
 (HERE/'warnings.json').write_text(json.dumps(warn,indent=2))
 print('Completed all 84 calls.',flush=True)

if __name__=='__main__':main()
