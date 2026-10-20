"""Audit one declared-letter novelty interval using saved pilot inputs."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
BASE=ROOT/'workbook/mnist_reach_novelty_2026-10-19'
sys.path.insert(0,str(ROOT))
from models.ball import MMDBall,embedding_radius
from real.audit import ALPHA


def minmass(ball,den):
 w=cp.Variable(len(den),nonneg=True)
 prob=cp.Problem(cp.Minimize(den@w),[cp.sum(w)==1,cp.norm(ball.conditionals.T@w-ball.empirical,2)<=ball.radius])
 prob.solve(solver=cp.CLARABEL)
 return dict(status=prob.status,value=None if prob.value is None else float(prob.value),weights=None if w.value is None else w.value.tolist(),residual=None if w.value is None else float(np.linalg.norm(w.value@ball.conditionals-ball.empirical)))


def interval(ball,lo,hi,q_radius,cell):
 col=ball.conditionals[:,cell];num=np.zeros_like(col);num[-1]=col[-1]
 plugin=ball.ratio_interval(num,num,col)
 qlo=max(col[-1]-q_radius,0);qhi=min(col[-1]+q_radius,1)
 nlo=np.zeros_like(col);nlo[-1]=qlo
 nhi=np.zeros_like(col);nhi[-1]=qhi
 dlo=np.r_[hi[:,cell],qlo];dhi=np.r_[lo[:,cell],qhi]
 low=ball.ratio_interval(nlo,nlo,dlo) if qlo>0 else dict(lower=0.,status='novel_entry_lower_zero')
 high=ball.ratio_interval(nhi,nhi,dhi) if qhi>0 else dict(upper=0.,status='novel_entry_upper_zero')
 return dict(plugin_lower=plugin['lower'],plugin_upper=plugin['upper'],plugin_status=plugin['status'],joint_lower=low['lower'],joint_upper=high['upper'],lower_status=low['status'],upper_status=high['status'],known_lower_sum=float(lo[:,cell].sum()),q_lower=qlo,q_upper=qhi), {'plugin_mass':minmass(ball,col),'lower_mass':minmass(ball,dlo),'upper_mass':minmass(ball,dhi)}


def main():
 with np.load(BASE/'single_run_inputs.npz') as z:
  estimated=np.vstack([z['known_conditionals'],z['novel_conditional']]);truth=z['truth_conditionals'];weights=z['true_weights'];prefix=z['log_cells'];ref=float(z['reference_radius']);entry=float(z['entry_radius'])
 cell=0;k=10;r=50;qrad=embedding_radius(26*r,ALPHA/(3*(k+1)))
 marginal=weights@truth;target=float(weights[-1]*truth[-1,cell]/marginal[cell])
 # Preserve the original 1000 observations. Additional independent simulated
 # cells follow the same finite-pool marginal; larger logs are nested prefixes.
 rng=np.random.default_rng(20261019)
 log=np.r_[prefix,rng.choice(len(marginal),64000-len(prefix),p=marginal)]
 np.savez(HERE/'diagnostic_inputs.npz',estimated=estimated,truth=truth,weights=weights,log=log)
 config=dict(cell=cell,novelty='EMNIST Letters',delta=.05,seed=0,n=[1000,4000,16000,64000],m=200,r=50,extra_log_seed=20261019,reference_radius=ref,novel_radius=qrad,entry_halfwidth=entry,
  input_sha256=hashlib.sha256((BASE/'single_run_inputs.npz').read_bytes()).hexdigest(),scope='single cell, single simulation; oracle finite-pool diagnostic, not deployable')
 (HERE/'config.json').write_text(json.dumps(config,indent=2))
 records=[];witnesses={}
 cases=['estimated','oracle_exact','oracle_with_reference_bounds','estimated_bounds_removed']
 for n in config['n']:
  for case in cases:
   a=truth if case.startswith('oracle') else estimated
   bounded=case in ['estimated','oracle_with_reference_bounds']
   radius=max(ref,qrad) if bounded else 0.
   qr=qrad if bounded else 0.
   lo=np.clip(a[:-1]-entry,0,1) if bounded else a[:-1]
   hi=np.clip(a[:-1]+entry,0,1) if bounded else a[:-1]
   ball=MMDBall.from_log(a,log[:n],ALPHA/3,radius,set_kind='l2')
   out,witness=interval(ball,lo,hi,qr,cell)
   records.append(dict(case=case,n=n,true_novel_posterior=target,eps_n=ball.eps_n,reference_radius=radius,total_radius=ball.radius,**out))
   witnesses[f'{case}:{n}']=witness
 # In the no-sampling-error limit, test whether true mixture weights can be
 # distinguished from the exact marginal with the exact source distributions.
 ball=MMDBall(truth,marginal,0.,0.,set_kind='l2')
 out,witness=interval(ball,truth[:-1],truth[:-1],0.,cell)
 records.append(dict(case='oracle_exact_marginal',n=0,true_novel_posterior=target,eps_n=0.,reference_radius=0.,total_radius=0.,**out))
 witnesses['oracle_exact_marginal']=witness
 df=pd.DataFrame(records);df['joint_width']=df.joint_upper-df.joint_lower;df['plugin_width']=df.plugin_upper-df.plugin_lower
 df['covers_truth']=(df.joint_lower<=target+1e-6)&(df.joint_upper>=target-1e-6)
 df.to_csv(HERE/'intervals.csv',index=False)
 (HERE/'mass_witnesses.json').write_text(json.dumps(witnesses,indent=2))
 original=pd.read_csv(BASE/'single_run.csv').query('cell == 0').iloc[0]
 baseline=df[(df['case']=='estimated')&(df.n==1000)].iloc[0]
 assert np.isclose(baseline.joint_lower,original.unk_joint_lower,atol=1e-6)
 assert np.isclose(baseline.joint_upper,original.unk_joint_upper,atol=1e-6)
 assert np.isfinite(df[['joint_lower','joint_upper']]).all().all()
 assert (df.joint_width>=-1e-6).all()
 singular=np.linalg.svd(truth,compute_uv=False)
 (HERE/'identifiability.json').write_text(json.dumps(dict(rank=int(np.linalg.matrix_rank(truth)),shape=list(truth.shape),singular_values=singular.tolist(),condition_number=float(np.linalg.cond(truth))),indent=2))
 plt.rcdefaults();fig,axes=plt.subplots(1,2,figsize=(12,4))
 labels={'estimated':'Estimated references','oracle_exact':'Exact conditionals, no reference bounds','oracle_with_reference_bounds':'Exact centres, original bounds','estimated_bounds_removed':'Estimated centres, bounds removed (uncalibrated)'}
 for case,label in labels.items():
  sub=df[df['case']==case];axes[0].plot(sub.n,sub.joint_width,'o-',label=label)
  axes[1].plot(sub.n,sub.plugin_width,'o-',label=label)
 for ax,title in zip(axes,['Joint novelty interval width','Plug-in novelty interval width']):
  ax.set(xscale='log',xlabel='Deployment observations',ylabel='Width',title=title,ylim=(-.02,1.02))
 axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(HERE/'oracle_comparison.png',dpi=150)
 print(df[['case','n','joint_lower','joint_upper','joint_width','plugin_width','lower_status','upper_status']].round(5).to_string(index=False))
 print('Known entry lower sum:',baseline.known_lower_sum,'q lower:',baseline.q_lower,'entry radius:',entry)
 print('Exact source matrix rank:',np.linalg.matrix_rank(truth),'condition number:',np.linalg.cond(truth))
 print('Original baseline reproduced; interval checks passed.')

if __name__=='__main__':main()
