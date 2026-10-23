"""Matched grouped-data frequentist confidence and bootstrap comparison."""
from pathlib import Path
import importlib.util,json
import numpy as np
from scipy.special import ndtr,ndtri
from scipy.stats import chi2,norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
EDGES=np.array([-np.inf,-2,-1,0,1,2,np.inf])
C=np.sqrt(-2*np.log(.05));REPS=100;BOOT=399;SEED=20261027
NS=[1000,10000,100000]

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
h=load('gauss',OUT.parent/'04_general_contamination_model/run.py')
f=load('freq',OUT.parent/'05_frequentist_theory/run.py')

def model(t):
 sigma=np.exp(t[:,2]);z=(EDGES[None,None,:]-t[:,:2,None])/sigma[:,None,None]
 pdf=norm.pdf(z);zpdf=np.multiply(z,pdf,out=np.zeros_like(z),where=np.isfinite(z))
 p=np.where(z[:,:,:-1]>=0,ndtr(-z[:,:,:-1])-ndtr(-z[:,:,1:]),ndtr(z[:,:,1:])-ndtr(z[:,:,:-1]))
 p=np.maximum(p,1e-300)
 dm=(pdf[:,:,:-1]-pdf[:,:,1:])/sigma[:,None,None]
 ds=zpdf[:,:,:-1]-zpdf[:,:,1:]
 score=np.zeros((len(t),2,6,3));score[:,0,:,0]=dm[:,0]/p[:,0];score[:,1,:,1]=dm[:,1]/p[:,1];score[:,:,:,2]=ds/p
 return p,score

def fit(counts,start=None):
 counts=np.asarray(counts);n=counts.sum(axis=1)
 joint=counts.reshape(-1,6,6);marg=np.stack([joint.sum(axis=1),joint.sum(axis=2)],axis=1)
 if start is None:
  t=np.zeros((len(counts),3));t[:,:2]=ndtri(np.clip(marg[:,:,3:].sum(axis=2)/n[:,None],.01,.99))
 else:t=np.tile(start,(len(counts),1))
 converged=np.zeros(len(t),bool)
 for iteration in range(80):
  p,s=model(t);grad=np.einsum('bij,bijk->bk',marg,s)/n[:,None]
  fisher=np.einsum('bij,bijk,bijl->bkl',p,s,s)
  delta=np.linalg.solve(fisher,grad[:,:,None])[:,:,0]
  done=np.max(abs(delta),axis=1)<1e-8;converged|=done
  if np.all(done):break
  delta/=np.maximum(1,np.max(abs(delta),axis=1)/.3)[:,None]
  current=np.sum(marg*np.log(p),axis=(1,2))/n
  step=np.ones(len(t));candidate=t+delta
  for _ in range(20):
   candidate=t+step[:,None]*delta
   pp,_=model(candidate);ll=np.sum(marg*np.log(pp),axis=(1,2))/n
   bad=ll<current-1e-13
   if not bad.any():break
   step[bad]*=.5
  t=candidate
 p,s=model(t);grad=np.einsum('bij,bijk->bk',marg,s)/n[:,None]
 fisher=np.einsum('bij,bijk,bijl->bkl',p,s,s)
 assert np.max(abs(grad))<2e-6,('fit gradient',np.max(abs(grad)))
 return t,np.linalg.inv(fisher)/n[:,None,None],float(np.max(abs(grad)))

def laws(theta):return h.laws(theta)

def lower_probs(lo,hi):
 # Every endpoint standardized value is bounded by its four (mu,sigma) corners.
 z=np.stack([(EDGES[None,None,:]-mu[:,:2,None])/sig[:,2,None,None] for mu in (lo,hi) for sig in (lo,hi)])
 zlo=z.min(axis=0);zhi=z.max(axis=0)
 p=np.maximum(0,ndtr(zlo[:,:,1:])-ndtr(zhi[:,:,:-1]))
 return (p[:,1,:,None]*p[:,0,None,:]).reshape(len(lo),36)

def reference(L,U,eps,max_rounds=6):
 lo=np.array([[-.5,-.5,.65]]);hi=np.array([[.5,.5,1.35]])
 for round in range(max_rounds+1):
  good=f.feasible(lower_probs(lo,hi),L,U,eps)
  lo,hi=lo[good],hi[good]
  if not len(lo):return lo,hi,0.,0.
  if round==max_rounds:break
  # Split all three coordinates: uniform nested boxes and safe pruning.
  mid=(lo+hi)/2
  bits=np.array([[a,b,c] for a in [0,1] for b in [0,1] for c in [0,1]])
  newlo=np.where(bits[None,:,:],mid[:,None,:],lo[:,None,:]).reshape(-1,3)
  newhi=np.where(bits[None,:,:],hi[:,None,:],mid[:,None,:]).reshape(-1,3)
  lo,hi=newlo,newhi
 widths=hi.max(axis=0)-lo.min(axis=0)
 upper=float(np.linalg.norm(widths[:2])+C*widths[2])
 rho=float(np.max(np.linalg.norm((hi-lo)[:,:2]/2,axis=1)+C*(hi-lo)[:,2]/2))
 return lo,hi,upper,rho

def wilson(hits,total):
 z=1.95996398454;p=hits/total;den=1+z*z/total
 mid=(p+z*z/(2*total))/den;half=z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
 return [mid-half,mid+half]

def main():
 rng=np.random.default_rng(SEED)
 left=np.array([-.2,0.,1.]);off=np.array([.173,-.137,.973]);right=np.array([.2,0.,1.])
 mix,_=h.mixture(.2)
 scenarios=[('clean_left',laws(left[None,:])[0],0.,[left]),('clean_offgrid',laws(off[None,:])[0],0.,[off]),('shared_mixture',mix,.2,[left,right]),('shifted_contamination',.8*laws(off[None,:])[0]+.2*laws(np.array([[1.,-.7,1.05]]))[0],.2,[off])]
 rows=[];raw={};max_grad=0;max_rho=0
 for name,m,eps,truths in scenarios:
  for n in NS:
   counts=rng.multinomial(n,m,size=REPS)
   ts,covs,err=fit(counts);max_grad=max(max_grad,err)
   L,U=f.cp(counts,n,.05/36)
   records=[]
   for j in range(REPS):
    boot=rng.multinomial(n,counts[j]/n,size=BOOT)
    tb,_,err=fit(boot,ts[j]);max_grad=max(max_grad,err)
    bh=np.linalg.norm(tb[:,:2]-ts[j,:2],axis=1)+C*np.abs(np.exp(tb[:,2])-np.exp(ts[j,2]))
    radius=float(np.quantile(bh,.95,method='higher'))
    # Report a containing-ball upper diameter for the Wald ellipsoid.
    wald_radius=np.sqrt(chi2.ppf(.95,3)*np.linalg.eigvalsh(covs[j,:2,:2]).max())+C*np.exp(ts[j,2])*np.expm1(np.sqrt(chi2.ppf(.95,3)*covs[j,2,2]))
    blo,bhi,refwidth,rho=reference(L[j],U[j],eps);max_rho=max(max_rho,rho)
    for idx,truth in enumerate(truths):
     error=float(np.linalg.norm(ts[j,:2]-truth[:2])+C*abs(np.exp(ts[j,2])-truth[2]))
     delta=np.r_[truth[:2],np.log(truth[2])]-ts[j]
     wc=bool(delta@np.linalg.solve(covs[j],delta)<=chi2.ppf(.95,3))
     rc=bool(np.any(np.all((blo<=truth)&(truth<=bhi),axis=1))) if len(blo) else False
     exact=bool(f.feasible(laws(truth[None,:])[0],L[j],U[j],eps))
     event=bool(np.all((L[j]<=m)&(m<=U[j])))
     assert not event or (exact and rc)
     assert not exact or rc
     records.append([j,idx,wc,error<=radius,rc,exact,event,error,2*wald_radius,2*radius,refwidth,len(blo),rho])
   raw[f'{name}_{n}']=np.array(records)
   for idx,truth in enumerate(truths):
    subset=np.array([r for r in records if r[1]==idx])
    for method,col,widthcol in [('wald',2,8),('bootstrap',3,9),('reference_outer',4,10)]:
     hits=int(subset[:,col].sum())
     rows.append(dict(scenario=name,n=n,truth=truth.tolist(),method=method,coverage=hits/REPS,coverage_wilson95=wilson(hits,REPS),mean_diameter_upper=float(subset[:,widthcol].mean()),mean_fit_error=float(subset[:,7].mean()),mean_reference_boxes=float(subset[:,11].mean()),empty_reference_runs=int(np.sum(subset[:,11]==0))))
   print(f'{name} n={n} completed',flush=True)
 summary=dict(replicates=REPS,bootstrap_draws=BOOT,seed=SEED,rows=rows,max_fit_gradient=max_grad,max_reference_box_radius=max_rho,record_columns=['replicate','truth_index','wald_covered','bootstrap_covered','reference_outer_covered','exact_reference_covered','cell_law_covered','fit_hausdorff_error','wald_diameter_upper','bootstrap_diameter_upper','reference_diameter_upper','reference_boxes','box_radius'])
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');np.savez_compressed(OUT/'replicates.npz',**raw)
 fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
 for method in ['wald','bootstrap','reference_outer']:
  sub=[r for r in rows if r['scenario']=='shared_mixture' and r['truth'][0]==-.2 and r['method']==method]
  axs[0].semilogx(NS,[r['coverage'] for r in sub],'o-',label=method)
  axs[1].loglog(NS,[r['mean_diameter_upper'] for r in sub],'o-',label=method)
 axs[0].axhline(.95,color='gray',linestyle=':');axs[0].set(xlabel='Sample size',ylabel='Whole-contour coverage',ylim=(-.02,1.03));axs[0].legend()
 axs[1].set(xlabel='Sample size',ylabel='Mean contour-set diameter upper bound');axs[1].legend()
 fig.savefig(OUT/'comparison.png',dpi=180);plt.close(fig)
 print('Finished',len(rows),'summary rows')

if __name__=='__main__':main()
