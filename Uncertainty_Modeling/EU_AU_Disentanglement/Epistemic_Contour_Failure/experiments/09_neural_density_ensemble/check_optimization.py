"""Continue fitted mixture laws directly to diagnose neural parameterization effects."""
import csv,json
import numpy as np
import torch
from torch import nn
import run as r

class Direct(nn.Module):
    def __init__(self,p):
        super().__init__()
        q=np.clip((p[:,3]-.35)/1.65,1e-14,1-1e-14)
        z=np.column_stack([np.log(p[:,0]),np.arctanh(np.clip(p[:,1:3]/2,-1+1e-14,1-1e-14)),np.log(q/(1-q))])
        self.z=nn.Parameter(torch.tensor(z.reshape(1,12)))
    def forward(self,x):return self.z

saved=np.load(r.OUT/'fits.npz');rows=[];output={};max_formula=0
for key in saved.files:
    if not key.endswith('_params'):continue
    prefix=key[:-7];params=saved[key];counts=saved[prefix+'_counts'];nets=[Direct(p) for p in params]
    initial=np.array([r.probs(net).detach().numpy() for net in nets]);final=[];newparams=[];g=[];improve=[]
    for net,p,p0 in zip(nets,params,initial):
        independent=(p[:,0,None]*r.prev.laws(p[:,1:])).sum(0)
        max_formula=max(max_formula,float(abs(p0-independent).max()))
        loss0=-np.sum(counts/counts.sum()*np.log(p0))
        loss,grad=r.train(net,counts,iterations=1000)
        final.append(r.probs(net).detach().numpy());newparams.append(r.packed(net));g.append(grad);improve.append(loss0-loss)
    final=np.array(final);newparams=np.array(newparams)
    combined=newparams.reshape(-1,4).copy();combined[:,0]/=r.MEMBERS
    center,*_=r.contour(combined)
    members=[r.contour(p)[0] for p in newparams]
    rows.append(dict(dataset=prefix,mi_before=r.mi(initial),mi_after=r.mi(final),
        max_probability_l1=float(abs(final-initial).sum(1).max()),max_nll_improvement=max(improve),
        max_gradient=max(g),contour_change=float(r.hd(center,saved[prefix+'_contour'])),
        spread_after=float(max(r.hd(c,center) for c in members))))
    output[prefix+'_params']=newparams;output[prefix+'_contour']=center
with (r.OUT/'optimization_sensitivity.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
np.savez_compressed(r.OUT/'polished_fits.npz',**output)
(r.OUT/'sensitivity_validation.json').write_text(json.dumps({'max_torch_scipy_probability_difference':max_formula,'datasets':len(rows)},indent=2)+'\n')
assert max_formula<1e-12
print('max contour change',max(x['contour_change'] for x in rows))
print('max remaining gradient',max(x['max_gradient'] for x in rows))
