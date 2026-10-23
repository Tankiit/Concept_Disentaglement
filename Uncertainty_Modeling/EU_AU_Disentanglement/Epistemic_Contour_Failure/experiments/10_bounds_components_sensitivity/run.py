"""Matched factorial sensitivity: component count and component parameter bounds."""
from pathlib import Path
import importlib.util
import json,csv,hashlib
import numpy as np
import torch
from torch import nn
from scipy.special import ndtr
import contourpy

OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('neural09',OUT.parent/'09_neural_density_ensemble/run.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
CONFIGS=[('narrow3',3,2.,.35,2.),('wide3',3,4.,.2,3.),('narrow5',5,2.,.35,2.),('wide5',5,4.,.2,3.)]
SEED=20261030


def decode(net,cfg):
    _,k,b,low,high=cfg;z=net(r.INPUT).reshape(k,4)
    return torch.softmax(z[:,0],0),b*torch.tanh(z[:,1:3]),low+(high-low)*torch.sigmoid(z[:,3])


def probs(net,cfg):
    w,mu,s=decode(net,cfg);k=len(w)
    cdf=torch.special.ndtr((r.CUTS[None,None,:]-mu[:,:,None])/s[:,None,None])
    cdf=torch.cat([torch.zeros((k,2,1)),cdf,torch.ones((k,2,1))],dim=-1)
    p=cdf.diff(dim=-1)
    return (w[:,None,None]*p[:,1,:,None]*p[:,0,None,:]).sum(0).reshape(36)


def train(net,counts,cfg):
    freq=torch.tensor(counts/counts.sum())
    opt=torch.optim.LBFGS(net.parameters(),lr=1,max_iter=400,tolerance_grad=1e-9,
                         tolerance_change=1e-12,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad();loss=-(freq*probs(net,cfg).clamp_min(1e-100).log()).sum();loss.backward();return loss
    opt.step(closure);loss=float(closure().detach())
    return loss,max(float(p.grad.abs().max()) for p in net.parameters())


def contour(params,h=.04):
    # Expand integration domain with actual component locations/scales, including tiny weights.
    limit=max(12.,float(np.max(abs(params[:,1:3])+7*params[:,3,None])))
    limit=np.ceil(limit/h)*h
    axis=np.linspace(-limit,limit,round(2*limit/h)+1);h=axis[1]-axis[0]
    x,y=np.meshgrid(axis,axis);f=r.density(params,x,y)
    enclosed=np.sum(params[:,0]*np.prod(ndtr((limit-params[:,1:3])/params[:,3,None])-ndtr((-limit-params[:,1:3])/params[:,3,None]),axis=1))
    assert enclosed>1-1e-9
    w=np.ones_like(f);w[[0,-1],:]*=.5;w[:,[0,-1]]*=.5
    order=np.argsort(f.ravel())[::-1];mass=np.cumsum((f*w).ravel()[order])*h*h
    assert abs(mass[-1]-enclosed)<1e-7
    level=float(f.ravel()[order[np.searchsorted(mass,.95)]])
    lines=contourpy.contour_generator(x=axis,y=axis,z=f).lines(level)
    assert lines and all(np.allclose(line[0],line[-1]) for line in lines)
    dense=[]
    for line in lines:
        for a,b in zip(line[:-1],line[1:]):
            k=max(1,int(np.ceil(np.linalg.norm(b-a)/.005)))
            dense.append(a+(b-a)*np.arange(k)[:,None]/k)
    return np.concatenate(dense),dict(limit=float(limit),enclosed_mass=float(enclosed),quadrature_mass=float(mass[-1]),level=level)


def save_csv(name,rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    source=OUT.parent/'09_neural_density_ensemble/fits.npz';data=np.load(source)
    datasets=[key[:-7] for key in data.files if key.endswith('_counts')]
    angle=np.linspace(0,2*np.pi,12000,endpoint=False)
    circle=r.prev.C*np.column_stack([np.cos(angle),np.sin(angle)])
    clean_error=r.hd(contour(np.array([[1.,-.2,0.,1.]]))[0],circle+[-.2,0])
    assert clean_error<.01
    allrows=[];training=[];refinements=[];max_formula=0.
    # Resume completed configurations after a failed diagnostic, preserving their fitted results.
    if (OUT/'rows.csv').exists():
        allrows=list(csv.DictReader((OUT/'rows.csv').open()))
        training=list(csv.DictReader((OUT/'training.csv').open()))
    for cfg in CONFIGS:
        label,k,b,low,high=cfg;saved={}
        if (OUT/(label+'_fits.npz')).exists():
            print('Reusing completed configuration',label,flush=True)
            continue
        for index,key in enumerate(datasets):
            # Paired initial random weights across bounds at fixed component count.
            torch.manual_seed(SEED+index)
            count=data[key+'_counts'];n=int(count.sum());scenario='shared_mixture' if key.startswith('shared') else 'clean'
            base=nn.Sequential(nn.Linear(1,16),nn.Tanh(),nn.Linear(16,4*k))
            nets=r.ensemble(base,num_members=5,reset_params=True)
            ptrs=[p.data_ptr() for net in nets for p in net.parameters()]
            assert len(ptrs)==len(set(ptrs))
            parameters=[];predictions=[]
            for j,net in enumerate(nets):
                first,_=train(net,count,cfg);p0=probs(net,cfg).detach().numpy().copy()
                last,g=train(net,count,cfg);p=probs(net,cfg).detach().numpy()
                assert last<=first+1e-9
                params=np.column_stack([v.detach().numpy() for v in decode(net,cfg)])
                z=(r.prev.EDGES[None,None,:]-params[:,1:3,None])/params[:,3,None,None]
                marginal=np.where(z[:,:,:-1]>=0,ndtr(-z[:,:,:-1])-ndtr(-z[:,:,1:]),ndtr(z[:,:,1:])-ndtr(z[:,:,:-1]))
                cell=(marginal[:,1,:,None]*marginal[:,0,None,:]).reshape(k,36)
                assert np.all(cell>=0)
                np.testing.assert_allclose(cell.sum(1),1,atol=1e-12)
                independent=(params[:,0,None]*cell).sum(0)
                max_formula=max(max_formula,float(abs(p-independent).max()))
                # Count near-bound components, both all weights and weights >= .01.
                near=(np.max(abs(params[:,1:3]),axis=1)>.99*b)|(params[:,3]<low+.01*(high-low))|(params[:,3]>high-.01*(high-low))
                training.append(dict(config=label,dataset=key,member=j,nll=last,gradient=g,
                    continuation_improvement=first-last,continuation_l1=float(abs(p-p0).sum()),
                    near_bound=int(near.sum()),near_bound_weight_ge_01=int((near&(params[:,0]>=.01)).sum())))
                parameters.append(params);predictions.append(p)
            parameters=np.array(parameters);predictions=np.array(predictions)
            combined=parameters.reshape(-1,4).copy();combined[:,0]/=5
            center,meta=contour(combined)
            curves=[contour(p)[0] for p in parameters]
            spread=float(max(r.hd(c,center) for c in curves))
            if key.endswith('_0'):
                fine,_=contour(combined,h=.02)
                refinements.append(dict(config=label,dataset=key,change=float(r.hd(center,fine))))
            truth_centers=[[-.2,0]] if scenario=='clean' else [[-.2,0],[.2,0]]
            for j,truth in enumerate(truth_centers):
                error=float(r.hd(center,circle+truth))
                allrows.append(dict(config=label,dataset=key,scenario=scenario,n=n,truth=j,
                    mi_nats=r.mi(predictions),contour_error=error,member_radius=spread,
                    inside_member_radius=int(error<=spread),**meta))
            saved[key+'_params']=parameters;saved[key+'_contour']=center
            print(label,key,'mi',r.mi(predictions),'radius',spread,flush=True)
        np.savez_compressed(OUT/(label+'_fits.npz'),**saved)
        save_csv('rows.csv',allrows);save_csv('training.csv',training)
    assert max_formula<1e-12
    # Recompute refinement checks for every saved configuration, including resumed runs.
    refinements=[]
    for cfg in CONFIGS:
        label=cfg[0];archive=np.load(OUT/(label+'_fits.npz'))
        for key in datasets:
            if not key.endswith('_0'):continue
            parameters=archive[key+'_params'];combined=parameters.reshape(-1,4).copy();combined[:,0]/=5
            fine,_=contour(combined,h=.02)
            refinements.append(dict(config=label,dataset=key,change=float(r.hd(archive[key+'_contour'],fine))))
    (OUT/'validation.json').write_text(json.dumps(dict(seed=SEED,configs=CONFIGS,datasets=len(datasets),members=5,
        source_counts_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),analytic_contour_error=float(clean_error),
        refinements=refinements,max_probability_formula_difference=max_formula,
        max_gradient=max(float(d['gradient']) for d in training),max_continuation_l1=max(float(d['continuation_l1']) for d in training)),indent=2)+'\n')


if __name__=='__main__':main()
