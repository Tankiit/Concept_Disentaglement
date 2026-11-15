"""Small independently initialized neural mixture-density ensemble pilot."""
from pathlib import Path
import importlib.util
import json
import csv
import hashlib
import importlib.metadata
import numpy as np
from scipy.special import ndtr
from scipy.spatial import cKDTree
import contourpy
import torch
from torch import nn
from probly.method import ensemble
from probly.quantification import SecondOrderEntropyDecomposition
from probly.representation.distribution import create_categorical_distribution
from probly.representation.sample import ArraySample

OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('previous',OUT.parent/'06_full_cell_frequentist/run.py')
prev=importlib.util.module_from_spec(spec);spec.loader.exec_module(prev)
torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
SEED=20261029;REPS=5;MEMBERS=5
CUTS=torch.tensor([-2.,-1.,0.,1.,2.])
INPUT=torch.ones((1,1))


def decode(net):
    z=net(INPUT).reshape(3,4)
    return torch.softmax(z[:,0],0),2*torch.tanh(z[:,1:3]),.35+1.65*torch.sigmoid(z[:,3])


def probs(net):
    w,mu,s=decode(net)
    cdf=torch.special.ndtr((CUTS[None,None,:]-mu[:,:,None])/s[:,None,None])
    cdf=torch.cat([torch.zeros((3,2,1)),cdf,torch.ones((3,2,1))],dim=-1)
    p=cdf.diff(dim=-1)
    return (w[:,None,None]*p[:,1,:,None]*p[:,0,None,:]).sum(0).reshape(36)


def packed(net):
    return np.column_stack([x.detach().numpy() for x in decode(net)])


def train(net,counts,iterations=300):
    freq=torch.tensor(counts/counts.sum())
    optimizer=torch.optim.LBFGS(net.parameters(),lr=1,max_iter=iterations,
        tolerance_grad=1e-9,tolerance_change=1e-12,line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad();loss=-(freq*torch.log(probs(net).clamp_min(1e-100))).sum()
        loss.backward();return loss
    optimizer.step(closure)
    loss=float(closure().detach())
    gradient=max(float(p.grad.abs().max()) for p in net.parameters())
    return loss,gradient


def density(params,x,y):
    result=np.zeros_like(x)
    for weight,mx,my,s in params:
        result+=weight*np.exp(-((x-mx)**2+(y-my)**2)/(2*s*s))/(2*np.pi*s*s)
    return result


def contour(params,h=.04):
    # Absolute probability mass, not renormalization of a truncated distribution.
    axis=np.linspace(-12,12,round(24/h)+1);h=axis[1]-axis[0]
    x,y=np.meshgrid(axis,axis);f=density(params,x,y)
    enclosed=np.sum(params[:,0]*np.prod(ndtr((12-params[:,1:3])/params[:,3,None])-ndtr((-12-params[:,1:3])/params[:,3,None]),axis=1))
    assert enclosed>1-2e-6
    weights=np.ones_like(f);weights[[0,-1],:]*=.5;weights[:,[0,-1]]*=.5
    order=np.argsort(f.ravel())[::-1]
    mass=np.cumsum((f*weights).ravel()[order])*h*h
    level=float(f.ravel()[order[np.searchsorted(mass,.95)]])
    lines=contourpy.contour_generator(x=axis,y=axis,z=f).lines(level)
    assert lines and all(np.allclose(line[0],line[-1]) for line in lines)
    dense=[]
    for line in lines:
        for a,b in zip(line[:-1],line[1:]):
            k=max(1,int(np.ceil(np.linalg.norm(b-a)/.005)))
            dense.append(a+(b-a)*np.arange(k)[:,None]/k)
    return np.concatenate(dense),level,float(enclosed),float(mass[-1])


def hd(a,b):
    return max(cKDTree(a).query(b)[0].max(),cKDTree(b).query(a)[0].max())


def mi(p):
    np.testing.assert_allclose(p.sum(-1),1,atol=1e-12)
    return float(SecondOrderEntropyDecomposition(ArraySample(create_categorical_distribution(p),sample_axis=0)).epistemic)


def main():
    torch.manual_seed(SEED);rng=np.random.default_rng(SEED)
    clean=np.array([-.2,0.,1.]);mix,_=prev.h.mixture(.2)
    angle=np.linspace(0,2*np.pi,12000,endpoint=False)
    circle=prev.C*np.column_stack([np.cos(angle),np.sin(angle)])
    analytic=circle+clean[:2]
    numerical,_,_,_=contour(np.array([[1.,-.2,0.,1.]]))
    analytic_error=float(hd(numerical,analytic));assert analytic_error<.01
    rows=[];saved={};diagnostics=[];refinements=[]
    for scenario,law,truths in [('clean',prev.laws(clean[None])[0],[clean[:2]]),('shared_mixture',mix,[clean[:2],np.array([.2,0.])])]:
        for n in [1000,100000]:
            for rep in range(REPS):
                counts=rng.multinomial(n,law)
                base=nn.Sequential(nn.Linear(1,16),nn.Tanh(),nn.Linear(16,12))
                nets=ensemble(base,num_members=MEMBERS,reset_params=True)
                flat=[torch.cat([p.detach().flatten() for p in net.parameters()]) for net in nets]
                assert all(not torch.equal(flat[0],p) for p in flat[1:])
                ptrs=[p.data_ptr() for net in nets for p in net.parameters()]
                assert len(set(ptrs))==len(ptrs)
                parameters=[];probabilities=[];polish_deltas=[]
                for j,net in enumerate(nets):
                    first,g1=train(net,counts)
                    p1=probs(net).detach().numpy().copy()
                    last,g2=train(net,counts)
                    p=probs(net).detach().numpy()
                    assert last<=first+1e-9
                    diagnostics.append(dict(scenario=scenario,n=n,rep=rep,member=j,first_nll=first,final_nll=last,gradient=g2,polish_l1=float(abs(p-p1).sum())))
                    parameters.append(packed(net));probabilities.append(p);polish_deltas.append(abs(p-p1).sum())
                params=np.array(parameters);p=np.array(probabilities)
                # Ensemble mean density is the equal-weight mixture of member densities.
                combined=params.reshape(-1,4).copy();combined[:,0]/=MEMBERS
                center,level,mass,quad=contour(combined)
                members=[contour(param)[0] for param in params]
                spread=float(max(hd(c,center) for c in members))
                errors=[float(hd(center,circle+truth)) for truth in truths]
                if rep==0:
                    fine,_,_,_=contour(combined,h=.02)
                    refinements.append(dict(scenario=scenario,n=n,contour_change=float(hd(center,fine))))
                key=f'{scenario}_{n}_{rep}'
                saved[key+'_counts']=counts;saved[key+'_params']=params;saved[key+'_contour']=center
                for j,error in enumerate(errors):
                    rows.append(dict(scenario=scenario,n=n,rep=rep,truth=j,mi_nats=mi(p),contour_error=error,
                        max_member_distance=spread,inside_member_radius=int(error<=spread),mi_two_members=mi(p[:2]),
                        max_polish_l1=max(polish_deltas),threshold=level,enclosed_mass=mass,quadrature_mass=quad))
                print(scenario,n,rep,'MI',mi(p),'errors',errors,'spread',spread,flush=True)
    for name,data in [('rows.csv',rows),('training.csv',diagnostics)]:
        with (OUT/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    np.savez_compressed(OUT/'fits.npz',**saved)
    import probly
    backend=Path(probly.__file__).parent/'transformation/ensemble/torch.py'
    validation=dict(seed=SEED,reps=REPS,members=MEMBERS,analytic_contour_error=analytic_error,
        refinement=refinements,torch=torch.__version__,probly=importlib.metadata.version('probly'),
        backend_sha256=hashlib.sha256(backend.read_bytes()).hexdigest(),
        max_gradient=max(d['gradient'] for d in diagnostics),max_polish_l1=max(d['polish_l1'] for d in diagnostics),
        max_polish_nll_improvement=max(d['first_nll']-d['final_nll'] for d in diagnostics))
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    for ax,scenario in zip(axes,['clean','shared_mixture']):
        key=f'{scenario}_100000_0';center=saved[key+'_contour']
        ax.plot(center[:,0],center[:,1],'.',ms=1,label='ensemble density contour')
        ax.plot(analytic[:,0],analytic[:,1],label='clean left contour')
        if scenario=='shared_mixture':ax.plot(circle[:,0]+.2,circle[:,1],label='clean right contour')
        ax.set_aspect('equal');ax.set_title(scenario+' (n=100,000)');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'contours.png',dpi=160)


if __name__=='__main__':main()
