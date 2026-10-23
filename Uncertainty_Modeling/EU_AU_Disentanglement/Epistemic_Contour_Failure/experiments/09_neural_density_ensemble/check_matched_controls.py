"""Gaussian bootstrap control on the exact neural-pilot datasets."""
import csv
import numpy as np
import run as r
saved=np.load(r.OUT/'fits.npz');rng=np.random.default_rng(r.SEED+1);rows=[]
for key in saved.files:
    if not key.endswith('_counts'):continue
    prefix=key[:-7];count=saved[key];n=int(count.sum())
    estimate,_,_=r.prev.fit(count[None]);estimate=estimate[0]
    boot,_,_=r.prev.fit(rng.multinomial(n,count/n,size=199),start=estimate)
    center=estimate.copy();center[2]=np.exp(center[2]);members=boot.copy();members[:,2]=np.exp(members[:,2])
    distance=lambda a,b: np.linalg.norm(a[...,:2]-b[...,:2],axis=-1)+r.prev.C*abs(a[...,2]-b[...,2])
    radius=float(np.quantile(distance(members,center),.95))
    truths=[np.array([-.2,0,1])]
    if prefix.startswith('shared'):truths.append(np.array([.2,0,1]))
    lower,upper=r.prev.f.cp(count,n,.05/36)
    for j,truth in enumerate(truths):
        error=float(distance(center,truth))
        compatible=bool(r.prev.f.feasible(r.prev.laws(truth[None]),lower,upper,.2 if prefix.startswith('shared') else 0)[0])
        rows.append(dict(dataset=prefix,truth=j,mi_nats=r.mi(r.prev.laws(members)),contour_error=error,
            bootstrap_radius=radius,bootstrap_contains=int(error<=radius),cp_projection_contains=int(compatible)))
with (r.OUT/'matched_controls.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('matched controls completed',len(rows),'truth evaluations')
