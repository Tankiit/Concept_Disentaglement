from pathlib import Path
import importlib.util,json
import numpy as np
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('comparison',P/'run.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
rng=np.random.default_rng(20261029);out=[]
for name,law,eps in [('clean',m.laws(np.array([[.173,-.137,.973]]))[0],0.),('shared',m.h.mixture(.2)[0],.2)]:
 counts=rng.multinomial(100000,law)
 t,_,_=m.fit(counts[None,:]);bs=rng.multinomial(100000,counts/100000,size=1999)
 tb,_,_=m.fit(bs,t[0]);dist=np.linalg.norm(tb[:,:2]-t[0,:2],axis=1)+m.C*abs(np.exp(tb[:,2])-np.exp(t[0,2]))
 L,U=m.f.cp(counts,100000,.05/36)
 refinement=[];previous=None
 for depth in [5,6,7]:
  lo,hi,width,rho=m.reference(L,U,eps,max_rounds=depth)
  refinement.append(dict(depth=depth,diameter_upper=width,box_radius=rho,boxes=len(lo)))
  if previous is not None:assert width<=previous+1e-12
  previous=width
 out.append(dict(scenario=name,bootstrap_radius_399=float(np.quantile(dist[:399],.95,method='higher')),bootstrap_radius_1999=float(np.quantile(dist,.95,method='higher')),refinement=refinement))
(P/'sensitivity.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
