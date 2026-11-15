"""Independent backend comparison across all saved fitted mixture components."""
import json
import numpy as np
import torch
from scipy.special import ndtr
import run as r
maximum=0.;zero_entries=0;members=0;min_occupied=1.
counts=np.load(r.OUT.parent/'09_neural_density_ensemble/fits.npz')
for cfg in r.CONFIGS:
 archive=np.load(r.OUT/(cfg[0]+'_fits.npz'))
 for key in archive.files:
  if not key.endswith('_params'):continue
  for p in archive[key]:
   k=len(p);members+=1
   z=(r.r.prev.EDGES[None,None,:]-p[:,1:3,None])/p[:,3,None,None]
   marginal=np.where(z[:,:,:-1]>=0,ndtr(-z[:,:,:-1])-ndtr(-z[:,:,1:]),ndtr(z[:,:,1:])-ndtr(z[:,:,:-1]))
   cell=(marginal[:,1,:,None]*marginal[:,0,None,:]).reshape(k,36)
   zero_entries+=int((cell==0).sum())
   assert np.all(cell>=0)
   np.testing.assert_allclose(cell.sum(1),1,atol=1e-12)
   reference=(p[:,0,None]*cell).sum(0)
   means=torch.tensor(p[:,1:3]);s=torch.tensor(p[:,3])
   cdf=torch.special.ndtr((r.r.CUTS[None,None,:]-means[:,:,None])/s[:,None,None])
   cdf=torch.cat([torch.zeros((k,2,1)),cdf,torch.ones((k,2,1))],dim=-1)
   q=cdf.diff(dim=-1)
   observed=(torch.tensor(p[:,0,None,None])*q[:,1,:,None]*q[:,0,None,:]).sum(0).reshape(36).numpy()
   np.testing.assert_allclose(observed.sum(),1,atol=1e-12)
   occupied=counts[key[:-7]+'_counts']>0
   min_occupied=min(min_occupied,float(observed[occupied].min()))
   assert np.all(observed[occupied]>1e-100)
   maximum=max(maximum,float(abs(reference-observed).max()))
assert members==400 and maximum<1e-12
result=dict(members=members,max_probability_difference=maximum,underflow_zero_component_cells=zero_entries,min_probability_in_occupied_cell=min_occupied)
(r.OUT/'probability_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(result)
