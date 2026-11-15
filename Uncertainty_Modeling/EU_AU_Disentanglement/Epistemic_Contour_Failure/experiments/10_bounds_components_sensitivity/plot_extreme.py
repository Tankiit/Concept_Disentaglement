"""Visualize the largest clean ensemble-contour error without hiding disconnected boundaries."""
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import run as r
rows=list(csv.DictReader((r.OUT/'rows.csv').open()))
case=max((x for x in rows if x['scenario']=='clean'),key=lambda x:float(x['contour_error']))
archive=np.load(r.OUT/(case['config']+'_fits.npz'));key=case['dataset']
params=archive[key+'_params'];center=archive[key+'_contour']
fig,ax=plt.subplots(figsize=(6,5))
for j,p in enumerate(params):
 c,_=r.contour(p)
 ax.scatter(c[:,0],c[:,1],s=.3,color='grey',alpha=.4,label='member contours' if j==0 else None)
ax.scatter(center[:,0],center[:,1],s=1,color='tab:blue',label='ensemble density contour')
t=np.linspace(0,2*np.pi,1000)
ax.plot(-.2+r.r.prev.C*np.cos(t),r.r.prev.C*np.sin(t),color='tab:orange',label='true clean contour')
ax.set_aspect('equal');ax.legend();ax.set_title(case['config']+' / '+key)
fig.tight_layout();fig.savefig(r.OUT/'extreme_clean_contours.png',dpi=160)
# Check the largest clean error at finer resolution, beyond the first-dataset checks.
combined=params.reshape(-1,4).copy();combined[:,0]/=5
fine,_=r.contour(combined,h=.02)
case['fine_grid_contour_change']=float(r.r.hd(center,fine))
(r.OUT/'extreme_case.json').write_text(json.dumps(case,indent=2)+'\n')
print(case)
