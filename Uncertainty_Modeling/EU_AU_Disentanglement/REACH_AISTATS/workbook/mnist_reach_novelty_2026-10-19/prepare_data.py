from pathlib import Path
import urllib.request, json, hashlib, io
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CACHE=ROOT/'cache/mnist_reach'
SOURCES={'mnist':('ylecun/mnist','mnist',['train','test']),
         'letters':('tanganke/emnist_letters','emnist-letters',['train','test']),
         'kmnist':('tanganke/kmnist','kmnist',['test'])}

def main():
 CACHE.mkdir(parents=True,exist_ok=True)
 manifest={}
 for name,(repo,folder,splits) in SOURCES.items():
  meta=json.load(urllib.request.urlopen(f'https://huggingface.co/api/datasets/{repo}'))
  revision=meta['sha']
  files=json.load(urllib.request.urlopen(f'https://huggingface.co/api/datasets/{repo}/tree/{revision}/{folder}'))
  for split in splits:
   info=next(x for x in files if f'/{split}-' in x['path'])
   path=CACHE/f'{name}_{split}.parquet'
   url=f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{info["path"]}'
   if not path.exists(): urllib.request.urlretrieve(url,path)
   digest=hashlib.sha256(path.read_bytes()).hexdigest()
   assert digest==info['lfs']['oid']
   df=pd.read_parquet(path)
   col='image' if 'image' in df else 'img'
   X=np.stack([np.array(Image.open(io.BytesIO(x['bytes'])).convert('L')) for x in df[col]])
   y=df.label.to_numpy(dtype=np.int64)
   assert X.shape[1:]==(28,28)
   np.savez_compressed(CACHE/f'{name}_{split}.npz',X=X,y=y)
   manifest[f'{name}_{split}']=dict(url=url,sha256=digest,rows=len(y),classes=np.unique(y).tolist(),shape=list(X.shape))
   print(name,split,X.shape,np.unique(y),flush=True)
 (HERE/'data_provenance.json').write_text(json.dumps(manifest,indent=2))
 plt.rcdefaults()
 fig,axes=plt.subplots(3,10,figsize=(12,4))
 for row,name in enumerate(SOURCES):
  a=np.load(CACHE/f'{name}_test.npz')
  for col in range(10):
   j=np.flatnonzero(a['y']==col)[0]
   axes[row,col].imshow(a['X'][j],cmap='gray',vmin=0,vmax=255)
   axes[row,col].set_title(f'{name}: {col}',fontsize=8);axes[row,col].axis('off')
 fig.tight_layout();fig.savefig(HERE/'input_examples.png',dpi=130)
if __name__=='__main__': main()
