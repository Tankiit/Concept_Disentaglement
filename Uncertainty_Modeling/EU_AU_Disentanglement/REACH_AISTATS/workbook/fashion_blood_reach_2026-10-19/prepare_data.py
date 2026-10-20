from pathlib import Path
import urllib.request,json,hashlib,io
import numpy as np
import pandas as pd
from PIL import Image
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CACHE=ROOT/'cache/mnist_reach'

def main():
 repo='zalando-datasets/fashion_mnist'
 revision=json.load(urllib.request.urlopen(f'https://huggingface.co/api/datasets/{repo}'))['sha']
 files=json.load(urllib.request.urlopen(f'https://huggingface.co/api/datasets/{repo}/tree/{revision}/fashion_mnist'))
 info=next(x for x in files if '/test-' in x['path'])
 url=f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{info["path"]}'
 path=CACHE/'fashion_test.parquet'
 if not path.exists(): urllib.request.urlretrieve(url,path)
 sha=hashlib.sha256(path.read_bytes()).hexdigest();assert sha==info['lfs']['oid']
 df=pd.read_parquet(path);col='image' if 'image' in df else 'img'
 X=np.stack([np.array(Image.open(io.BytesIO(v['bytes'])).convert('L')) for v in df[col]])
 y=df.label.to_numpy(dtype=np.int64);assert X.shape==(10000,28,28)
 np.savez_compressed(CACHE/'fashion_test.npz',X=X,y=y)
 manifest={'fashion':dict(url=url,sha256=sha,rows=len(y),preprocessing='native grayscale /255 in model')}
 url='https://zenodo.org/records/10519652/files/bloodmnist.npz?download=1'
 path=CACHE/'bloodmnist_original.npz'
 if not path.exists(): urllib.request.urlretrieve(url,path)
 md5=hashlib.md5(path.read_bytes()).hexdigest();assert md5=='7053d0359d879ad8a5505303e11de1dc'
 with np.load(path,allow_pickle=False) as a:
  rgb=a['test_images'];y=a['test_labels'].reshape(-1).astype(np.int64)
 assert rgb.shape==(3421,28,28,3)
 X=np.stack([np.array(Image.fromarray(im).convert('L')) for im in rgb])
 np.savez_compressed(CACHE/'blood_test.npz',X=X,y=y)
 manifest['blood']=dict(url=url,md5=md5,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rows=len(y),preprocessing='Pillow RGB to L, no inversion or contrast normalization; /255 in model',metadata='https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py',license='CC BY 4.0')
 (HERE/'data_provenance.json').write_text(json.dumps(manifest,indent=2))
 print('Prepared Fashion-MNIST 10000 and BloodMNIST 3421 test images.',flush=True)
if __name__=='__main__':main()
