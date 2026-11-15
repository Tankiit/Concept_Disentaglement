"""Score cached test images with the existing frozen model and partition."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
BASE=HERE.parent/'mnist_reach_novelty_2026-10-19'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(BASE))
from run_reach import DigitNet,tensor,data
from real.head import LDAHead


def main():
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    net=DigitNet().to(device);net.load_state_dict(torch.load(BASE/'digit_net.pt',map_location=device,weights_only=True))
    net.eval();net.requires_grad_(False)
    with np.load(BASE/'partition.npz') as a:head=LDAHead(**{k:a[k] for k in a.files})
    rows=[];hashes={}
    for name in ['mnist','fashion','blood']:
        X,y=data(name,'test');features=[]
        with torch.no_grad():
            for start in range(0,len(X),256):features.append(net.features(tensor(X[start:start+256],device)).cpu().numpy())
        cells=head.cell(np.concatenate(features))
        rows.extend(dict(dataset=name,index=i,label=int(y[i]),cell=int(cells[i])) for i in range(len(y)))
        path=ROOT/f'cache/mnist_reach/{name}_test.npz'
        hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        print(name,len(y),flush=True)
    for p in [BASE/'digit_net.pt',BASE/'partition.npz']:hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    pd.DataFrame(rows).to_csv(HERE/'image_cells.csv',index=False)
    (HERE/'input_hashes.json').write_text(json.dumps(hashes,indent=2))


if __name__=='__main__':main()
