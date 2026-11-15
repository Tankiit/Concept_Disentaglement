"""Train a fresh 0--5 classifier and fit a partition without digits 6--9."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
import torch
from torch import nn

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT),str(HERE.parent/'mnist_reach_novelty_2026-10-19')]
from run_reach import DigitNet, data, tensor
from real.head import LDAHead


def main():
    torch.set_num_threads(4);torch.manual_seed(20261021)
    rng=np.random.default_rng(20261021)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    X,y=data('mnist','train')
    fit,dev=[],[]
    for c in range(6):
        ids=rng.permutation(np.flatnonzero(y==c))
        fit.extend(ids[:3000]);dev.extend(ids[3000:4000])
    fit,dev=np.array(fit),np.array(dev)
    assert not set(fit)&set(dev)
    assert max(y[fit])==5 and max(y[dev])==5
    net=DigitNet();net.output=nn.Linear(64,6);net.to(device)
    opt=torch.optim.Adam(net.parameters(),lr=.001)
    training=[]
    for epoch in range(5):
        net.train();order=rng.permutation(fit);loss_sum=0
        for start in range(0,len(order),256):
            ix=order[start:start+256]
            opt.zero_grad();loss=nn.functional.cross_entropy(net(tensor(X[ix],device)),torch.from_numpy(y[ix]).to(device))
            loss.backward();opt.step();loss_sum+=loss.item()*len(ix)
        training.append(dict(epoch=epoch+1,loss=loss_sum/len(fit)))
        print(training[-1],flush=True)
    net.eval();net.requires_grad_(False)
    def encode(images):
        with torch.no_grad():
            return np.concatenate([net.features(tensor(images[s:s+256],device)).cpu().numpy() for s in range(0,len(images),256)])
    head=LDAHead.fit(encode(X[dev]),y[dev],np.arange(6),dim=16)
    xt,yt=data('mnist','test');f=encode(xt);cells=head.cell(f)
    with torch.no_grad():
        logits=net.output(torch.from_numpy(f).to(device));prob=logits.softmax(1)
        pred=prob.argmax(1).cpu().numpy();confidence=prob.max(1).values.cpu().numpy()
    laws=np.stack([np.bincount(cells[yt==c],minlength=7)/(yt==c).sum() for c in range(10)])
    pd.DataFrame(dict(index=np.arange(len(yt)),label=yt,cell=cells,prediction=pred,confidence=confidence)).to_csv(HERE/'heldout_predictions.csv',index=False)
    pd.DataFrame(training).to_csv(HERE/'training.csv',index=False)
    pd.DataFrame([dict(index=int(i),label=int(y[i]),role=role) for role,ids in [('train',fit),('partition_dev',dev)] for i in ids]).to_csv(HERE/'training_assignments.csv',index=False)
    np.savez(HERE/'heldout_source_laws.npz',laws=laws)
    np.savez(HERE/'heldout_partition.npz',**head.__dict__)
    torch.save({k:v.cpu() for k,v in net.state_dict().items()},HERE/'heldout_digit_net.pt')
    diagnostics=dict(seed=20261021,train_images=len(fit),partition_images=len(dev),epochs=5,
        train_classes=list(range(6)),heldout_classes=list(range(6,10)),device=device,
        known_accuracy=float((pred[yt<6]==yt[yt<6]).mean()),
        known_outlier_fraction=float((cells[yt<6]==6).mean()),
        heldout_outlier_fraction=float((cells[yt>=6]==6).mean()),
        heldout_mean_confidence=float(confidence[yt>=6].mean()),
        heldout_confident_fraction=float((confidence[yt>=6]>.9).mean()),
        model_sha256=hashlib.sha256((HERE/'heldout_digit_net.pt').read_bytes()).hexdigest(),
        scope='Fresh six-output CNN; only 0--5 in training and partition fitting. Test digit laws used after freezing for finite-pool REACH simulations.')
    (HERE/'model_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    print(json.dumps(diagnostics,indent=2))


if __name__=='__main__':main()
