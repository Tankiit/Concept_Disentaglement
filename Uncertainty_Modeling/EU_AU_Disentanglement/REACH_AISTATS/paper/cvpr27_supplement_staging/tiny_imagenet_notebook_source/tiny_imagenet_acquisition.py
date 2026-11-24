#!/usr/bin/env python3
"""Tiny ImageNet: image-conditioned, heterogeneous information acquisition.

Subcommands
-----------
train    Train a 64x64 ResNet18 (no external ImageNet pretraining).
cache    Freeze classifier predictions/features on strictly disjoint fit/eval images.
acquire  Run conjugate, image-cell-conditional acquisition and held-out checks.

Observation design at a fixed fitted cell c:
  identity : source S from the unlabeled deployment population conditional on C=c
  label_s  : classifier error Y from a source-known reference reservoir (S=s,C=c)

T(c) = P(classifier makes an error | C=c) = sum_s pi_{s|c} theta_{s,c}
with pi_c ~ Dirichlet(alpha_c), theta_{s,c} ~ Beta(a_{sc}, b_{sc}).
Only acquired fields update these distributions. Evaluation labels are never used
for state construction, action selection, or hyperparameter selection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

SOURCES = ("clean", "blur", "noise")


def stable_seed(seed: int, item: str) -> int:
    digest = hashlib.sha256(f"{seed}|{item}".encode("utf8")).digest()
    return int.from_bytes(digest[:8], "little")


def parse_source_weights(value: str) -> np.ndarray:
    w = np.asarray([float(v) for v in value.split(",")], dtype=float)
    if w.shape != (3,) or not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError("--weights must contain three strictly positive numbers")
    return w / w.sum()


def split_manifest(root: Path, seed: int, holdout_per_class: int = 100):
    """Hold out original train images by label, BEFORE any corruption is made."""
    root = Path(root)
    wnids = [v.strip() for v in (root / "wnids.txt").read_text().splitlines() if v.strip()]
    class_to_id = {w: i for i, w in enumerate(wnids)}
    if len(wnids) != 200:
        raise ValueError(f"Expected 200 Tiny ImageNet classes; got {len(wnids)}")
    train_model, fit = [], []
    for wnid in wnids:
        paths = sorted((root / "train" / wnid / "images").glob("*.JPEG"))
        if len(paths) != 500:
            raise ValueError(f"Expected 500 train examples for {wnid}; got {len(paths)}")
        rng = np.random.default_rng(stable_seed(seed, wnid))
        order = rng.permutation(len(paths))
        model_idx, fit_idx = order[:-holdout_per_class], order[-holdout_per_class:]
        train_model += [(str(paths[k]), class_to_id[wnid]) for k in model_idx]
        fit += [(str(paths[k]), class_to_id[wnid]) for k in fit_idx]
    ann = root / "val" / "val_annotations.txt"
    if not ann.exists():
        raise FileNotFoundError(f"Missing validation labels: {ann}")
    val = []
    for line in ann.read_text().splitlines():
        name, wnid, *_ = line.split("\t")
        if wnid not in class_to_id:
            raise ValueError(f"Unknown validation class {wnid}")
        p = root / "val" / "images" / name
        if not p.exists():
            raise FileNotFoundError(p)
        val.append((str(p), class_to_id[wnid]))
    if len(val) != 10000:
        raise ValueError(f"Expected 10k validation images, got {len(val)}")
    return train_model, fit, val, wnids


def deterministic_source(seed: int, path: str, weights: np.ndarray) -> int:
    u = stable_seed(seed, path) / float(2**64)
    return min(2, int(np.searchsorted(np.cumsum(weights), u, side="right")))


def resnet18_64(num_classes: int = 200):
    import torch
    from torch import nn
    from torchvision.models import resnet18
    net = resnet18(weights=None, num_classes=num_classes)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    return net


def _make_transform(train: bool):
    from torchvision import transforms
    steps = []
    if train:
        steps += [transforms.RandomCrop(64, padding=4), transforms.RandomHorizontalFlip()]
    steps += [transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406),
                                                         (0.229, 0.224, 0.225))]
    return transforms.Compose(steps)


def corrupt(image, source: int, seed: int, path: str):
    from PIL import ImageFilter
    if source == 1:
        return image.filter(ImageFilter.GaussianBlur(radius=1.3))
    if source == 2:
        # 64x64 input, deterministic per path and source. No new data leakage.
        arr = np.asarray(image, dtype=np.float32)
        rng = np.random.default_rng(stable_seed(seed + 947, path))
        arr = np.clip(arr + rng.normal(0.0, 30.0, arr.shape), 0, 255).astype(np.uint8)
        from PIL import Image
        return Image.fromarray(arr, mode="RGB")
    return image


class TinyImages:
    def __init__(self, examples, train, seed, weights=None):
        self.examples = examples
        self.transform = _make_transform(train)
        self.seed = seed
        self.sources = [0 if weights is None else deterministic_source(seed, p, weights)
                        for p, _ in examples]
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, i):
        from PIL import Image
        p, label = self.examples[i]
        with Image.open(p) as im:
            x = im.convert("RGB")
        s = self.sources[i]
        if s:
            x = corrupt(x, s, self.seed, p)
        return self.transform(x), int(label), int(s)


def cmd_train(args):
    import torch
    from torch.utils.data import DataLoader
    from torch.nn import functional as F
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    train_examples, _, _, wnids = split_manifest(Path(args.data), args.seed, args.holdout)
    train = DataLoader(TinyImages(train_examples, True, args.seed), batch_size=args.batch,
                       shuffle=True, num_workers=args.workers, pin_memory=True,
                       persistent_workers=args.workers > 0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = resnet18_64(len(wnids)).to(device)
    opt = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,
                          nesterov=True, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    if args.resume and output.exists():
        previous=torch.load(output,map_location='cpu',weights_only=False)
        if (previous.get('seed') != args.seed or
            previous.get('holdout_per_class') != args.holdout or
            previous.get('classes') != wnids or
            previous.get('total_epochs') != args.epochs):
            raise ValueError('Resume checkpoint has incompatible split/classes/total_epochs')
        if not all(k in previous for k in ('optimizer','scheduler','scaler','completed_epochs')):
            raise ValueError('Resume checkpoint does not contain full optimizer state')
        net.load_state_dict(previous['model'])
        net.to(device)
        opt.load_state_dict(previous['optimizer'])
        for state in opt.state.values():
            for key,value in state.items():
                if isinstance(value,torch.Tensor):state[key]=value.to(device)
        scheduler.load_state_dict(previous['scheduler'])
        scaler.load_state_dict(previous['scaler'])
        start_epoch=int(previous['completed_epochs'])
        print(f'Resuming training from epoch {start_epoch}/{args.epochs}',flush=True)
    elif output.exists() and not args.resume:
        raise FileExistsError('Checkpoint exists. Set --resume or a fresh --output path.')
    if start_epoch>=args.epochs:
        print(f'Checkpoint already complete at {start_epoch} epochs: {output}')
        return
    for epoch in range(start_epoch,args.epochs):
        net.train(); total=0; good=0; loss_total=0.
        for x,y,_ in train:
            x,y=x.to(device, non_blocking=True),y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = net(x); loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            total += len(y); good += (logits.argmax(1) == y).sum().item()
            loss_total += loss.detach().item() * len(y)
        scheduler.step()
        print(f"epoch={epoch+1}/{args.epochs} train_loss={loss_total/total:.4f} "
              f"train_accuracy={good/total:.4f}", flush=True)
        if (epoch+1) % args.save_every == 0 or epoch+1==args.epochs:
            payload={"model":net.state_dict(),"optimizer":opt.state_dict(),
                     "scheduler":scheduler.state_dict(),"scaler":scaler.state_dict(),
                     "seed":args.seed,"holdout_per_class":args.holdout,
                     "classes":wnids,"epochs":args.epochs,"total_epochs":args.epochs,
                     "completed_epochs":epoch+1,
                     "source":"trained from scratch on Tiny ImageNet"}
            tmp=output.with_suffix('.temporary.pt')
            torch.save(payload,tmp)
            tmp.replace(output)
            print(f"Checkpoint saved at epoch {epoch+1}: {output}",flush=True)
    print(f"Training finished: {output}")


def resnet_features(net, x):
    x = net.conv1(x); x = net.bn1(x); x = net.relu(x); x = net.maxpool(x)
    x = net.layer1(x); x = net.layer2(x); x = net.layer3(x); x = net.layer4(x)
    x = net.avgpool(x)
    return x.flatten(1)


def cmd_cache(args):
    import torch
    from torch.utils.data import DataLoader
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_examples, fit_examples, val_examples, wnids = split_manifest(Path(args.data),args.seed,args.holdout)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["seed"] != args.seed or checkpoint["holdout_per_class"] != args.holdout:
        raise ValueError("Checkpoint train/fit split differs from requested seed/holdout")
    if checkpoint["classes"] != wnids:
        raise ValueError("Checkpoint classes differ from dataset")
    weights = parse_source_weights(args.weights)
    net = resnet18_64(200)
    net.load_state_dict(checkpoint["model"])
    net = net.to(device).eval()

    def get_cache(examples, name):
        ds = TinyImages(examples, False, args.seed + 1009, weights)
        loader=DataLoader(ds,batch_size=args.batch,shuffle=False,num_workers=args.workers,
                          pin_memory=True, persistent_workers=args.workers > 0)
        features=[]; probabilities=[]; sources=[]; labels=[]; predictions=[]
        with torch.inference_mode():
            for x,y,s in loader:
                x=x.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    feats=resnet_features(net,x)
                    logits=net.fc(feats)
                p=logits.float().softmax(1)
                features.append(feats.float().cpu().numpy()); probabilities.append(p.cpu().numpy())
                sources.extend(s.numpy()); labels.extend(y.numpy()); predictions.extend(p.argmax(1).cpu().numpy())
        features=np.concatenate(features).astype(np.float32)
        probabilities=np.concatenate(probabilities).astype(np.float32)
        labels=np.asarray(labels, dtype=np.int16); sources=np.asarray(sources,dtype=np.int8)
        predictions=np.asarray(predictions,dtype=np.int16)
        errors=(predictions != labels).astype(np.int8)
        print(f"{name}: n={len(labels)}, accuracy={1-errors.mean():.4f}; source counts="
              f"{np.bincount(sources,minlength=3).tolist()}",flush=True)
        return {f"{name}_features":features, f"{name}_scores":probabilities,
                f"{name}_source":sources, f"{name}_label":labels,
                f"{name}_pred":predictions, f"{name}_error":errors}

    arrays={**get_cache(fit_examples,"fit"),**get_cache(val_examples,"eval")}
    arrays["sources_names"]=np.array(SOURCES)
    arrays["source_weights"]=weights
    arrays["seed"]=np.asarray(args.seed)
    arrays["holdout_per_class"]=np.asarray(args.holdout)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,**arrays)
    print(f"Saved frozen, labeled simulation cache: {output}")


def posterior_stats(alpha: np.ndarray, pos: np.ndarray, neg: np.ndarray):
    """Exact moments of T=sum_s pi_s theta_s under Dirichlet x independent Betas."""
    alpha, pos, neg = [np.asarray(v, dtype=float) for v in (alpha,pos,neg)]
    if (alpha<=0).any() or (pos<=0).any() or (neg<=0).any():
        raise ValueError("Dirichlet/Beta shape parameters must be positive")
    A=alpha.sum(); n=pos+neg; wm=alpha/A; tm=pos/n
    tv=tm*(1-tm)/(n+1)
    wm2=np.outer(alpha,alpha) / (A*(A+1))
    wm2[np.diag_indices_from(wm2)] += alpha/(A*(A+1))
    tm2=np.outer(tm,tm)
    tm2[np.diag_indices_from(tm2)] += tv
    mu=float(wm@tm)
    var=float(max(0.,np.sum(wm2*tm2)-mu*mu))
    identity_value=float(np.sum(wm*(tm-mu)**2)/(A+1)**2)
    label_values=(wm**2)*tm*(1-tm)/(n+1)**2
    return mu,var,identity_value,label_values


def one_step_empirical(alpha, pos, neg, source_pool, label_pools, heldout_truth):
    """Exact one-step fixed-population MSE over recorded fit-pool observation law."""
    mu, var, vid, vl = posterior_stats(alpha,pos,neg)
    initial=(mu-heldout_truth)**2
    out={"initial_mse":initial,"initial_posterior_variance":var,
         "modeled_identity_value":vid}
    for action in ["identity", "label_0", "label_1", "label_2"]:
        if action=="identity":
            qs=np.bincount(source_pool,minlength=3)/len(source_pool)
            mus=[]
            for s in range(3):
                nxt=alpha.copy();nxt[s]+=1
                mus.append(posterior_stats(nxt,pos,neg)[0])
            mse=float(np.sum(qs*(np.asarray(mus)-heldout_truth)**2))
        else:
            s=int(action.split("_")[1]);rate=float(np.mean(label_pools[s]))
            pos1=pos.copy();pos1[s]+=1
            neg0=neg.copy();neg0[s]+=1
            mu1=posterior_stats(alpha,pos1,neg)[0]
            mu0=posterior_stats(alpha,pos,neg0)[0]
            mse=float(rate*(mu1-heldout_truth)**2+(1-rate)*(mu0-heldout_truth)**2)
            out[f"modeled_{action}_value"]=float(vl[s])
        out[f"heldout_{action}_mse"]=mse
        out[f"heldout_{action}_delta"]=initial-mse
    return out


def choose_action(policy, alpha, pos, neg, rng, label_cost):
    _,_,vI,vL=posterior_stats(alpha,pos,neg)
    if policy=="identity_only":
        return "identity"
    if policy=="label_only":
        return f"label_{int(np.argmax(vL))}"
    if policy=="random":
        return "identity" if rng.random()<0.5 else f"label_{int(rng.integers(0,3))}"
    if policy=="adaptive":
        best=int(np.argmax(vL))
        # maximize expected Bayes-risk reduction per unit acquisition cost
        return "identity" if vI >= (vL[best]/label_cost) else f"label_{best}"
    raise ValueError(policy)


def _write_csv(path, rows):
    if not rows: return
    with Path(path).open("w",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)


def cmd_acquire(args):
    from sklearn.cluster import MiniBatchKMeans
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    d=np.load(args.cache,allow_pickle=False)
    feat_fit=d["fit_features"]; feat_eval=d["eval_features"]
    src_fit=d["fit_source"]; src_eval=d["eval_source"]
    err_fit=d["fit_error"]; err_eval=d["eval_error"]
    # KMeans is fitted exclusively on fit images; never on eval features.
    km=MiniBatchKMeans(n_clusters=args.cells,random_state=args.seed,n_init=5,batch_size=1024)
    c_fit=km.fit_predict(feat_fit)
    c_eval=km.predict(feat_eval)
    policies=["adaptive","identity_only","label_only","random"]
    budgets=sorted(set(int(s) for s in args.budgets.split(",")))
    if budgets[0] != 0 or budgets[-1] < 1:
        raise ValueError("Budgets must include 0 and a positive step")
    one_steps=[]; trajectories=[]; audits=[]
    max_budget=budgets[-1]
    for c in range(args.cells):
        pool_source=src_fit[c_fit==c]
        pool_error=err_fit[c_fit==c]
        eval_err=err_eval[c_eval==c]
        eval_src=src_eval[c_eval==c]
        label_pools=[pool_error[pool_source==s] for s in range(3)]
        q=np.bincount(pool_source,minlength=3).astype(float)/max(1,len(pool_source))
        qeval=np.bincount(eval_src,minlength=3).astype(float)/max(1,len(eval_src))
        audits.append({"cell":c,"fit_n":len(pool_source),"eval_n":len(eval_err),
                        "fit_by_source":np.bincount(pool_source,minlength=3).tolist(),
                        "eval_by_source":np.bincount(eval_src,minlength=3).tolist(),
                        "fit_p_source_given_cell":q.tolist(),
                        "eval_p_source_given_cell":qeval.tolist(),
                        "fit_error_rate":float(pool_error.mean()) if len(pool_error) else None,
                        "eval_error_rate":float(eval_err.mean()) if len(eval_err) else None})
        if len(eval_err)<args.min_cell or min(map(len,label_pools))<args.min_stratum:
            print(f"Skipping cell {c}: insufficient independent fit/eval records",flush=True)
            continue
        truth=float(eval_err.mean())
        for rep in range(args.repeats):
            seed=stable_seed(args.seed+rep,f"cell{c}")
            rng=np.random.default_rng(seed)
            base_id = rng.choice(pool_source,size=args.baseline_id,replace=True)
            alpha = np.ones(3,dtype=float)+np.bincount(base_id,minlength=3)
            pos=np.ones(3,dtype=float)
            neg=np.ones(3,dtype=float)
            for s in range(3):
                obs=rng.choice(label_pools[s],size=args.baseline_labels,replace=True)
                pos[s]+=int(obs.sum());neg[s]+=int(len(obs)-obs.sum())
            exact=one_step_empirical(alpha,pos,neg,pool_source,label_pools,truth)
            one_steps.append({"cell":c,"repeat":rep,"heldout_target":truth,**exact})
            stream_id=rng.choice(pool_source,size=max_budget,replace=True)
            stream_label=[rng.choice(v,size=max_budget,replace=True) for v in label_pools]
            for policy in policies:
                pa,pb,pc=alpha.copy(),pos.copy(),neg.copy()
                offsets=np.zeros(4,dtype=int)
                action_rng=np.random.default_rng(stable_seed(int(seed%2**32),policy))
                for step in range(max_budget+1):
                    if step in budgets:
                        mu,var,VI,VL=posterior_stats(pa,pb,pc)
                        trajectories.append({"cell":c,"repeat":rep,"policy":policy,
                                             "budget":step,"posterior_mean":mu,
                                             "posterior_variance":var,"heldout_target":truth,
                                             "heldout_mse":(mu-truth)**2,
                                             "identities_acquired":int(offsets[0]),
                                             "labels_0_acquired":int(offsets[1]),
                                             "labels_1_acquired":int(offsets[2]),
                                             "labels_2_acquired":int(offsets[3]),
                                             "eval_cell_n":len(eval_err)})
                    if step==max_budget: break
                    action=choose_action(policy,pa,pb,pc,action_rng,args.label_cost)
                    if action=="identity":
                        s=int(stream_id[offsets[0]]);pa[s]+=1;offsets[0]+=1
                    else:
                        s=int(action.split("_")[1]);j=offsets[s+1]
                        err=int(stream_label[s][j]);offsets[s+1]+=1
                        if err:pb[s]+=1
                        else:pc[s]+=1
    if not trajectories:
        raise RuntimeError("No populated cells; reduce --min-stratum or --cells")
    _write_csv(output/"trajectories.csv",trajectories)
    _write_csv(output/"exact_one_step.csv",one_steps)
    (output/"audit.json").write_text(json.dumps({"cells":audits,
        "train_fit_eval_are_image_disjoint":True,
        "sources":SOURCES,"label_cost":args.label_cost,
        "observation_design":"identity conditional on C; label from known source/C reference pool",
        "prior":"Dirichlet(1,1,1) and independent Beta(1,1)"},indent=2))
    # Repeated trajectory summary, macro-averaging over retained cells and repeats.
    grouped=defaultdict(list)
    for row in trajectories:
        grouped[(row["policy"],row["budget"])].append(row)
    summ=[]
    for (policy,budget),rows in sorted(grouped.items()):
        mse=np.array([v["heldout_mse"] for v in rows])
        pv=np.array([v["posterior_variance"] for v in rows])
        weights=np.array([v["eval_cell_n"] for v in rows],dtype=float)
        summ.append({"policy":policy,"budget":budget,"mean_heldout_mse":float(mse.mean()),
                     "eval_weighted_mse":float(np.average(mse,weights=weights)),
                     "mean_posterior_variance":float(pv.mean()),
                     "n_cell_repeats":len(rows)})
    _write_csv(output/"summary.csv",summ)
    fig,axes=plt.subplots(1,2,figsize=(11,4.1))
    for policy in policies:
        rows=sorted((r for r in summ if r["policy"]==policy),key=lambda r:r["budget"])
        x=[r["budget"] for r in rows]
        axes[0].plot(x,[r["eval_weighted_mse"] for r in rows],marker="o",label=policy)
        axes[1].plot(x,[r["mean_posterior_variance"] for r in rows],marker="o",label=policy)
    axes[0].set_title("Held-out conditional error-rate MSE")
    axes[1].set_title("Model posterior variance")
    for ax in axes:
        ax.set_xlabel("Acquired fields (nominal)")
        ax.set_xscale("symlog",linthresh=8);ax.grid(alpha=.2)
    axes[0].set_ylabel("MSE (evaluation-cell-size weighted)")
    axes[1].set_ylabel("Posterior variance (macro over cells)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output/"acquisition_curves.png",dpi=180)
    plt.close(fig)
    print(f"Experiment artifacts in {output}")
    for row in summ:
        if row["budget"]==max_budget:
            print(row)


def build_parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    sub=p.add_subparsers(dest="command",required=True)
    tr=sub.add_parser("train",help="Train clean Tiny ImageNet classifier from scratch")
    tr.add_argument("--data",required=True);tr.add_argument("--output",default="tiny_resnet18.pt")
    tr.add_argument("--seed",type=int,default=20261111);tr.add_argument("--holdout",type=int,default=100)
    tr.add_argument("--epochs",type=int,default=40);tr.add_argument("--batch",type=int,default=128)
    tr.add_argument("--workers",type=int,default=4);tr.add_argument("--lr",type=float,default=0.1)
    tr.add_argument("--resume",action="store_true");tr.add_argument("--save-every",type=int,default=1)
    tr.set_defaults(func=cmd_train)
    ca=sub.add_parser("cache",help="Freeze features/predictions for disjoint fit and held-out images")
    ca.add_argument("--data",required=True);ca.add_argument("--checkpoint",required=True)
    ca.add_argument("--output",default="tiny_frozen.npz")
    ca.add_argument("--seed",type=int,default=20261111);ca.add_argument("--holdout",type=int,default=100)
    ca.add_argument("--weights",default="0.5,0.3,0.2")
    ca.add_argument("--batch",type=int,default=128);ca.add_argument("--workers",type=int,default=4)
    ca.set_defaults(func=cmd_cache)
    ac=sub.add_parser("acquire",help="Run conjugate acquisition with paired held-out evaluation")
    ac.add_argument("--cache",required=True);ac.add_argument("--output",default="tiny_acquisition_run")
    ac.add_argument("--seed",type=int,default=20261111);ac.add_argument("--cells",type=int,default=4)
    ac.add_argument("--repeats",type=int,default=20);ac.add_argument("--baseline-id",type=int,default=8)
    ac.add_argument("--baseline-labels",type=int,default=8)
    ac.add_argument("--budgets",default="0,8,32,128,512")
    ac.add_argument("--min-stratum",type=int,default=20);ac.add_argument("--min-cell",type=int,default=100)
    ac.add_argument("--label-cost",type=float,default=1.0)
    ac.set_defaults(func=cmd_acquire)
    return p

if __name__=="__main__":
    parser=build_parser();args=parser.parse_args()
    if getattr(args,"label_cost",1.0)<=0:parser.error("--label-cost must be positive")
    args.func(args)