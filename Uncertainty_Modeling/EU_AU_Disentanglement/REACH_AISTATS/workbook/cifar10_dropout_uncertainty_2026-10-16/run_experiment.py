"""Real CIFAR-10 test images, frozen ResNet-18, and an OSSIM dropout head."""
from pathlib import Path
import sys
import json
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from example import Head, train
from oss_from_cache import make_oss, make_loader


def entropy(p):
    return -(p * p.clamp_min(1e-12).log()).sum(-1)


def extract_features(device):
    import hashlib
    import io
    import urllib.request
    from PIL import Image
    import timm

    source = ROOT / 'cache/cifar10/test.parquet'
    expected = '841389e6f2d64f28bf17310e430aebac20ec3ba611a3c5e231dc93c645ce84de'
    url = 'https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/test-00000-of-00001.parquet'
    source.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        urllib.request.urlretrieve(url, source)
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        raise ValueError('CIFAR-10 mirror checksum mismatch')
    out = ROOT / 'cache/features/timm_resnet18_c10test.npz'
    if out.exists():
        print('Reusing feature cache:', out, flush=True)
        return
    frame = pd.read_parquet(source)
    labels = frame['label'].to_numpy(dtype=np.int64)
    assert len(frame) == 10000 and np.array_equal(np.bincount(labels), np.full(10, 1000))
    model = timm.create_model('resnet18', pretrained=True, num_classes=0).to(device).eval()
    transform = timm.data.create_transform(**timm.data.resolve_model_data_config(model))
    features = []
    with torch.no_grad():
        for start in range(0, len(frame), 128):
            images = [transform(Image.open(io.BytesIO(item['bytes'])).convert('RGB'))
                      for item in frame['img'].iloc[start:start+128]]
            features.append(model(torch.stack(images).to(device)).cpu().numpy())
            if start % 1280 == 0:
                print(f'Extracted {min(start+128, len(frame))}/{len(frame)} images', flush=True)
    X = np.concatenate(features).astype(np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, X=X, y=labels)
    provenance = dict(source_url=url, source_sha256=expected, rows=len(frame),
                      feature_shape=list(X.shape), timm_version=timm.__version__,
                      pretrained_config=model.pretrained_cfg)
    (HERE / 'feature_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print('Saved features:', X.shape, flush=True)


def main():
    import os
    os.chdir(ROOT)
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Feature extraction device:', device, flush=True)
    extract_features(device)
    path = ROOT / 'cache/features/timm_resnet18_c10test.npz'
    world = make_oss(path, seed=0)
    X, y = world['X'], world['y']
    classes = sorted(world['kkc'])
    lut = torch.full((int(y.max()) + 1,), -1, dtype=torch.long)
    for i, c in enumerate(classes):
        lut[c] = i
    idx = world['idx']['train']
    idx = idx[np.isin(y[idx], classes)]
    loader = make_loader(X, y, indices=idx, batch_size=256, shuffle=True, seed=0)
    torch.manual_seed(0)
    head = Head(d=X.shape[1], k=len(classes))
    losses = train(head, loader, lut, epochs=40, seed=0)
    print('Training loss:', losses[0], '->', losses[-1], flush=True)
    pd.DataFrame({'epoch': np.arange(1, 41), 'loss': losses}).to_csv(HERE / 'training.csv', index=False)
    torch.save(head.state_dict(), HERE / 'head.pt')
    rows = []
    torch.manual_seed(123)
    for split in ('val', 'test'):
        loader = make_loader(X, y, indices=world['idx'][split], batch_size=256)
        for b in loader:
            head.eval()
            with torch.no_grad():
                deterministic = head(b['X'])
                pred = np.asarray(classes)[deterministic.argmax(-1).numpy()]
                head.train()
                p = torch.stack([head(b['X']).softmax(-1) for _ in range(30)])
                total = entropy(p.mean(0))
                within = entropy(p).mean(0)
            head.eval()
            for j in range(len(b['y'])):
                label = int(b['y'][j])
                rows.append(dict(split=split, index=int(b['index'][j]), label=label,
                                 unknown=label not in classes, prediction=int(pred[j]),
                                 predictive_entropy=float(total[j]), mean_pass_entropy=float(within[j]),
                                 eu=float(total[j]-within[j])))
    df = pd.DataFrame(rows)
    df.to_csv(HERE / 'scores.csv', index=False)
    metrics = {}
    for split in ('val', 'test'):
        sub = df[df.split == split]
        known = sub[~sub.unknown]
        metrics[split] = dict(known_rows=len(known), unknown_rows=int(sub.unknown.sum()),
                             known_accuracy=float((known.label == known.prediction).mean()),
                             known_mean_eu=float(known.eu.mean()),
                             unknown_mean_eu=float(sub[sub.unknown].eu.mean()),
                             eu_auroc=float(roc_auc_score(sub.unknown, sub.eu)),
                             entropy_auroc=float(roc_auc_score(sub.unknown, sub.predictive_entropy)))
    config = dict(dataset='CIFAR10 official test split, repartitioned by OSSIM', backbone='timm resnet18 pretrained default',
                  feature_cache=str(path.relative_to(ROOT)), classes=[int(c) for c in classes],
                  split_seed=0, training_seed=0, mc_seed=123, epochs=40, passes=30,
                  features=int(X.shape[1]), extraction_device=device, head_device='cpu',
                  torch_version=torch.__version__, metrics=metrics)
    (HERE / 'results.json').write_text(json.dumps(config, indent=2)+'\n')
    print(json.dumps(metrics, indent=2), flush=True)
    plot_results(df)


def plot_results(df):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.sans-serif': ['DejaVu Sans']})
    test = df[df.split == 'test']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, column, title in zip(axes[:2], ['eu', 'predictive_entropy'], ['Dropout disagreement (EU)', 'Predictive entropy']):
        bins = np.linspace(test[column].min(), test[column].max(), 31)
        for unknown, label in [(False, 'Known'), (True, 'Unknown')]:
            ax.hist(test.loc[test.unknown == unknown, column], bins=bins, density=True, alpha=.5, label=label)
        ax.set(xlabel='Score (nats)', ylabel='Density', title=title)
        ax.legend()
    for column, label in [('eu', 'EU'), ('predictive_entropy', 'Entropy')]:
        fpr, tpr, _ = roc_curve(test.unknown, test[column])
        auc = roc_auc_score(test.unknown, test[column])
        axes[2].plot(fpr, tpr, label=f'{label}: AUROC {auc:.3f}')
    axes[2].plot([0,1], [0,1], '--', color='gray')
    axes[2].set(xlabel='False positive rate (known)', ylabel='True positive rate (unknown)', title='Test ROC')
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(HERE / 'uncertainty_comparison.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
