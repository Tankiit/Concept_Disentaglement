"""Expose the quantities computed by example.eu_bald on a synthetic dataset."""
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from oss_from_cache import make_oss, make_loader
from example import Head, train, eu_bald


def entropy(p):
    return -(p * p.clamp_min(1e-12).log()).sum(dim=-1)


def components(p):
    total = entropy(p.mean(dim=0))
    within = entropy(p).mean(dim=0)
    return total, within, total - within


def sample_predictions(head, X, T=30):
    previous_mode = head.training
    try:
        head.train()  # For this head, activates dropout. No optimizer update occurs.
        with torch.no_grad():
            return torch.stack([head(X).softmax(dim=-1) for _ in range(T)])
    finally:
        head.train(previous_mode)


def main():
    torch.set_num_threads(1)
    print('Controlled two-class probability examples (natural logs / nats):')
    for name, probs in [('confident agreement', [[.99, .01], [.99, .01]]),
                        ('uncertain agreement', [[.5, .5], [.5, .5]]),
                        ('confident disagreement', [[.99, .01], [.01, .99]])]:
        total, within, eu = components(torch.tensor(probs)[:, None, :])
        print(f'{name}: H(mean)={total.item():.4f}, mean H={within.item():.4f}, EU={eu.item():.4f}')

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(10), 100)
    X = (3 * np.eye(10)[y] + rng.normal(0, .1, size=(len(y), 10))).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'synthetic.npz'
        np.savez(path, X=X, y=y)
        world = make_oss(path, seed=0)
    classes = sorted(world['kkc'])
    lut = torch.full((10,), -1, dtype=torch.long)
    for i, c in enumerate(classes):
        lut[c] = i
    idx = world['idx']['train']
    idx = idx[np.isin(y[idx], classes)]
    loader = make_loader(X, y, indices=idx, batch_size=32, shuffle=True, seed=0)
    head = Head(d=10, hidden=32, k=len(classes))
    train(head, loader, lut, epochs=20, seed=0)
    head.eval()

    idx = world['idx']['test']
    features = torch.from_numpy(X[idx])
    known = np.isin(y[idx], classes)
    weights_before = {k: v.clone() for k, v in head.state_dict().items()}
    torch.manual_seed(123)
    p = sample_predictions(head, features)
    total, within, eu = components(p)
    print('\nPrediction tensor shape (passes, rows, classes):', tuple(p.shape))
    print('Probability columns correspond to original classes:', classes)
    for group, mask in [('known', known), ('unknown', ~known)]:
        j = int(np.flatnonzero(mask)[0])
        print(f'\n{group} example: original row={idx[j]}, original class={y[idx[j]]}')
        print('First five predictions:', np.round(p[:5, j].numpy(), 4).tolist())
        print('Mean prediction:', np.round(p[:, j].mean(0).numpy(), 4).tolist())
        print(f'H(mean)={total[j]:.4f}, mean H={within[j]:.4f}, EU={eu[j]:.4f}')
        print(f'Group mean EU ({mask.sum()} rows): {eu[mask].mean():.4f}')
    print('\nUnknown-positive test AUROC:', round(roc_auc_score(~known, eu.numpy()), 4))

    # Same RNG seed and batch reproduce the original function's dropout masks.
    torch.manual_seed(123)
    original = eu_bald(head, features)
    print('Matches original eu_bald:', torch.allclose(eu, original, atol=1e-7))
    print('Original eu_bald leaves training mode enabled:', head.training)
    head.eval()
    print('Weights unchanged during scoring:',
          all(torch.equal(weights_before[k], v) for k, v in head.state_dict().items()))
    with torch.no_grad():
        deterministic = torch.stack([head(features).softmax(-1) for _ in range(2)])
    print('Repeated predictions identical with dropout disabled:',
          torch.equal(deterministic[0], deterministic[1]))
    print('Synthetic demonstration only; unknown classes need not have higher EU.')


if __name__ == '__main__':
    main()
