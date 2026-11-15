"""Trace label remapping and head training on synthetic, separable features."""
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from oss_from_cache import make_oss, make_loader
from example import Head, train


def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)  # Seed BEFORE creating the head to reproduce its weights.
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(10), 100)
    # Each class has a distinct one-hot centre; this is deliberately easy toy data.
    X = (3 * np.eye(10)[y] + rng.normal(0, .1, size=(len(y), 10))).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'synthetic.npz'
        np.savez(path, X=X, y=y)
        world = make_oss(path, seed=0)

    classes = sorted(world['kkc'])
    lut = torch.full((10,), -1, dtype=torch.long)
    for output_index, original_label in enumerate(classes):
        lut[original_label] = output_index
    print('\nOriginal label -> classifier output:', {int(c): int(lut[c]) for c in classes})
    print('All labels through lookup:', lut.tolist())

    idx = world['idx']['train']
    idx = idx[np.isin(y[idx], classes)]
    loader = make_loader(X, y, indices=idx, batch_size=32, shuffle=True, seed=0)
    batch = next(iter(loader))
    print('Example original labels:', batch['y'][:12].tolist())
    print('Example remapped labels:', lut[batch['y'][:12]].tolist())

    # Explicit input dimension: example.py's default Head assumes 512 features.
    head = Head(d=X.shape[1], hidden=32, k=len(classes))
    print('Logit shape:', tuple(head(batch['X']).shape))
    losses = train(head, loader, lut, epochs=20, seed=0)
    print('Training loss, epoch 1:', round(losses[0], 4))
    print('Training loss, epoch 20:', round(losses[-1], 4))

    val_idx = world['idx']['val']
    val_idx = val_idx[np.isin(y[val_idx], classes)]
    head.eval()
    with torch.no_grad():
        logits = head(torch.from_numpy(X[val_idx]))
        outputs = logits.argmax(dim=1)
        original_predictions = torch.tensor(classes)[outputs]
        accuracy = (original_predictions == torch.from_numpy(y[val_idx])).float().mean()
    print('Known-class validation rows:', len(val_idx))
    print('Known-class validation accuracy:', round(float(accuracy), 4))
    print('Toy result only; this does not measure image classification or uncertainty quality.')


if __name__ == '__main__':
    main()
