"""Inspect OSSIM splits using synthetic features; no downloads required."""
from pathlib import Path
import sys
import tempfile

import numpy as np

# Resolve the original project modules independently of the shell's directory.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from oss_from_cache import make_oss, make_loader


def main():
    # Ten classes with 100 rows each; features are arbitrary, not image embeddings.
    X = np.random.default_rng(0).normal(size=(1000, 4)).astype(np.float32)
    y = np.repeat(np.arange(10), 100)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'synthetic.npz'
        np.savez(path, X=X, y=y)
        world = make_oss(path, seed=0)

    print('\nRows per split and group:')
    for split, idx in world['idx'].items():
        counts = {g: int(np.isin(y[idx], list(world[g])).sum())
                  for g in ('kkc', 'kuc', 'uuc')}
        print(f'{split}: {counts}; total={len(idx)}')
    val_unknown = set(world['oss'].classes['val']['uuc'])
    test_unknown = set(world['oss'].classes['test']['uuc'])
    print('Validation unknown classes:', sorted(val_unknown))
    print('Test unknown classes:', sorted(test_unknown))
    print('Unknown class overlap:', sorted(val_unknown & test_unknown))

    loader = make_loader(X, y, indices=world['idx']['train'], batch_size=8)
    batch = next(iter(loader))
    print('\nOne training batch:')
    for key, value in batch.items():
        print(f'{key}: shape={tuple(value.shape)}, dtype={value.dtype}')
    print('Original row IDs:', batch['index'].tolist())
    print('Original labels:', batch['y'].tolist())
    print('Features match original rows:',
          np.array_equal(batch['X'].numpy(), X[batch['index'].numpy()]))


if __name__ == '__main__':
    main()
