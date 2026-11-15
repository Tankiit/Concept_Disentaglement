"""Execute the original pool bookkeeping on real CIFAR labels and row-ID markers.

X contains image row IDs, NOT embeddings. This tests assignments and alignment;
it performs no feature extraction, model fitting, or REACH experiment.
"""
from pathlib import Path
import sys
import json
import hashlib
from unittest.mock import patch
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from data.pools import load_pools, check_disjoint

SOURCES = {
 'train': ('8428b53a88a11ac374111006708df51469e315a22ac6d66470afd9c78d2ae883', 50000),
 'test': ('841389e6f2d64f28bf17310e430aebac20ec3ba611a3c5e231dc93c645ce84de', 10000),
}


def main():
    arrays = {}
    provenance = {}
    for split, (expected, count) in SOURCES.items():
        path = ROOT / 'cache/cifar10' / f'{split}.parquet'
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f'{split}: checksum mismatch')
        y = pq.read_table(path, columns=['label']).column('label').to_numpy().astype(np.int16)
        assert len(y) == count
        arrays[f'cifar10_{split}'] = {'X': np.arange(count, dtype=np.int64)[:, None], 'y': y}
        provenance[split] = dict(path=str(path.relative_to(ROOT)), sha256=digest,
            rows=count, source_url=f'https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/{split}-00000-of-00001.parquet')

    summaries, assignments = [], []
    # Substitute data access only: run the original splitting/checking functions.
    with patch('data.pools.load', side_effect=lambda name, root: arrays[name]):
        for config, m in [('reference_200_per_class', 200), ('full_reference', None)]:
            pools = load_pools(seed=0, m_per_class=m)
            check_disjoint(pools)
            print(f'\n{config}:')
            for name, pool in pools.items():
                idx = pool['index']
                assert np.array_equal(pool['X'][:, 0], idx)
                assert np.array_equal(pool['y'], arrays[pool['pool']]['y'][idx])
                counts = np.bincount(pool['y'], minlength=10)
                summaries.append(dict(config=config, pool=name, source=pool['pool'], rows=len(idx),
                                      **{f'class_{c}': int(n) for c,n in enumerate(counts)}))
                assignments.extend(dict(config=config, pool=name, source=pool['pool'], index=int(i), label=int(y))
                                   for i,y in zip(idx,pool['y']))
                print(f'{name:10s} {len(idx):6d} {pool["pool"]}: {counts.tolist()}')
            assert len(set(pools['log']['index']) & set(pools['eval']['index'])) == 0
            assert len(set(pools['dev']['index']) & set(pools['reference']['index'])) == 0
            assert np.array_equal(np.r_[pools['log']['index'], pools['eval']['index']], np.arange(10000))
            if m is None:
                assert len(set(pools['dev']['index']) | set(pools['reference']['index'])) == 50000
    pd.DataFrame(summaries).to_csv(HERE/'pool_summary.csv', index=False)
    pd.DataFrame(assignments).to_csv(HERE/'pool_assignments.csv', index=False)
    provenance['scope'] = 'Actual CIFAR labels; X replaced by row-ID markers. Original pool bookkeeping only.'
    provenance['pool_code_sha256'] = hashlib.sha256((ROOT/'data/pools.py').read_bytes()).hexdigest()
    (HERE/'pool_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    inventory=[]
    for path in sorted((ROOT/'cache/features').glob('*.npz')):
        with np.load(path,allow_pickle=False) as f:
            inventory.append(dict(file=path.name, arrays={k:dict(shape=list(f[k].shape),dtype=str(f[k].dtype)) for k in f.files}))
    (HERE/'cache_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    print('\nVerified disjointness, row/label alignment, and expected pool totals.')
    print('No CLIP feature values were loaded, and no notebook cells were executed.')


if __name__ == '__main__':
    main()
