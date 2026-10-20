"""Call the original loader module on the existing real-image feature cache."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from oss_from_cache import group_loaders

if __name__ == '__main__':
    loaders = group_loaders(path=ROOT / 'cache/features/timm_resnet18_c10test.npz',
                            batch_size=256, shuffle=False, seed=0)
    for split, groups in loaders.items():
        for group, loader in groups.items():
            print(f'\n{split}/{group}: {len(loader.dataset)} rows, {len(loader)} batches')
            if not len(loader.dataset):
                continue
            batch = next(iter(loader))
            print('Batch shapes:', {key: tuple(value.shape) for key, value in batch.items()})
            print('First five original row IDs:', batch['index'][:5].tolist())
            print('First five labels:', batch['y'][:5].tolist())
            print('Group codes:', batch['group'].unique().tolist())
