"""Download the pinned public mirror, retaining every shard (no first-shard sampling)."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import subprocess
ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'data/wilds/camelyon17-mirror'
REVISION = 'd784d5344ba6c967f83f9f3d9b2f1e2a4d6eb78f'
URL = f'https://huggingface.co/datasets/wltjr1007/Camelyon17-WILDS/resolve/{REVISION}/data/'

def fetch(name):
    target = DEST / name
    if target.exists():
        with target.open('rb') as f:
            f.seek(-4, 2)
            if f.read() == b'PAR1':
                return name + ' cached'
    partial = target.with_suffix('.partial')
    subprocess.run(['curl', '-fLsS', '--retry', '3', '--max-time', '1800',
                    '-o', str(partial), URL + name], check=True)
    partial.replace(target)
    return name + ' downloaded'

if __name__ == '__main__':
    DEST.mkdir(parents=True, exist_ok=True)
    files = [f'{split}-{i:05d}-of-{n:05d}.parquet'
             for split, n in [('train', 14), ('validation', 3), ('test', 4)] for i in range(n)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(fetch, files):
            print(result, flush=True)
