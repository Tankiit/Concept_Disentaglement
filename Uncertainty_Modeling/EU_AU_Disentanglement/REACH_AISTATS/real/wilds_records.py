"""Auditable local adapters. No image, class, or domain is synthesized.

Accept official WILDS folders or the documented parquet mirrors. Split by
patient (Camelyon) or location (FMoW), never by individual image.
"""
from pathlib import Path
import hashlib
import io
import json
import numpy as np
import pandas as pd
from PIL import Image

MIRRORS = {
    'camelyon17': ('wltjr1007/Camelyon17-WILDS', 'd784d5344ba6c967f83f9f3d9b2f1e2a4d6eb78f'),
    'fmow': ('danielz01/fMoW', '8a27a7988a20341352b85bf0e65e88a6c1337f1c'),
}
SOURCES = {'camelyon17': ['0', '3', '2'], 'fmow': ['Americas', 'Europe', 'Asia']}


def read_records(dataset, root):
    root = Path(root).expanduser().resolve()
    shards = sorted(root.rglob('*.parquet'))
    if shards:
        import pyarrow.parquet as pq
        frames = []
        # Fail on incomplete splits, rather than silently sampling early shards.
        import re
        for f in shards:
            match = re.search(r'(.+)-(\d{5})-of-(\d{5})', f.name)
            if match and len([p for p in shards if p.name.startswith(match[1] + '-')]) != int(match[3]):
                raise ValueError(f'Incomplete parquet split: {match[1]} needs {match[3]} shards')
            cols = [c for c in pq.ParquetFile(f).schema_arrow.names if c != 'image']
            frame = pq.read_table(f, columns=cols).to_pandas()
            frame['image_file'] = str(f)
            frame['image_row'] = np.arange(len(frame))
            frame['official_split'] = f.name.split('-')[0]
            frame['record_id'] = [f'{f.name}:{i}' for i in range(len(frame))]
            frames.append(frame)
        raw = pd.concat(frames, ignore_index=True)
        if dataset == 'camelyon17':
            raw['source_name'] = raw.center.astype(str)
            raw['group'] = raw.patient.astype(str)
            raw['target'] = raw.label.astype(int)
        else:
            if 'domain_labels_readable' not in raw or not {'lat', 'lon'} <= set(raw):
                raise ValueError('FMoW mirror must preserve readable regions and location metadata')
            raw['source_name'] = raw.domain_labels_readable.map(lambda x: x['region'])
            raw['target'] = raw.label.astype(int)
            raw['group'] = location_groups(raw)
        raw['storage'] = 'parquet'
    elif dataset == 'camelyon17' and (root / 'metadata.csv').exists():
        raw = pd.read_csv(root / 'metadata.csv', index_col=0, dtype={'patient': str})
        raw['record_id'] = raw.index.astype(str)
        raw['image_file'] = [str(root / f'patches/patient_{p}_node_{n}/patch_patient_{p}_node_{n}_x_{x}_y_{y}.png')
                             for p,n,x,y in raw[['patient','node','x_coord','y_coord']].itertuples(index=False, name=None)]
        raw['source_name'] = raw.center.astype(str)
        raw['group'] = raw.patient.astype(str)
        raw['target'] = raw.tumor.astype(int)
        raw['official_split'] = raw.split.astype(str)
        raw['storage'] = 'image'
    elif dataset == 'fmow' and (root / 'rgb_metadata.csv').exists():
        raw = pd.read_csv(root / 'rgb_metadata.csv')
        raw['record_id'] = raw.index.astype(str)
        raw['image_file'] = [str(root / 'images' / f'rgb_img_{i}.png') for i in raw.index]
        countries = pd.read_csv(root / 'country_code_mapping.csv')
        mapping = dict(zip(countries['alpha-3'], countries['region']))
        raw['source_name'] = raw.country_code.map(mapping).fillna('Other')
        categories = sorted(raw.loc[raw.split != 'seq', 'category'].unique())
        raw['target'] = raw.category.map(dict(zip(categories, range(len(categories)))))
        raw['group'] = location_groups(raw)
        raw['official_split'] = raw.split.astype(str)
        raw = raw[raw.split != 'seq'].copy()
        raw['storage'] = 'image'
    else:
        raise FileNotFoundError(f'{dataset}: no supported local data at {root}')
    raw['source'] = raw.source_name.map(dict(zip(SOURCES[dataset], range(3))))
    raw = raw[raw.source.notna()].copy()
    raw['source'] = raw.source.astype(int)
    raw['target'] = raw.target.astype(int)
    if dataset == 'fmow':
        raw['year'] = pd.to_datetime(raw.timestamp, format='ISO8601', utc=True).dt.year
        # Preserve later years for a separate, declared temporal stress test.
        raw['temporal_ood'] = raw.year >= 2016
    else:
        raw['temporal_ood'] = False
    if raw.record_id.duplicated().any():
        raise ValueError('Record IDs must be unique')
    if set(raw.source) != {0,1,2}:
        raise ValueError(f'Missing declared sources: expected {SOURCES[dataset]}')
    return raw.reset_index(drop=True)


def location_groups(frame):
    if not {'lat','lon'} <= set(frame) or frame[['lat','lon']].isna().any().any():
        raise ValueError('FMoW requires finite lat/lon to prevent cross-location leakage')
    # Conservative roughly 1-km grid; nearby repeated observations remain grouped.
    return frame.lat.round(2).astype(str) + ':' + frame.lon.round(2).astype(str)


def group_partition(frame, seed=20261030):
    """Group identities alone select roles. No labels choose splits.

    Per-source sorted hash permutation allocates 60/20/20% of groups. Every
    source needs >=5 groups. Repeated locations across sources are forbidden.
    """
    f = frame.copy()
    if f.groupby('group').source.nunique().max() != 1:
        raise ValueError('A grouping unit spans source definitions')
    assignments = {}
    for source, part in f.groupby('source'):
        groups = sorted(part.group.unique(), key=lambda g: hashlib.sha256(f'{seed}:{g}'.encode()).hexdigest())
        n = len(groups)
        if n < 5:
            raise ValueError(f'Source {source} has only {n} independent groups; need >=5')
        nr = max(1, n // 5)
        for i,g in enumerate(groups):
            assignments[g] = 'reference' if i < nr else 'evaluation' if i < 2*nr else 'fit'
    f['role'] = f.group.map(assignments)
    assert f.groupby('group').role.nunique().max() == 1
    return f


def image_sample(frame, per_role_source=2000, seed=20261030):
    """Uniform image subsamples within each role/source; no label balancing."""
    parts = []
    for _, part in frame.groupby(['role','source'], sort=True):
        parts.append(part.sample(n=min(len(part), per_role_source), random_state=seed))
    return pd.concat(parts).sort_values('record_id').reset_index(drop=True)


def load_images(frame):
    """Yield (row position, PIL image), reading each selected parquet shard once."""
    import pyarrow.parquet as pq
    for path, part in frame.groupby('image_file', sort=True):
        if part.storage.iloc[0] == 'parquet':
            table = pq.read_table(path, columns=['image'])
            for pos, row in part.iterrows():
                value = table['image'][int(row.image_row)].as_py()
                if not value.get('bytes'):
                    raise ValueError('Mirror image bytes are missing; external paths are not trusted')
                yield pos, Image.open(io.BytesIO(value['bytes'])).convert('RGB')
        else:
            for pos, _ in part.iterrows():
                yield pos, Image.open(path).convert('RGB')


def save_data_audit(frame, root, output, dataset):
    output = Path(output)
    frame.to_csv(output / 'record_index.csv.gz', index=False)
    frame.groupby(['source_name','role']).agg(records=('record_id','size'),groups=('group','nunique')).to_csv(output/'split_counts.csv')
    info = dict(dataset=dataset, data_root=str(Path(root).resolve()), source_names=SOURCES[dataset],
                split='custom group-disjoint acquisition split, NOT the official WILDS benchmark split',
                mirror_if_used=MIRRORS[dataset], records=len(frame), groups=frame.group.nunique())
    (output/'data_audit.json').write_text(json.dumps(info, indent=2))
