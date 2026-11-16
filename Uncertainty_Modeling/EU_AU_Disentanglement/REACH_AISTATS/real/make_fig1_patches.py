"""Pick one CAMELYON17 tumor patch per known hospital (0 and 3) for the medical Fig 1.

Mirror branch of the collaborator's make_fig1_camelyon.py: the official WILDS
v1.0 folder layout is not on disk, so records come from the pinned parquet
mirror (wltjr1007/Camelyon17-WILDS @ d784d534, sha-pinned by
data/wilds/camelyon17-mirror/verified_manifest.json), as in the pilots.

Shard choice: the mirror's test split is entirely center 2 (the candidate
hospital), so hospital-0/3 tumor rows are read from ONE train shard —
train-00000-of-00014 — which contains tumor rows of both known hospitals
(2,152 center-0 and 19,451 center-3 tumor rows). Sampling is seeded; every
candidate's row coordinates are recorded.

    python -m real.make_fig1_patches            # write the contact sheets
    python -m real.make_fig1_patches --pick 0=3 3=5
                                                # copy chosen patches to paper/figures/

Writes workbook/fig1_patches_h03/: candidates_h{0,3}.png/.csv (8 seeded
candidates per hospital), cand_h{c}_{i}.png, candidates_meta.json
(provenance), and on --pick fig1_patch_h{0,3}.png into paper/figures/ plus a
fig1_patches_h03.json provenance record beside them.
"""
import argparse
import json
from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path.home() / '.cache/huggingface/hub/datasets--wltjr1007--Camelyon17-WILDS' \
       / 'snapshots/d784d5344ba6c967f83f9f3d9b2f1e2a4d6eb78f/data'
SHARD = ROOT / 'train-00000-of-00014.parquet'
SHARD_SHA256 = '9e6f5425a371de015b9bd1bfcd6901554a8b4b4ab7df268f768e2629a55f2829'  # verified_manifest.json

TUMOR = 1            # a tumor patch from each hospital, so the two tiles differ only by source
N_CANDIDATES = 8     # contact sheet; the clearest one is picked by eye
SEED = 0
BASE = Path(__file__).resolve().parents[1]
OUT = BASE / 'workbook' / 'fig1_patches_h03'
FIG = BASE / 'paper' / 'figures'


def load_candidates():
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(SHARD)
    meta = pf.read(columns=[c for c in pf.schema_arrow.names if c != 'image']).to_pandas()
    meta['image_row'] = range(len(meta))
    picks = {}
    for center in (0, 3):
        rows = meta[(meta.center == center) & (meta.label == TUMOR)].sample(
            N_CANDIDATES, random_state=SEED).reset_index(drop=True)
        image_col = pf.read(columns=['image']).column('image')
        tiles = [Image.open(BytesIO(image_col[r.image_row].as_py()['bytes'])).convert('RGB')
                 for _, r in rows.iterrows()]
        sheet = Image.new('RGB', (96 * len(tiles), 96), 'white')
        for k, t in enumerate(tiles):
            t.save(OUT / f'cand_h{center}_{k}.png')
            sheet.paste(t, (96 * k, 0))
        sheet.save(OUT / f'candidates_h{center}.png')
        rows.drop(columns=['image_row']).to_csv(OUT / f'candidates_h{center}.csv', index=False)
        print(f'hospital {center}: {OUT}/candidates_h{center}.png / .csv')
        picks[center] = rows
    (OUT / 'candidates_meta.json').write_text(json.dumps(dict(
        source_repo='wltjr1007/Camelyon17-WILDS', revision=SHARD.parents[1].name,
        shard=SHARD.name, shard_sha256=SHARD_SHA256, tumor=TUMOR, seed=SEED,
        n_candidates=N_CANDIDATES,
        rows={str(c): picks[c].to_dict('records') for c in (0, 3)}), indent=2))


def pick(selection):
    meta = json.loads((OUT / 'candidates_meta.json').read_text())
    prov = {'description': 'CAMELYON17 tumor patches per known hospital for the medical Fig 1',
            'source_repo': meta['source_repo'], 'revision': meta['revision'],
            'shard': meta['shard'], 'shard_sha256': meta['shard_sha256'],
            'tumor': meta['tumor'], 'seed': meta['seed'],
            'chosen': {}}
    for center, idx in selection.items():
        src = OUT / f'cand_h{center}_{idx}.png'
        dst = FIG / f'fig1_patch_h{center}.png'
        dst.write_bytes(src.read_bytes())
        row = meta['rows'][str(center)][idx]
        import hashlib
        prov['chosen'][f'hospital_{center}'] = dict(
            candidate_index=idx, png=str(dst.relative_to(BASE)),
            png_sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),
            patient=int(row['patient']), node=int(row['node']),
            x_coord=int(row['x_coord']), y_coord=int(row['y_coord']),
            slide=int(row['slide']), image_id=int(row['image_id']))
        print(f'hospital {center}: cand {idx} -> {dst}')
    (FIG / 'fig1_patches_h03.json').write_text(json.dumps(prov, indent=2))
    print(FIG / 'fig1_patches_h03.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pick', nargs='*', default=[], metavar='CENTER=IDX',
                    help='e.g. --pick 0=3 3=5 ; copies the chosen candidate patches')
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if not (OUT / 'candidates_meta.json').exists():
        load_candidates()
    if args.pick:
        pick({int(s.split('=')[0]): int(s.split('=')[1]) for s in args.pick})
    elif (OUT / 'candidates_meta.json').exists():
        print('candidates exist; rerun with --pick 0=IDX 3=IDX to choose')
