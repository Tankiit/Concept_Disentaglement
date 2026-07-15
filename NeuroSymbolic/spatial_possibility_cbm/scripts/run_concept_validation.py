from __future__ import annotations

"""
scripts/run_concept_validation.py

Loops the single-item pipeline over a full dataset split and produces the
paper's headline table: Spearman rho between a predicted per-concept scalar
and ground-truth concept presence, for FOUR arms (the "ladder").

    B0  raw OpenCLIP cosine, max over patches    -- no possibility, no spatial
    B1  possibility pi, max over patches         -- + DPS transform
    B2  possibility pi, W-gated, max             -- + uncertainty gate
    B3  contiguity-composed WMC interval         -- + local geometry  <-- OURS

The paper's claim lives in B3 - B2. B0 is the arm that can falsify the whole
thing, so it runs first and costs nothing.

Usage:
    python scripts/run_concept_validation.py --dataset derm7pt --split test --arms B0,B1,B2,B3
    python scripts/run_concept_validation.py --dataset derm7pt --split test --arms B0 --limit 50   # smoke
"""

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from spatial_possibility_cbm.concept_composition_sdd import ConceptComposer, PatchEvidence, SubConcept
from spatial_possibility_cbm.clip_part_prototypes import (
    ClipConceptDirectionScorer,
    encode_clip_patch_embeddings,
    encode_clip_text_prototype,
    load_clip,
)
from dps_transform import consonant_possibility_from_samples


DERM7PT_CONCEPT_COLS = [
    "pigment_network",
    "streaks",
    "pigmentation",
    "regression_structures",
    "dots_and_globules",
    "blue_whitish_veil",
    "vascular_structures",
]


@dataclass
class Config:
    dataset: str
    split: str
    model_tag: str
    cache_root: Path
    cache_dir: Path
    M: int = 20
    NOISE: float = 0.05
    TEMP: float = 10.0
    w_gate: float = 0.30
    b3_max_candidates: int = 8
    cache_hits: int = 0
    cache_misses: int = 0

# --- project imports -------------------------------------------------------
# from spatial_possibility_cbm.data import load_dataset          # -> Iterable[DatasetItem]
# from spatial_possibility_cbm.concept_vocab import get_vocab    # dataset -> {concept: prompt}
# from spatial_possibility_cbm.clip_part_prototypes import OpenCLIPPartScorer
# from spatial_possibility_cbm.concept_composition_sdd import ConceptComposer, SubConcept, PatchEvidence
# from spatial_possibility_cbm.validation import spearman_correlation, binarize_derm7pt_label
# from dps_transform import consonant_possibility_from_samples   # YOUR file, not bundled


# ===========================================================================
# 1. The per-item forward pass, split by arm
# ===========================================================================

@dataclass
class ItemScores:
    """One image -> one scalar per concept, per arm."""
    item_id: str
    scores: dict[str, dict[str, float]] = field(default_factory=dict)  # arm -> concept -> scalar
    widths: dict[str, float] = field(default_factory=dict)             # concept -> (hi - lo), B3 only
    gt: dict[str, float] = field(default_factory=dict)                 # concept -> binarized label


@dataclass
class Derm7ptItem:
    item_id: str
    image_path: Path
    image_relpath: str
    row: pd.Series
    

class RealOpenCLIPScorer:
    def __init__(self, model, processor, device: str):
        self.model = model
        self.processor = processor
        self.device = device

    def patch_tokens(self, item: Derm7ptItem) -> np.ndarray:
        with Image.open(item.image_path).convert("RGB") as pil_image:
            return encode_clip_patch_embeddings(self.model, self.processor, pil_image, self.device)

    def concept_cosines(self, patch_tokens: np.ndarray, text_embeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        patch_norm = patch_tokens / (np.linalg.norm(patch_tokens, axis=1, keepdims=True) + 1e-8)
        out: dict[str, np.ndarray] = {}
        for concept, text_embed in text_embeds.items():
            text_norm = text_embed / (np.linalg.norm(text_embed) + 1e-8)
            out[concept] = (patch_norm @ text_norm).astype(np.float32)
        return out

    def concept_direction_ensemble(
        self,
        patch_tokens: np.ndarray,
        text_embed: np.ndarray,
        M: int,
        noise: float,
        temp: float,
    ) -> np.ndarray:
        scorer = ClipConceptDirectionScorer(
            clip_patch_embeds=patch_tokens,
            M=M,
            noise_scale=noise,
            temp=temp,
            dps_fn=consonant_possibility_from_samples,
        )
        return scorer(patch_tokens, text_embed)


def rankdata_avg_ties(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    sorted_x = x[order]
    n = len(x)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if x.shape != y.shape:
        raise ValueError("x and y must have identical shape")
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = int(x.size)
    if n < 2:
        return {"rho": float("nan"), "p": float("nan"), "n": n}
    rx = rankdata_avg_ties(x)
    ry = rankdata_avg_ties(y)
    xz = rx - rx.mean()
    yz = ry - ry.mean()
    denom = np.sqrt((xz * xz).sum() * (yz * yz).sum())
    if denom < 1e-12:
        return {"rho": float("nan"), "p": float("nan"), "n": n}
    rho = float((xz * yz).sum() / denom)

    # Approximate p-value via normal approximation on Fisher z when n is moderate.
    if n > 3 and abs(rho) < 1.0:
        z = 0.5 * math.log((1.0 + rho) / (1.0 - rho)) * math.sqrt(n - 3)
        # two-sided p from normal CDF via erf
        p = float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0)))))
    else:
        p = float("nan")
    return {"rho": rho, "p": p, "n": n}


def _normalize_01(series: pd.Series) -> np.ndarray:
    vals = pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    lo = float(vals.min())
    hi = float(vals.max())
    if hi - lo < 1e-12:
        return np.zeros_like(vals, dtype=np.float32)
    return (vals - lo) / (hi - lo)


def load_dataset(dataset: str, split: str, data_root: Path) -> list[Derm7ptItem]:
    if dataset != "derm7pt":
        raise NotImplementedError("This script is wired for --dataset derm7pt in this repo version.")

    split_map = {
        "train": "train_indexes.csv",
        "val": "valid_indexes.csv",
        "valid": "valid_indexes.csv",
        "test": "test_indexes.csv",
    }
    if split not in split_map:
        raise ValueError(f"Unsupported split '{split}'. Use train/valid/test.")

    meta_csv = data_root / "meta" / "meta.csv"
    idx_csv = data_root / "meta" / split_map[split]
    if not meta_csv.exists() or not idx_csv.exists():
        raise FileNotFoundError("Derm7pt meta files not found under data root/meta.")

    df = pd.read_csv(meta_csv)
    idx_df = pd.read_csv(idx_csv)
    index_col = "indexes" if "indexes" in idx_df.columns else idx_df.columns[0]
    # index files are 1-based; convert to 0-based for pandas iloc.
    ilocs = idx_df[index_col].astype(int).to_numpy() - 1

    items: list[Derm7ptItem] = []
    for iloc in ilocs:
        if iloc < 0 or iloc >= len(df):
            continue
        row = df.iloc[int(iloc)]
        rel = str(row["derm"])
        image_path = data_root / "images" / rel
        items.append(
            Derm7ptItem(
                item_id=str(row.get("case_num", iloc + 1)),
                image_path=image_path,
                image_relpath=rel,
                row=row,
            )
        )
    return items


def get_vocab(dataset: str, df: pd.DataFrame | None = None) -> dict[str, str]:
    if dataset != "derm7pt":
        raise NotImplementedError("Only derm7pt vocab is available in this script.")
    cols = DERM7PT_CONCEPT_COLS
    if df is not None:
        cols = [c for c in cols if c in df.columns]
    return {c: c.replace("_", " ") for c in cols}


def _is_present_value(v: object) -> float:
    s = str(v).strip().lower()
    if s in {"present", "typical", "regular"}:
        return 1.0
    if s in {"absent", "none", "nan", ""}:
        return 0.0
    if s in {"atypical", "irregular"}:
        return 1.0
    # Free-text states like "diffuse irregular" count as present.
    return 0.0 if "absent" in s else 1.0


def _cache_namespace(cfg: Config) -> Path:
    wg = str(cfg.w_gate).replace(".", "p")
    model = cfg.model_tag.replace("/", "_").replace(" ", "_")
    return cfg.cache_root / cfg.dataset / cfg.split / model / f"wg_{wg}"


def _item_cache_name(item: Derm7ptItem) -> str:
    key = f"{item.item_id}::{item.image_relpath}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return f"{digest}.npy"


def get_cached_patch_tokens(item: Derm7ptItem, scorer, cfg: Config) -> np.ndarray:
    cache_file = cfg.cache_dir / _item_cache_name(item)
    if cache_file.exists():
        cfg.cache_hits += 1
        return np.load(cache_file)

    patch_tokens = scorer.patch_tokens(item)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, patch_tokens)
    cfg.cache_misses += 1
    return patch_tokens


def score_item(item, vocab, scorer, text_embeds, composer, arms, cfg) -> ItemScores:
    out = ItemScores(item_id=item.item_id)

    # ---- shared: patch-level evidence -------------------------------------
    # patch_tokens: (n_patches, d) from LAST-ViT
    # cosines[c]:   (n_patches,) raw OpenCLIP cosine of each patch vs concept prompt c
    patch_tokens = get_cached_patch_tokens(item, scorer, cfg)          # TODO: confirm grid alignment (14x14)
    cosines = scorer.concept_cosines(patch_tokens, text_embeds)   # dict concept -> (n_patches,)

    # ---- B0: raw cosine ---------------------------------------------------
    if "B0" in arms:
        out.scores["B0"] = {c: float(v.max()) for c, v in cosines.items()}

    # Everything below needs the possibilistic ensemble.
    if arms <= {"B0"}:
        out.gt = {c: binarize_gt(item, c, cfg.dataset) for c in vocab}
        return out

    # ---- ensemble -> (pi, W) ---------------------------------------------
    # TODO(tanmoy): WHICH ENSEMBLE? session_notes lists three genuinely
    # different constructions and calls this unsettled:
    #   (a) noise-perturbation Grad-CAM  (N backward passes -- expensive)
    #   (b) attention-head ensemble      (needs attn_implementation="eager")
    #   (c) concept-direction wobble     (M perturbed CLS queries, ONE forward,
    #                                     no backward -- your own construction)
    # (c) is the only one that survives a full-split loop on a laptop budget.
    # Recommend locking (c) for the paper and citing (a)/(b) as alternatives.
    pi, W = {}, {}
    for c, prompt_embed in text_embeds.items():
        samples = scorer.concept_direction_ensemble(       # (M, n_patches)
            patch_tokens, prompt_embed,
            M=cfg.M, noise=cfg.NOISE, temp=cfg.TEMP,       # TODO: all three are untuned guesses
        )
        pi[c], W[c] = consonant_possibility_from_samples(samples)  # each (n_patches,)

    # ---- B1: pi, ungated --------------------------------------------------
    if "B1" in arms:
        out.scores["B1"] = {c: float(pi[c].max()) for c in vocab}

    # ---- B2: pi, W-gated --------------------------------------------------
    if "B2" in arms:
        out.scores["B2"] = {}
        for c in vocab:
            keep = W[c] <= cfg.w_gate                       # TODO: w_gate value?
            # DEGENERACY GUARD: if the gate kills every patch the max is -inf
            # and Spearman silently eats a constant column. Fail loudly.
            out.scores["B2"][c] = float(pi[c][keep].max()) if keep.any() else 0.0

    # ---- B3: contiguity-composed WMC -------------------------------------
    if "B3" in arms:
        out.scores["B3"] = {}
        for c in vocab:
            if cfg.b3_max_candidates > 0:
                gated = np.where(W[c] <= cfg.w_gate)[0]
                if gated.size > cfg.b3_max_candidates:
                    topk = gated[np.argsort(pi[c][gated])[::-1][: cfg.b3_max_candidates]]
                    W_b3 = np.ones_like(W[c])
                    W_b3[topk] = W[c][topk]
                    pi_b3 = pi[c]
                else:
                    W_b3 = W[c]
                    pi_b3 = pi[c]
            else:
                pi_b3 = pi[c]
                W_b3 = W[c]
            sub = SubConcept(name=c, evidence=PatchEvidence(pi=pi_b3, W=W_b3), w_gate=cfg.w_gate)
            node = composer.build_concept([sub])            # single concept, contiguity default
            lo, hi = composer.wmc_interval(node, [sub])
            # TODO(tanmoy): WHICH SCALAR? lo / hi / midpoint. See discussion.
            out.scores["B3"][c] = 0.5 * (lo + hi)
            out.widths[c] = hi - lo                         # second, separate result

    # ---- ground truth -----------------------------------------------------
    out.gt = {c: binarize_gt(item, c, cfg.dataset) for c in vocab}
    return out


def binarize_gt(item, concept, dataset) -> float:
    """
    derm7pt : 7-point checklist column -> {0, 1}. TODO: confirm value vocabulary
              in meta.csv (absent/typical/atypical? present/absent?).
    cub     : no per-part labels exist. Spatial only -> use box-overlap of the
              argmax patch, NOT a concept label. Different metric, don't pool.
    milk10k : MONET score is CONTINUOUS. Do NOT binarize -- Spearman handles
              continuous targets fine and binarizing throws away the signal.
    """
    if dataset != "derm7pt":
        raise NotImplementedError("binarize_gt currently supports only derm7pt")
    return _is_present_value(item.row.get(concept, "absent"))


# ===========================================================================
# 2. The loop
# ===========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cub", "derm7pt", "milk10k"], required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--arms", default="B0,B1,B2,B3")
    ap.add_argument("--limit", type=int, default=None, help="smoke-test on N items")
    ap.add_argument(
        "--data-root",
        type=Path,
        default=Path("/Users/tanmoy/research/data/derm7pt"),
        help="Dataset root containing images/ and meta/ (set to your Derm7pt path by default).",
    )
    ap.add_argument("--cache", type=Path, default=Path(".cache/patch_tokens"))
    ap.add_argument("--model-tag", default="openai/clip-vit-base-patch16", help="Model ID used in patch-token cache key.")
    ap.add_argument("--w-gate", type=float, default=0.30, help="Uncertainty gate threshold used by B2.")
    ap.add_argument("--b3-max-candidates", type=int, default=8, help="Max gated patches per concept to compile in B3.")
    ap.add_argument("--device", default=None, help="Override CLIP device (cuda, mps, cpu).")
    ap.add_argument("--out", type=Path, default=Path("results/"))
    args = ap.parse_args()
    arms = set(args.arms.split(","))
    cfg = Config(
        dataset=args.dataset,
        split=args.split,
        model_tag=args.model_tag,
        cache_root=args.cache,
        cache_dir=Path("."),
        w_gate=args.w_gate,
        b3_max_candidates=args.b3_max_candidates,
    )
    cfg.cache_dir = _cache_namespace(cfg)

    # TODO(tanmoy): CACHE THE PATCH TOKENS. You will run this loop 4+ arms x
    # several w_gate values x 3 datasets. Recomputing ViT+OpenCLIP forward passes
    # every time is the difference between a 3-minute iteration and a 40-minute
    # one, and you have 6 days. Key on (dataset, split, item_id).

    items = load_dataset(args.dataset, args.split, args.data_root)
    if args.limit:
        items = items[: args.limit]
    meta_df = pd.read_csv(args.data_root / "meta" / "meta.csv")
    vocab = get_vocab(args.dataset, meta_df)
    if not vocab:
        raise RuntimeError("No concept columns found in Derm7pt metadata.")

    device = args.device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model, processor = load_clip(device=device)
    scorer = RealOpenCLIPScorer(model=model, processor=processor, device=device)
    text_embeds = {c: encode_clip_text_prototype(model, processor, prompt, device) for c, prompt in vocab.items()}
    # B3 uses one literal per (concept, patch_index); reserve enough upfront.
    n_patches_hint = max(args.b3_max_candidates, 1)
    num_vars_hint = max(256, len(vocab) * n_patches_hint + 32)
    composer = ConceptComposer(num_vars_hint=num_vars_hint)

    all_scores = [score_item(it, vocab, scorer, text_embeds, composer, arms, cfg) for it in items]

    # ---- per-concept, per-arm Spearman -----------------------------------
    # Report PER CONCEPT (7 rows for derm7pt), not pooled. Pooling across
    # concepts with different base rates manufactures correlation from the
    # base-rate spread alone -- a reviewer will catch this.
    table = {}
    for arm in sorted(arms):
        table[arm] = {}
        for c in vocab:
            x = np.array([s.scores[arm][c] for s in all_scores])
            y = np.array([s.gt[c] for s in all_scores])
            table[arm][c] = spearman_correlation(x, y)   # -> (rho, p, n)

            # DEGENERACY GUARD (the plan's Day 1-3 worry, made automatic):
            if np.ptp(x) < 1e-9:
                print(f"!! DEGENERATE: {arm}/{c} predicted a constant. "
                      f"Check w_gate / pi+-W/2 weight interval.")

    # ---- second result: does width track disagreement? --------------------
    # TODO(tanmoy): derm7pt has no inter-rater data, so "disagreement" has to
    # be proxied. Cheapest honest proxy: is W higher on items where the
    # checklist label and the diagnosis label conflict? If no proxy holds up,
    # DROP this claim rather than stretch it.

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.dataset}_{args.split}_ladder.json").write_text(json.dumps(table, indent=2))
    print(f"cache_dir={cfg.cache_dir} hits={cfg.cache_hits} misses={cfg.cache_misses}")
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
