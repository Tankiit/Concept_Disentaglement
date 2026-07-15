"""
concept_composition_sdd.py
---------------------------
Compose hierarchical concepts (e.g. bird = head AND wing AND tail) over
LAST-ViT patch tokens using pySDD, with the possibilistic quantities from
`dps_transform.py` (consonant_possibility_from_samples -> pi, W) feeding the
SDD as WEIGHTS rather than plain probabilities.

SKELETON. Everything marked TODO is a real open decision, not a detail —
see "Open design questions" at the bottom before running this on a real
image.

Core idea
---------
1. For a given image, run one of the possibilistic-CAM ensembles from
   `attention_possibilistic_gradcam*.py` / `concept_direction_possibilistic_cam.py`
   to get, per patch i: pi_i (possibility the patch supports the concept)
   and W_i (specificity width / how "unspecific"-uncertain that possibility is).

2. For each sub-concept c (head / wing / tail / ...), gate which patches are
   even allowed to become SDD literals using W_i (tight, low-uncertainty
   patches only), then build a Boolean formula over the surviving patch
   literals for c (currently: OR — "at least one gated patch fires").

3. AND the sub-concept formulas into the top-level concept ("bird").

4. Compile with pySDD and run WEIGHTED MODEL COUNTING TWICE, once with a
   pessimistic (lower) literal-weight assignment and once with an optimistic
   (upper) one, both derived from (pi, W). The result is a WMC INTERVAL
   [lo, hi] for the composed concept — i.e. a credal-set-style plausibility
   bound on "this image instantiates the composed concept", computed
   symbolically instead of by point-probability WMC.

Install: pip install pysdd
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from pysdd.sdd import SddManager, Vtree

# from dps_transform import consonant_possibility_from_samples  # TODO: wire in


# --------------------------------------------------------------------------
# 1. Per-patch possibilistic evidence
# --------------------------------------------------------------------------

@dataclass
class PatchEvidence:
    """pi, W per patch, for ONE sub-concept's ensemble run."""
    pi: np.ndarray   # shape (num_patches,), possibility in [0, 1]
    W: np.ndarray    # shape (num_patches,), specificity width in [0, 1]


def get_patch_evidence(patch_tokens, subconcept_query, ensemble_fn) -> PatchEvidence:
    """
    Delegates to ensemble_fn(patch_tokens, subconcept_query) -> (pi, W).
    ensemble_fn owns HOW pi/W get computed (e.g. concept-direction
    perturbation scored via consonant_possibility_from_samples — see
    clip_part_prototypes.py for a real implementation against CLIP text
    prototypes). Falls back to random placeholder only if no ensemble_fn
    is supplied, purely so this file stays runnable standalone.
    """
    if ensemble_fn is not None:
        pi, W = ensemble_fn(patch_tokens, subconcept_query)
        return PatchEvidence(pi=np.asarray(pi), W=np.asarray(W))
    n = patch_tokens.shape[0]
    rng = np.random.default_rng(abs(hash(subconcept_query)) % (2**32))
    return PatchEvidence(pi=rng.uniform(0, 1, n), W=rng.uniform(0, 1, n))


# --------------------------------------------------------------------------
# 2. Sub-concept definition: which patches are candidate literals for c,
#    and how they combine. Default is now spatial CONTIGUITY, not plain OR
#    — this is the paper's actual differentiating claim (local geometry),
#    not just an implementation detail.
# --------------------------------------------------------------------------

GRID_SIZE = 14  # 14x14 patch grid: 16px patches on a 224px input (LAST-ViT and
                 # CLIP ViT-B/16 both patchify this way — see clip_part_prototypes.py's
                 # grid-alignment TODO before assuming this holds for your exact pipeline)


def patch_index_to_rc(i: int, grid_size: int = GRID_SIZE) -> tuple[int, int]:
    """Row-major patch index -> (row, col) on the grid. TODO: confirm this
    matches your model's actual patch ordering — a transposed grid silently
    produces plausible-looking but wrong adjacency."""
    return i // grid_size, i % grid_size


def grid_adjacent_pairs(candidates: np.ndarray, grid_size: int = GRID_SIZE,
                         connectivity: int = 4) -> list[tuple[int, int]]:
    """
    All pairs (i, j) in `candidates` that are spatially adjacent on the
    patch grid. connectivity=4 (up/down/left/right) or 8 (+ diagonals).
    """
    coords = {int(i): patch_index_to_rc(int(i), grid_size) for i in candidates}
    if connectivity == 4:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif connectivity == 8:
        offsets = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    else:
        raise ValueError("connectivity must be 4 or 8")

    by_rc = {rc: idx for idx, rc in coords.items()}
    pairs = []
    seen = set()
    for idx, (r, c) in coords.items():
        for dr, dc in offsets:
            neighbor = by_rc.get((r + dr, c + dc))
            if neighbor is not None:
                key = tuple(sorted((idx, neighbor)))
                if key not in seen:
                    seen.add(key)
                    pairs.append(key)
    return pairs


@dataclass
class SubConcept:
    name: str
    evidence: PatchEvidence
    w_gate: float = 0.5           # TODO: tune; only patches with W_i < w_gate become literals
    combine: str = "contiguity"   # "contiguity" (default) | "or" | "k_of_n" (k_of_n still TODO)
    grid_size: int = GRID_SIZE
    connectivity: int = 4         # 4 or 8 -- only used when combine="contiguity"

    def candidate_patches(self) -> np.ndarray:
        return np.where(self.evidence.W < self.w_gate)[0]

    def weight_interval(self, patch_idx: int) -> tuple[float, float]:
        """
        Map (pi_i, W_i) -> [lower, upper] literal weight.
        Convention here: pi_i is the point estimate; W_i widens a symmetric
        band around it. This is a DESIGN CHOICE, not yet reconciled with your
        AAAI Theorem 1 specificity-width construction — see open questions.
        """
        pi = self.evidence.pi[patch_idx]
        w = self.evidence.W[patch_idx]
        lo = max(0.0, pi - w / 2)
        hi = min(1.0, pi + w / 2)
        return lo, hi


# --------------------------------------------------------------------------
# 3. Build the hierarchical SDD: concept = AND over sub-concepts,
#    each sub-concept = OR over its gated patch literals
# --------------------------------------------------------------------------

class ConceptComposer:
    def __init__(self, num_vars_hint: int = 512):
        # TODO: a real vtree order matters a lot for SDD size — right now
        # this is the trivial balanced default, not tuned for your patch
        # adjacency structure (14x14 grid locality would be a better order).
        self.mgr = SddManager(var_count=num_vars_hint, auto_gc_and_minimize=False)
        self._next_var = 1
        self._literal_map: dict[tuple[str, int], int] = {}  # (subconcept, patch_idx) -> var id

    def _literal(self, subconcept_name: str, patch_idx: int):
        key = (subconcept_name, patch_idx)
        if key not in self._literal_map:
            self._literal_map[key] = self._next_var
            self._next_var += 1
        return self.mgr.literal(self._literal_map[key])

    def build_subconcept_formula(self, sc: SubConcept):
        # INVARIANT: this file only ever builds positive-literal AND/OR
        # formulas (monotone). That's what makes wmc_interval()'s two-point
        # trick an EXACT credal lower/upper bound rather than an
        # approximation — see wmc_interval()'s docstring. Do not introduce
        # a NOT here without re-deriving the interval math (constrained
        # optimization over the credal set, à la PASOCS/PASTA, would be
        # needed instead).
        candidates = sc.candidate_patches()
        if len(candidates) == 0:
            # TODO: decide the right degenerate behavior — right now an
            # empty gated set makes the sub-concept UNSATISFIABLE, which
            # will make the whole composed concept unsatisfiable too.
            return self.mgr.false()

        # Register a literal for every candidate UP FRONT, regardless of
        # whether the chosen combine rule ends up using all of them in the
        # formula (e.g. contiguity with zero adjacent pairs uses none) —
        # wmc_interval() iterates candidate_patches() when assigning
        # weights, so every candidate needs a registered literal or that
        # lookup KeyErrors. Caught by test_empty_gated_set_is_unsatisfiable
        # when a lone unpaired candidate slipped through un-registered.
        for i in candidates:
            self._literal(sc.name, i)

        if sc.combine == "or":
            lits = [self._literal(sc.name, i) for i in candidates]
            formula = lits[0]
            for lit in lits[1:]:
                formula = formula | lit
            return formula

        elif sc.combine == "contiguity":
            # "The sub-concept is present if at least one spatially adjacent
            # PAIR of gated patches both fire" — this is the local-geometry
            # claim made concrete as a Boolean formula, not just a
            # description. Still monotone (OR of ANDs, no negation), so
            # wmc_interval()'s exactness argument is untouched.
            pairs = grid_adjacent_pairs(candidates, grid_size=sc.grid_size,
                                         connectivity=sc.connectivity)
            if not pairs:
                # No two gated patches are adjacent on the grid — evidence
                # is scattered, not localized. TODO: same degenerate-behavior
                # question as the empty-candidates case above; currently
                # this also makes the sub-concept (and thus the whole
                # concept) unsatisfiable, which is a real modeling choice,
                # not an accident — a "wing" that isn't spatially coherent
                # arguably SHOULDN'T satisfy a contiguity-gated concept.
                return self.mgr.false()
            formula = None
            for i, j in pairs:
                pair_and = self._literal(sc.name, i) & self._literal(sc.name, j)
                formula = pair_and if formula is None else (formula | pair_and)
            return formula

        elif sc.combine == "k_of_n":
            raise NotImplementedError(
                "k_of_n combine rule is still TODO — see open design questions. "
                "Note it WOULD stay monotone (threshold functions are monotone), "
                "so it's compatible with wmc_interval()'s exactness guarantee "
                "once implemented."
            )
        else:
            raise ValueError(f"Unknown combine rule: {sc.combine!r}")

    def build_concept(self, subconcepts: list[SubConcept]):
        sub_formulas = [self.build_subconcept_formula(sc) for sc in subconcepts]
        concept = sub_formulas[0]
        for f in sub_formulas[1:]:
            concept = concept & f     # AND across sub-concepts (head AND wing AND tail)
        return concept

    def wmc_interval(self, concept_node, subconcepts: list[SubConcept]) -> tuple[float, float]:
        """
        Compute [lower, upper] WMC by running weighted model counting twice:
        once with every literal weight at its lower bound (pessimistic) and
        once at its upper bound (optimistic).

        EXACTNESS: as long as the formula stays monotone (AND/OR only, no
        NOT — see build_subconcept_formula's invariant) and literal-level
        credal intervals are treated as independent, this two-point
        evaluation IS the true credal lower/upper probability, not an
        approximation. That equivalence breaks the moment negation, k-of-N
        thresholds that aren't monotone, or inter-patch correlation enter
        the picture — at that point use real constrained optimization over
        the credal set instead (PASOCS / PASTA implement this properly for
        PASP under the credal semantics).
        """
        wmc = concept_node.wmc(log_mode=False)

        def set_weights(bound: str):
            for sc in subconcepts:
                for i in sc.candidate_patches():
                    var = self._literal_map[(sc.name, i)]
                    lo, hi = sc.weight_interval(i)
                    w = lo if bound == "lower" else hi
                    wmc.set_literal_weight(self.mgr.literal(var), w)
                    wmc.set_literal_weight(self.mgr.literal(-var), 1 - w)

        set_weights("lower")
        lo_val = wmc.propagate()
        set_weights("upper")
        hi_val = wmc.propagate()
        return float(lo_val), float(hi_val)


# --------------------------------------------------------------------------
# 4. Example wiring (bird = head AND wing AND tail)
# --------------------------------------------------------------------------

def compose_bird_concept(patch_tokens, ensemble_fn: Callable):
    subconcept_names = ["head", "wing", "tail"]   # TODO: real prototype queries per part
    subconcepts = [
        SubConcept(
            name=name,
            evidence=get_patch_evidence(patch_tokens, subconcept_query=name, ensemble_fn=ensemble_fn),
        )
        for name in subconcept_names
    ]

    # size the manager exactly to the literals we'll actually create — an
    # oversized var_count hint leaves "phantom" unused variables with
    # unset (garbage-default) weights, which silently corrupts the WMC.
    n_vars_needed = sum(len(sc.candidate_patches()) for sc in subconcepts)
    composer = ConceptComposer(num_vars_hint=max(n_vars_needed, 1))
    concept_node = composer.build_concept(subconcepts)
    lo, hi = composer.wmc_interval(concept_node, subconcepts)

    print(f"bird concept WMC interval: [{lo:.4f}, {hi:.4f}]  (width={hi - lo:.4f})")
    return subconcepts, concept_node, (lo, hi)


if __name__ == "__main__":
    # Smoke test with random patch tokens — NOT a real LAST-ViT run.
    fake_patch_tokens = np.random.randn(196, 768)
    compose_bird_concept(fake_patch_tokens, ensemble_fn=None)


# --------------------------------------------------------------------------
# Open design questions (do not skip before using this on a real paper claim)
# --------------------------------------------------------------------------
# 1. Patch -> sub-concept assignment is currently "run a separate ensemble
#    per part query" (head/wing/tail as prototype directions). Is that right,
#    or should a single ensemble produce a joint (patch, part) score matrix
#    and this file just does the gating/composition on top of it?
# 2. RESOLVED (was: "combine='or' is a placeholder"): default combine is now
#    "contiguity" — a sub-concept fires only if some spatially adjacent PAIR
#    of gated patches both fire (OR over adjacent-pair ANDs on the 14x14
#    grid), not scattered singletons. This is the actual "local geometry"
#    differentiator vs. Credal SDDs / NPC / DCR / LENs, none of which have
#    any spatial structure at all. Still monotone (OR-of-ANDs, no negation),
#    so wmc_interval()'s exactness argument is untouched — see
#    build_subconcept_formula(). "at least K of N" (k_of_n) remains
#    unimplemented but would also stay monotone if added later.
# 3. weight_interval()'s pi +/- W/2 band is invented for this file. Your
#    dps_transform.py's W has a precise meaning (order-statistic specificity)
#    that may map to a *different* interval construction (e.g. W as a direct
#    upper/lower bound rather than a symmetric band) — worth 10 minutes of
#    reconciliation before trusting the numbers.
# 4. Two-point WMC is EXACT (not approximate) for monotone formulas over
#    independent per-literal possibility intervals — see wmc_interval()'s
#    docstring. This only needs revisiting if negation, non-monotone
#    combine rules, or inter-patch correlation get introduced later.
#    (Note: we deliberately don't brand this "credal" — possibility theory
#    is technically a consonant special case of credal sets, but Credal
#    SDDs (Antonucci et al. 2019; Mattei et al. 2020) already own that name
#    for a different, non-spatial mechanism — see novelty-sweep notes.)
# 5. Vtree order uses pySDD's default, NOT yet grid-adjacency-informed. The
#    contiguity combine rule (point 2) makes this more urgent, not less: an
#    adjacency-aware vtree should compile noticeably smaller/faster SDDs
#    for the OR-of-adjacent-pair-ANDs formula than a default vtree would.
#    Worth profiling once real (non-toy-grid) patch counts are in play.
