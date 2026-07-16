"""
Core test suite — runs with just `pip install -e .[dev]`, no pastasolver
CLI required (that lives in tests/test_pasta_crosscheck.py, skipped if the
CLI isn't on PATH).
"""
from __future__ import annotations
import itertools
import numpy as np
import pytest

from spatial_possibility_cbm.concept_composition_sdd import (
    ConceptComposer, SubConcept, PatchEvidence, get_patch_evidence, grid_adjacent_pairs,
)


def _small_subconcepts():
    """3 sub-concepts x 3 candidate patches — small enough to brute-force.
    combine='or' pinned explicitly: brute_force_wmc_interval below encodes
    OR semantics specifically; contiguity is SubConcept's new default
    elsewhere and gets its own brute-force check further down."""
    rng = np.random.default_rng(42)
    subs = []
    for name in ["head", "wing", "tail"]:
        pi = rng.uniform(0.1, 0.9, 3)
        W = rng.uniform(0.0, 0.3, 3)  # keep W small so weight_interval stays in [0,1]
        subs.append(SubConcept(name=name, evidence=PatchEvidence(pi=pi, W=W), w_gate=1.0, combine="or"))
    return subs


def brute_force_wmc_interval(subconcepts, composer: ConceptComposer, concept_node) -> tuple[float, float]:
    """
    Independent ground truth: enumerate every 2^n literal assignment by
    hand and sum weighted satisfying assignments, at both the all-lower
    and all-upper corner weightings. This does NOT use pysdd's wmc() at
    all, so it validates the SDD-based wmc_interval() against a totally
    separate (if exponential) computation.
    """
    lits = sorted(composer._literal_map.items(), key=lambda kv: kv[1])  # [(key, var_id), ...]
    n = len(lits)

    def corner_weight(key, bound):
        sc_name, patch_idx = key
        sc = next(s for s in subconcepts if s.name == sc_name)
        lo, hi = sc.weight_interval(patch_idx)
        return lo if bound == "lower" else hi

    def evaluate(bound):
        weights = {key: corner_weight(key, bound) for key, _ in lits}
        total = 0.0
        for bits in itertools.product([0, 1], repeat=n):
            assignment = {lits[i][0]: bits[i] for i in range(n)}
            # subconcept true iff OR over its assigned-1 literals
            sub_true = {}
            for sc in subconcepts:
                cand = sc.candidate_patches()
                sub_true[sc.name] = any(assignment.get((sc.name, int(i)), 0) == 1 for i in cand)
            if all(sub_true.values()):
                prob = 1.0
                for key, bit in assignment.items():
                    w = weights[key]
                    prob *= w if bit == 1 else (1 - w)
                total += prob
        return total

    return evaluate("lower"), evaluate("upper")


def test_wmc_interval_matches_brute_force_enumeration():
    subconcepts = _small_subconcepts()
    n_vars = sum(len(sc.candidate_patches()) for sc in subconcepts)
    composer = ConceptComposer(num_vars_hint=max(n_vars, 1))
    concept_node = composer.build_concept(subconcepts)
    sdd_lo, sdd_hi = composer.wmc_interval(concept_node, subconcepts)
    bf_lo, bf_hi = brute_force_wmc_interval(subconcepts, composer, concept_node)
    assert sdd_lo == pytest.approx(bf_lo, abs=1e-9)
    assert sdd_hi == pytest.approx(bf_hi, abs=1e-9)


def test_wmc_interval_lower_le_upper():
    subconcepts = _small_subconcepts()
    n_vars = sum(len(sc.candidate_patches()) for sc in subconcepts)
    composer = ConceptComposer(num_vars_hint=max(n_vars, 1))
    concept_node = composer.build_concept(subconcepts)
    lo, hi = composer.wmc_interval(concept_node, subconcepts)
    assert lo <= hi + 1e-9


def test_zero_width_collapses_to_point_wmc():
    """W=0 for every literal should make lower == upper (the NPC-equivalent
    degenerate case discussed for the AAAI related-work positioning)."""
    rng = np.random.default_rng(0)
    subs = [SubConcept(name=n, evidence=PatchEvidence(pi=rng.uniform(0.2, 0.8, 3), W=np.zeros(3)), w_gate=1.0)
            for n in ["head", "wing", "tail"]]
    n_vars = sum(len(sc.candidate_patches()) for sc in subs)
    composer = ConceptComposer(num_vars_hint=max(n_vars, 1))
    concept_node = composer.build_concept(subs)
    lo, hi = composer.wmc_interval(concept_node, subs)
    assert lo == pytest.approx(hi, abs=1e-9)


def test_get_patch_evidence_differs_per_subconcept_name():
    """Regression test for the seed-reuse bug: head/wing/tail must not get
    identical placeholder evidence when no ensemble_fn is supplied."""
    patch_tokens = np.zeros((10, 4))
    ev_head = get_patch_evidence(patch_tokens, "head", ensemble_fn=None)
    ev_wing = get_patch_evidence(patch_tokens, "wing", ensemble_fn=None)
    assert not np.allclose(ev_head.pi, ev_wing.pi)


def test_get_patch_evidence_delegates_to_ensemble_fn():
    """Regression test for the ignored-ensemble_fn bug."""
    patch_tokens = np.zeros((5, 4))
    called_with = {}

    def fake_ensemble_fn(pt, query):
        called_with["query"] = query
        return np.full(5, 0.5), np.full(5, 0.1)

    ev = get_patch_evidence(patch_tokens, "wing", ensemble_fn=fake_ensemble_fn)
    assert called_with["query"] == "wing"
    assert np.allclose(ev.pi, 0.5)
    assert np.allclose(ev.W, 0.1)


def test_empty_gated_set_is_unsatisfiable():
    """candidate_patches() empty -> sub-concept formula is False -> whole
    concept unsatisfiable. Documents the degenerate-behavior TODO rather
    than letting it fail silently."""
    subs = [
        SubConcept(name="head", evidence=PatchEvidence(pi=np.array([0.9]), W=np.array([0.9])), w_gate=0.1),
        SubConcept(name="wing", evidence=PatchEvidence(pi=np.array([0.9]), W=np.array([0.05])), w_gate=0.5),
    ]
    n_vars = sum(len(sc.candidate_patches()) for sc in subs)
    composer = ConceptComposer(num_vars_hint=max(n_vars, 1))
    concept_node = composer.build_concept(subs)
    lo, hi = composer.wmc_interval(concept_node, subs)
    assert lo == 0.0 and hi == 0.0


# --------------------------------------------------------------------------
# Spatial contiguity combine rule (the new default) — independent checks
# --------------------------------------------------------------------------

def test_grid_adjacent_pairs_on_small_grid():
    """3x3 toy grid (grid_size=3), patches 0..8, row-major:
        0 1 2
        3 4 5
        6 7 8
    Patch 4 (center) should be adjacent to 1,3,5,7 under 4-connectivity."""
    candidates = np.array([0, 1, 3, 4, 5, 7, 8])
    pairs = grid_adjacent_pairs(candidates, grid_size=3, connectivity=4)
    pairs_set = {frozenset(p) for p in pairs}
    for neighbor in [1, 3, 5, 7]:
        assert frozenset((4, neighbor)) in pairs_set
    # 0 and 8 are diagonal, not 4-adjacent
    assert frozenset((0, 8)) not in pairs_set


def _brute_force_contiguity_wmc(subconcepts, composer: ConceptComposer) -> tuple[float, float]:
    """Independent ground truth for combine='contiguity': enumerate all
    2^n literal assignments; a sub-concept is true iff some grid-adjacent
    PAIR of its candidates are both assigned 1. Does not call pysdd wmc()."""
    lits = sorted(composer._literal_map.items(), key=lambda kv: kv[1])
    n = len(lits)

    def corner_weight(key, bound):
        sc_name, patch_idx = key
        sc = next(s for s in subconcepts if s.name == sc_name)
        lo, hi = sc.weight_interval(patch_idx)
        return lo if bound == "lower" else hi

    def evaluate(bound):
        weights = {key: corner_weight(key, bound) for key, _ in lits}
        total = 0.0
        for bits in itertools.product([0, 1], repeat=n):
            assignment = {lits[i][0]: bits[i] for i in range(n)}
            sub_true = {}
            for sc in subconcepts:
                cand = sc.candidate_patches()
                pairs = grid_adjacent_pairs(cand, grid_size=sc.grid_size, connectivity=sc.connectivity)
                sub_true[sc.name] = any(
                    assignment.get((sc.name, i), 0) == 1 and assignment.get((sc.name, j), 0) == 1
                    for i, j in pairs
                )
            if all(sub_true.values()):
                prob = 1.0
                for key, bit in assignment.items():
                    w = weights[key]
                    prob *= w if bit == 1 else (1 - w)
                total += prob
        return total

    return evaluate("lower"), evaluate("upper")


def test_contiguity_wmc_matches_independent_brute_force():
    """Uses a small 3x3 grid so adjacency + candidate gating stays tiny
    enough to enumerate exhaustively, and picks pi/W so several patches
    per sub-concept are gated in (otherwise contiguity trivially unsat)."""
    rng = np.random.default_rng(7)
    subs = []
    for name in ["head", "wing"]:
        pi = rng.uniform(0.3, 0.9, 9)   # all 9 patches on a 3x3 grid
        W = rng.uniform(0.0, 0.2, 9)    # small W keeps most patches gated in (w_gate=1.0 below)
        subs.append(SubConcept(name=name, evidence=PatchEvidence(pi=pi, W=W),
                                w_gate=1.0, combine="contiguity", grid_size=3, connectivity=4))
    n_vars = sum(len(sc.candidate_patches()) for sc in subs)
    composer = ConceptComposer(num_vars_hint=max(n_vars, 1))
    concept_node = composer.build_concept(subs)
    sdd_lo, sdd_hi = composer.wmc_interval(concept_node, subs)
    bf_lo, bf_hi = _brute_force_contiguity_wmc(subs, composer)
    assert sdd_lo == pytest.approx(bf_lo, abs=1e-9)
    assert sdd_hi == pytest.approx(bf_hi, abs=1e-9)


def test_contiguity_unsat_when_candidates_scattered():
    """If the gated patches are all mutually non-adjacent, contiguity
    should make the sub-concept (and thus the whole concept) unsatisfiable
    — this is a deliberate modeling choice, documented in build_subconcept_formula,
    not a bug."""
    # On a 5x5 grid, corners 0, 4, 20, 24 are mutually non-adjacent under 4-connectivity.
    pi = np.zeros(25)
    W = np.full(25, 1.0)  # gate everything OUT except the four scattered corners
    for idx in [0, 4, 20, 24]:
        W[idx] = 0.0
        pi[idx] = 0.9
    sc = SubConcept(name="scattered", evidence=PatchEvidence(pi=pi, W=W),
                     w_gate=0.5, combine="contiguity", grid_size=5, connectivity=4)
    n_vars = len(sc.candidate_patches())
    composer = ConceptComposer(num_vars_hint=max(n_vars, 1))
    formula = composer.build_subconcept_formula(sc)
    assert formula == composer.mgr.false()
