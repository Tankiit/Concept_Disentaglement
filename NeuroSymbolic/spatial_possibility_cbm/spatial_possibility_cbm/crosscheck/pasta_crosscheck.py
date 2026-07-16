"""
crosscheck_pysdd_vs_pasta.py
-----------------------------
Ground-truth cross-check: build the SAME hierarchical concept
(bird = head AND wing AND tail, each an OR over a few gated patches) in
two independent systems and confirm they agree at both corners of the
credal interval:

  1. pySDD  — concept_composition_sdd.py's ConceptComposer.wmc_interval()
  2. PASTA  — an independent PASP credal-semantics solver
              (github.com/damianoazzolini/pasta), given the SAME point
              probabilities at each corner (all-lower / all-upper).

Why this validates something real: pySDD's interval comes from running
weighted model counting twice with hand-picked corner weights (see
wmc_interval()'s exactness argument for monotone formulas). PASTA computes
its own lower/upper bound via a completely different mechanism — projected
answer-set enumeration over "worlds" — but for a NORMAL program with no
choice/disjunctive rules (which is what a plain AND/OR formula compiles
to), every world has exactly one answer set, so PASTA's lower and upper
collapse to the same single number: the standard weighted probability of
the query. That number should exactly match pySDD's WMC at the
corresponding corner. Agreement here checks the ARITHMETIC (is the pySDD
wiring bug-free), not the monotone-interval THEORY (which is a separate,
already-argued claim, not something either solver can validate for you).

Small by design (3 sub-concepts x 4 candidate patches = 12 literals) so
PASTA's exact enumeration stays fast; this is a correctness check, not a
scale test.
"""

from __future__ import annotations
import subprocess
import tempfile
import os

from spatial_possibility_cbm.concept_composition_sdd import ConceptComposer, SubConcept, PatchEvidence


# --------------------------------------------------------------------------
# Small, hand-inspectable evidence — not random, so the numbers in the ASP
# program and the pySDD run are traceably the same for debugging.
# --------------------------------------------------------------------------

EVIDENCE = {
    "head": {"pi": [0.90, 0.20, 0.55, 0.10], "W": [0.10, 0.40, 0.30, 0.60]},
    "wing": {"pi": [0.80, 0.70, 0.15, 0.05], "W": [0.20, 0.10, 0.50, 0.70]},
    "tail": {"pi": [0.60, 0.30, 0.25, 0.85], "W": [0.30, 0.50, 0.40, 0.10]},
}
W_GATE = 0.55  # only patches with W < W_GATE become literals


def make_subconcepts() -> list[SubConcept]:
    subs = []
    for name, d in EVIDENCE.items():
        import numpy as np
        subs.append(SubConcept(
            name=name,
            evidence=PatchEvidence(pi=np.array(d["pi"]), W=np.array(d["W"])),
            w_gate=W_GATE,
            combine="or",  # this cross-check's ASP program encodes OR specifically —
                           # contiguity is SubConcept's new default elsewhere, but this
                           # test targets the OR arithmetic, so pin it explicitly.
        ))
    return subs


# --------------------------------------------------------------------------
# pySDD side
# --------------------------------------------------------------------------

def run_pysdd() -> tuple[float, float]:
    subconcepts = make_subconcepts()
    n_vars = sum(len(sc.candidate_patches()) for sc in subconcepts)
    composer = ConceptComposer(num_vars_hint=max(n_vars, 1))
    concept_node = composer.build_concept(subconcepts)
    return composer.wmc_interval(concept_node, subconcepts)


# --------------------------------------------------------------------------
# PASTA side — same formula, point weights fixed at one corner per run
# --------------------------------------------------------------------------

def build_pasta_program(bound: str) -> str:
    """
    bound: "lower" or "upper" — which corner of each literal's
    weight_interval() to fix as PASTA's (point-valued) probabilistic fact.
    """
    subconcepts = make_subconcepts()
    lines = []
    for sc in subconcepts:
        for i in sc.candidate_patches():
            lo, hi = sc.weight_interval(int(i))
            w = lo if bound == "lower" else hi
            w = min(max(w, 1e-6), 1.0)  # PASTA requires 0 < p <= 1
            lines.append(f"{w:.6f}::patch({sc.name},p{i}).")
    lines.append("subconcept_true(C) :- patch(C,I).")
    lines.append("concept :- subconcept_true(head), subconcept_true(wing), subconcept_true(tail).")
    return "\n".join(lines)


def run_pasta(bound: str) -> float:
    program = build_pasta_program(bound)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".lp", delete=False) as f:
        f.write(program)
        path = f.name
    try:
        result = subprocess.run(
            ["pastasolver", path, "--query=concept"],
            capture_output=True, text=True, timeout=60,
        )
        out = result.stdout
        # PASTA prints "Lower probability for the query: X" /
        # "Upper probability for the query: Y" — for this normal
        # (non-disjunctive) program the two should coincide.
        lower = upper = None
        for line in out.splitlines():
            if "Lower probability == upper probability" in line:
                val = float(line.split(":")[-1].strip())
                lower = upper = val
            elif "Lower probability" in line:
                lower = float(line.split(":")[-1].strip())
            elif "Upper probability" in line:
                upper = float(line.split(":")[-1].strip())
        if lower is None or upper is None:
            raise RuntimeError(f"Could not parse PASTA output:\n{out}\n{result.stderr}")
        assert abs(lower - upper) < 1e-6, (
            f"Expected PASTA lower==upper for a normal program, got {lower} vs {upper} "
            f"— program may have accidentally introduced a choice/disjunctive rule."
        )
        return lower
    finally:
        os.unlink(path)


if __name__ == "__main__":
    pysdd_lo, pysdd_hi = run_pysdd()
    print(f"pySDD interval:  [{pysdd_lo:.6f}, {pysdd_hi:.6f}]")

    pasta_lo = run_pasta("lower")
    pasta_hi = run_pasta("upper")
    print(f"PASTA corners:   lower={pasta_lo:.6f}  upper={pasta_hi:.6f}")

    tol = 1e-4
    lo_ok = abs(pysdd_lo - pasta_lo) < tol
    hi_ok = abs(pysdd_hi - pasta_hi) < tol
    print(f"\nlower match: {lo_ok}  (|{pysdd_lo:.6f} - {pasta_lo:.6f}| = {abs(pysdd_lo-pasta_lo):.2e})")
    print(f"upper match: {hi_ok}  (|{pysdd_hi:.6f} - {pasta_hi:.6f}| = {abs(pysdd_hi-pasta_hi):.2e})")
    if lo_ok and hi_ok:
        print("\nCROSS-CHECK PASSED — pySDD's two-point WMC interval matches an")
        print("independent PASP credal-semantics solver at both corners.")
    else:
        print("\nCROSS-CHECK FAILED — investigate before trusting pySDD numbers.")
