from __future__ import annotations

"""
learnable_span.py

The learnable extension of the patch-rationale: learn WHICH patches enter the
per-image span (CLARITY's span selector, 2-D), while keeping the logical
composition symbolic and EXACT at inference.

Design contract:
  TRAIN  -> soft gate g in [0,1], subgradient max/min, TV + sparsity penalties.
  INFER  -> harden g to {0,1}, largest connected component = the per-image
            rationale, run the EXACT possibilistic interval over it.
The exactness theorem is an inference-time statement, so it is untouched.

Special case (this is the July submission):
    g_p := 1[W_p < w_gate]     # unsupervised, no training, no selector net
i.e. the hard W-gate is the g in {0,1} limit of the learned gate below. The
learnable model GENERALIZES the submission; it does not replace it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

GRID = 14


def adjacency_pairs(grid: int = GRID) -> torch.Tensor:
    """Build 4-connected adjacency pairs (p, q) with p < q on a square grid."""
    pairs = []
    for r in range(grid):
        for c in range(grid):
            p = r * grid + c
            if c + 1 < grid:
                pairs.append((p, p + 1))  # right
            if r + 1 < grid:
                pairs.append((p, p + grid))  # down
    return torch.tensor(pairs, dtype=torch.long)


class LearnableSpanCBM(nn.Module):
    """
    ONE per-image span (answer to Q1: per-image, for spatial faithfulness),
    shared across concepts. Per-concept relevance still differs; the SELECTION
    is a single image-level region the diagnosis rests on.
    """

    def __init__(self, d_model: int, n_concepts: int, temp: float = 1.0):
        super().__init__()
        self.relevance = nn.Linear(d_model, n_concepts)  # pi_p(c) scorer
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.register_buffer("pairs", adjacency_pairs())
        self.temp = temp

    def forward(self, patch_tokens: torch.Tensor, hard: bool = False):
        # patch_tokens: (B, 196, d)
        pi = torch.sigmoid(self.relevance(patch_tokens))  # (B,196,C) relevance
        logits = self.gate(patch_tokens).squeeze(-1)  # (B,196) selection logits

        if hard:
            # INFERENCE: straight-through hard gate. TODO: replace with
            # threshold + largest-connected-component (patch_rationale.py) so
            # the span is genuinely connected, not just top-activated.
            g = (torch.sigmoid(logits) > 0.5).float()
        else:
            # TRAIN: soft gate. TODO(tanmoy): Gumbel/HardKuma vs plain sigmoid.
            # Plain sigmoid is the least fuss and often enough; HardKuma gives
            # you actual zeros (true sparsity) which matches CLARITY's stance.
            g = torch.sigmoid(logits / self.temp)  # (B,196)

        # selected relevance: soft-AND of "selected" and "relevant"
        s = g.unsqueeze(-1) * pi  # (B,196,C)

        # symbolic composition: concept_pi(c) = max over adjacent pairs of
        # min(s_p, s_q). max/min are differentiable by subgradient.
        p_idx, q_idx = self.pairs[:, 0], self.pairs[:, 1]
        s_p = s[:, p_idx, :]  # (B,E,C)
        s_q = s[:, q_idx, :]  # (B,E,C)
        pair_and = torch.minimum(s_p, s_q)  # AND
        concept_pi = pair_and.max(dim=1).values  # OR over pairs -> (B,C)
        return concept_pi, g, pi

    def losses(self, concept_pi, g, y, lam_sparse: float = 1e-3, lam_tv: float = 1e-2):
        # task loss on concepts (CBM head is downstream; sketch a concept BCE)
        task = F.binary_cross_entropy(concept_pi, y)  # (B,C) targets

        # sparsity: keep the span small (few selected patches)
        sparsity = g.mean()

        # CONTIGUITY = 2-D total variation over the grid. This is CLARITY's
        # fused-lasso span penalty, ported to the patch grid. Pulls the learned
        # mask toward CONNECTED blobs -> the learned version of your
        # connected-component rationale.
        p_idx, q_idx = self.pairs[:, 0], self.pairs[:, 1]
        tv = (g[:, p_idx] - g[:, q_idx]).abs().mean()

        loss = task + lam_sparse * sparsity + lam_tv * tv
        return loss, {
            "task": task.item(),
            "sparsity": sparsity.item(),
            "tv": tv.item(),
        }


# ---------------------------------------------------------------------------
# Where SDD re-enters (only if you want gradients through EXACT composition):
#   The max/min above is already differentiable, so for THIS monotone fragment
#   you do NOT need a compiled circuit to train. You'd reach for a differentiable
#   WMC / semantic loss (compile -> WMC differentiable in literal weights) only
#   when you add a fragment where subgradient max/min is NOT the semantics you
#   want (e.g. probabilistic composition, or a non-idempotent t-norm). Keep the
#   SDD for that case; don't pay for it here.
# ---------------------------------------------------------------------------
