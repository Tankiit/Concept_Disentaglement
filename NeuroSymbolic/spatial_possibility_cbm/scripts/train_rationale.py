from __future__ import annotations

"""
train_rationale.py  (Option 1 - sequential, gate->concept)

CLARITY-style faithful concept rationales on real cached LAST-ViT features.
Predicts each concept FROM THE GATED REGION ONLY -> sufficiency by construction;
COMPREHENSIVENESS (removing the region collapses the score) is the untrained,
honest faithfulness test and the reported headline.

Init: relevance layer seeded from the fitted probe (W (7,768), b (7,)) so the
gated model starts at the probe's rho=0.5 and only has to LOCALIZE, not relearn
the concept direction. The probe is both init and baseline.

One SHARED gate (per-image rationale region), read by all concepts.
Image-level labels only -> the gate localizes with NO location supervision;
"selects the RIGHT region" is verified on CUB box-hit, not here. Here we can
only claim "selects a SUFFICIENT region."
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

GRID = 14


def adjacency_pairs(grid=GRID):
    pr = []
    for r in range(grid):
        for c in range(grid):
            p = r * grid + c
            if c + 1 < grid:
                pr.append((p, p + 1))
            if r + 1 < grid:
                pr.append((p, p + grid))
    return torch.tensor(pr, dtype=torch.long)


class RationaleHead(nn.Module):
    def __init__(
        self, d, probe_W, probe_b, temp=1.0, hidden_dim=32, dropout=0.3,
        min_hard_mass=0.05,
    ):
        super().__init__()
        C = probe_W.shape[0]
        self.relevance = nn.Linear(d, C)
        with torch.no_grad():
            self.relevance.weight.copy_(torch.tensor(probe_W, dtype=torch.float32))
            self.relevance.bias.copy_(torch.tensor(probe_b, dtype=torch.float32))
        self.gate = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.register_buffer("pairs", adjacency_pairs())
        self.temp = temp
        self.min_hard_patches = max(1, int(np.ceil(min_hard_mass * GRID * GRID)))

    @staticmethod
    def _neighbors(index):
        r, c = divmod(index, GRID)
        if r > 0:
            yield index - GRID
        if r + 1 < GRID:
            yield index + GRID
        if c > 0:
            yield index - 1
        if c + 1 < GRID:
            yield index + 1

    def _connected_hard_mask(self, logits):
        """Largest thresholded component, grown contiguously to the size floor."""
        active = set(torch.nonzero(torch.sigmoid(logits) > 0.5, as_tuple=False).flatten().tolist())
        components = []
        unseen = set(active)
        while unseen:
            component = {unseen.pop()}
            stack = list(component)
            while stack:
                p = stack.pop()
                for q in self._neighbors(p):
                    if q in unseen:
                        unseen.remove(q)
                        component.add(q)
                        stack.append(q)
            components.append(component)
        if components:
            region = max(components, key=len)
        else:
            region = {int(logits.argmax().item())}

        # Add the strongest boundary patch at each step. This preserves one
        # connected component while guaranteeing the hard minimum mass.
        while len(region) < self.min_hard_patches:
            boundary = {
                q for p in region for q in self._neighbors(p) if q not in region
            }
            best = max(boundary, key=lambda q: float(logits[q]))
            region.add(best)
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[list(region)] = True
        return mask

    def _gate(self, patches, hard):
        logits = self.gate(patches).squeeze(-1)
        if not hard:
            return torch.sigmoid(logits / self.temp)
        return torch.stack([self._connected_hard_mask(row) for row in logits]).float()

    def concept_from_region(self, patches, mask):
        """
        Predict concept from patches weighted by mask (region-only readout).
        Soft masked mean-pool (not a hard -inf max) so gradients survive and an
        (almost) empty mask degrades smoothly instead of exploding to garbage.
        """
        pi = self.relevance(patches)
        w = mask.unsqueeze(-1)
        num = (w * pi).sum(dim=1)
        den = w.sum(dim=1).clamp_min(1e-6)
        return num / den

    def forward(self, patches, hard=False):
        g = self._gate(patches, hard)
        concept = self.concept_from_region(patches, g)
        return concept, g

    def reg(self, g, lam_sparse=1e-4, lam_tv=1e-2, min_mass=0.05, lam_floor=1.0):
        """
        Sparsity (small), TV (contiguity), AND a minimum-mass hinge that PUNISHES
        the empty-gate degenerate. Without the floor, sparsity wins and the gate
        selects nothing (the bug the smoke test caught).
        """
        p_i, q_i = self.pairs[:, 0], self.pairs[:, 1]
        sparsity = g.mean()
        tv = (g[:, p_i] - g[:, q_i]).abs().mean()
        floor = F.relu(min_mass - g.mean(dim=1)).mean()
        return lam_sparse * sparsity + lam_tv * tv + lam_floor * floor


def train(
    head, patches, labels, valid=None, epochs=200, lr=1e-3, batch_size=16,
    device="cpu", scheduler_start=25, min_lr=1e-5, weight_decay=1e-4,
):
    head.to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    X = torch.as_tensor(patches, dtype=torch.float32)
    Y = torch.as_tensor(labels, dtype=torch.float32)
    generator = torch.Generator().manual_seed(0)
    for ep in range(epochs):
        if ep >= scheduler_start:
            progress = (ep - scheduler_start) / max(epochs - scheduler_start - 1, 1)
            scheduled_lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + np.cos(np.pi * progress)
            )
            for group in opt.param_groups:
                group["lr"] = scheduled_lr
        head.temp = max(0.05, 1.0 * (1 - ep / epochs))
        losses, gate_means = [], []
        for idx in torch.randperm(len(X), generator=generator).split(batch_size):
            xb, yb = X[idx].to(device), Y[idx].to(device)
            concept, g = head(xb, hard=False)
            loss = F.binary_cross_entropy_with_logits(concept, yb) + head.reg(g)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.detach().cpu())
            gate_means.append(g.mean().detach().cpu())
        if ep % 50 == 0 or ep == epochs - 1:
            print(
                f"ep {ep:3d} | loss {torch.stack(losses).mean().item():.4f} "
                f"| temp {head.temp:.2f} | gate_mean "
                f"{torch.stack(gate_means).mean().item():.3f} "
                f"| lr {opt.param_groups[0]['lr']:.2e}"
            )
    return head.cpu()


# ---- faithfulness on held-out test (the honest metrics) -------------------
def spearman(x, y):
    from scipy.stats import spearmanr

    return float(spearmanr(x, y).correlation)


def evaluate(head, patches, labels_numeric):
    """Returns per-concept: gated rho, sufficiency, comprehensiveness."""
    head.eval()
    X = torch.tensor(patches, dtype=torch.float32)
    with torch.no_grad():
        concept, g = head(X, hard=True)
        # sufficiency: score from gated region (== forward, by construction)
        suff_score = torch.sigmoid(concept)
        # comprehensiveness: score with gated region REMOVED
        inv = 1.0 - g
        comp_removed = torch.sigmoid(head.concept_from_region(X, inv))
        # full-image score (all patches) as the reference
        full = torch.sigmoid(head.concept_from_region(X, torch.ones_like(g)))
    out = {}
    C = concept.shape[1]
    for c in range(C):
        rho = spearman(suff_score[:, c].numpy(), labels_numeric[:, c])
        # comprehensiveness = full - (score when region removed); want LARGE
        comp = float((full[:, c] - comp_removed[:, c]).mean())
        # sufficiency = full - (score from region only); want ~0
        suff = float((full[:, c] - suff_score[:, c]).mean())
        out[c] = {"gated_rho": rho, "comprehensiveness": comp, "sufficiency": suff}
    return out


if __name__ == "__main__":
    # SMOKE: real SHAPES (196,768), probe-shaped init, planted contiguous signal
    # per concept, so gate must localize and comprehensiveness must be > 0.
    torch.manual_seed(0)
    np.random.seed(0)
    N, P, d, C = 300, GRID * GRID, 768, 7
    patches = (np.random.randn(N, P, d) * 0.4).astype(np.float32)
    y = (np.random.rand(N, C) > 0.6).astype(np.float32)
    # each concept c has its own 3x3 blob; plant signal in positive images
    for c in range(C):
        r0, c0 = (c % 4) + 4, (c % 3) + 5
        blob = [r * GRID + col for r in range(r0, r0 + 3) for col in range(c0, c0 + 3)]
        probe_dir = np.random.randn(d)
        for n in range(N):
            if y[n, c]:
                patches[n, blob] += probe_dir * 1.2
    probe_W = np.random.randn(C, d).astype(np.float32)
    probe_W /= np.linalg.norm(probe_W, axis=1, keepdims=True)
    probe_b = np.zeros(C, dtype=np.float32)

    head = RationaleHead(d, probe_W, probe_b)
    train(head, patches, y, epochs=150)
    res = evaluate(head, patches, y)
    print("\nper-concept (smoke; planted signal):")
    for c, m in res.items():
        print(
            f"  c{c}: rho={m['gated_rho']:.3f} comp={m['comprehensiveness']:.3f} "
            f"suff={m['sufficiency']:.3f}"
        )
    comps = [m["comprehensiveness"] for m in res.values()]
    print(
        f"\nmean comprehensiveness={np.mean(comps):.3f} "
        f"(want > 0: removing the gated region should hurt)"
    )
    print("SMOKE OK" if np.mean(comps) > 0 else "!! gate not load-bearing")
