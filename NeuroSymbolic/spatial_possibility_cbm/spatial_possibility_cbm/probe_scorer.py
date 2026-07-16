"""Supervised concept directions fitted on pooled frozen-ViT features."""
from __future__ import annotations

import numpy as np


class ProbeConceptScorer:
    """Fit on image-level train features; apply directions to patch features."""

    def __init__(self, concept_names: list[str]):
        self.concept_names = concept_names
        self.W: np.ndarray | None = None
        self.b: np.ndarray | None = None

    def fit(self, pooled_feats: np.ndarray, labels: np.ndarray, l2: float = 1.0):
        from sklearn.linear_model import LogisticRegression

        d = pooled_feats.shape[1]
        C = len(self.concept_names)
        self.W = np.zeros((C, d), dtype=np.float32)
        self.b = np.zeros(C, dtype=np.float32)
        for c in range(C):
            y = labels[:, c]
            if len(np.unique(y)) < 2:
                print(f"!! concept '{self.concept_names[c]}' single-class in train; skipped")
                continue
            clf = LogisticRegression(C=1.0 / l2, max_iter=1000)
            clf.fit(pooled_feats, y)
            w = clf.coef_[0]
            self.W[c] = (w / (np.linalg.norm(w) + 1e-8)).astype(np.float32)
            self.b[c] = clf.intercept_[0]
        return self

    def concept_cosines(self, patch_tokens: np.ndarray, vocab=None) -> dict[str, np.ndarray]:
        assert self.W is not None, "call fit() on the train split first"
        pn = patch_tokens / (
            np.linalg.norm(patch_tokens, axis=1, keepdims=True) + 1e-8
        )
        return {
            name: (pn @ self.W[c]).astype(np.float32)
            for c, name in enumerate(self.concept_names)
        }

    def concept_direction_ensemble(self, patch_tokens, concept_name, M, noise, temp):
        assert self.W is not None, "call fit() on the train split first"
        c = self.concept_names.index(concept_name)
        pn = patch_tokens / (
            np.linalg.norm(patch_tokens, axis=1, keepdims=True) + 1e-8
        )
        rs = np.random.RandomState(abs(hash(concept_name)) % (2**31))
        samples = []
        for _ in range(max(M, 1)):
            direction = self.W[c] + rs.normal(0.0, noise, size=self.W[c].shape)
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            cos = pn @ direction
            samples.append(1.0 / (1.0 + np.exp(-temp * (cos - cos.mean()))))
        return np.stack(samples, axis=0)
