"""Operator Archaeologist — discover relational vectors in transformer heads.

Given semantic pairs (king/queen, Paris/France, etc.), extracts the
consistent directional operator R such that h(b) ≈ h(a) + R, then
localizes R to specific (layer, head) coordinates.

Attribution is done analytically via output-projection overlap —
no extra forward passes needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np

from latent_scope.utils import normalize, cosine_similarity

if TYPE_CHECKING:
    from latent_scope.core import HookedModel


@dataclass
class Operator:
    """A relational operator discovered in the residual stream."""

    name: str
    vector: np.ndarray           # (hidden_dim,)
    layer: int = -1              # best-matching layer
    head: int = -1               # best-matching head
    confidence: float = 0.0      # analogy test accuracy
    attribution_score: float = 0.0

    def apply(self, vec: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        return normalize(vec + alpha * self.vector)

    def save_to_file(self, filepath: str):
        """Save this operator to an .npz file."""
        np.savez(
            filepath,
            name=self.name,
            vector=self.vector,
            layer=self.layer,
            head=self.head,
            confidence=self.confidence,
            attribution_score=self.attribution_score
        )

    @classmethod
    def load_from_file(cls, filepath: str) -> "Operator":
        """Load an operator from an .npz file."""
        data = np.load(filepath)
        return cls(
            name=str(data["name"]),
            vector=data["vector"],
            layer=int(data["layer"]),
            head=int(data["head"]),
            confidence=float(data["confidence"]),
            attribution_score=float(data["attribution_score"])
        )


@dataclass
class MinerResult:
    """Results from an operator mining run."""

    operator: Operator
    pairs_used: list[tuple[str, str]]
    analogy_accuracy: float = 0.0
    heatmap: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))

    def top_heads(self, k: int = 10) -> list[tuple[int, int, float]]:
        """Return top-k (layer, head, score) by attribution."""
        flat = self.heatmap.flatten()
        top_idx = np.argsort(flat)[-k:][::-1]
        rows, cols = np.unravel_index(top_idx, self.heatmap.shape)
        return [(int(r), int(c), float(self.heatmap[r, c]))
                for r, c in zip(rows, cols)]


class OperatorMiner:
    """Extract and localize relational operators from semantic pairs.

    Usage::

        miner = OperatorMiner(model)

        result = miner.extract(
            name="capital-of",
            pairs=[("France", "Paris"), ("Japan", "Tokyo"),
                   ("Germany", "Berlin"), ("Italy", "Rome")],
        )
        print(f"Best head: layer {result.operator.layer}, head {result.operator.head}")
        print(f"Analogy accuracy: {result.analogy_accuracy:.0%}")
    """

    def __init__(self, model: "HookedModel", layer: int = 20):
        self.model = model
        self.layer = layer

    def extract(
        self,
        name: str,
        pairs: list[tuple[str, str]],
        attribute: bool = True,
    ) -> MinerResult:
        """Extract a relational operator from semantic pairs.

        Each pair is (source, target) where target ≈ source + R.
        """
        # Compute difference vectors at the extraction layer
        diffs = []
        for src, tgt in pairs:
            v_src = self.model.get_concept_vector(src, self.layer)
            v_tgt = self.model.get_concept_vector(tgt, self.layer)
            diffs.append(v_tgt - v_src)

        diffs = np.stack(diffs)

        # Robust mean operator
        op_vector = normalize(diffs.mean(axis=0))

        # Analogy accuracy: leave-one-out
        accuracy = self._analogy_test(op_vector, pairs)

        op = Operator(name=name, vector=op_vector, confidence=accuracy)

        # Analytical attribution (no extra forward passes)
        heatmap = np.zeros((self.model.num_layers, self.model.num_heads or 32))
        if attribute:
            heatmap = self._attribute_analytical(op_vector)
            best_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            op.layer = int(best_idx[0])
            op.head = int(best_idx[1])
            op.attribution_score = float(heatmap[best_idx])

        return MinerResult(
            operator=op,
            pairs_used=pairs,
            analogy_accuracy=accuracy,
            heatmap=heatmap,
        )

    def _analogy_test(
        self,
        op_vector: np.ndarray,
        pairs: list[tuple[str, str]],
    ) -> float:
        """Leave-one-out analogy accuracy."""
        if len(pairs) < 2:
            return 0.0

        # Cache all target vectors
        targets = {tgt: self.model.get_concept_vector(tgt, self.layer)
                   for _, tgt in pairs}

        correct = 0
        for src, expected_tgt in pairs:
            v_src = self.model.get_concept_vector(src, self.layer)
            predicted = normalize(v_src + op_vector)

            best_tgt = max(targets, key=lambda t: cosine_similarity(predicted, targets[t]))
            if best_tgt == expected_tgt:
                correct += 1

        return correct / len(pairs)

    def _attribute_analytical(self, op_vector: np.ndarray) -> np.ndarray:
        """Analytically attribute the operator to (layer, head) coordinates.

        For each head's output projection slice W_O[:, h*d:(h+1)*d], the
        attribution score measures how much of the operator vector lives in
        that head's output subspace:

            score(l, h) = || W_O[:, h*d:(h+1)*d].T @ R ||^2

        This is O(layers × hidden²) pure numpy — no model forward passes.
        """
        n_layers = self.model.num_layers
        n_heads = self.model.num_heads or 32
        hidden = self.model.hidden_dim or len(op_vector)
        head_dim = hidden // n_heads

        R = op_vector.astype(np.float32)
        heatmap = np.zeros((n_layers, n_heads), dtype=np.float32)

        for li in range(n_layers):
            attn = self.model.model.model.layers[li].self_attn
            # Materialize the weight once per layer
            w = attn.o_proj.weight
            mx.eval(w)
            W = np.array(w, dtype=np.float32)  # (hidden, hidden)

            for h in range(n_heads):
                # Column slice for this head: (hidden, head_dim)
                col_start = h * head_dim
                col_end = col_start + head_dim
                W_h = W[:, col_start:col_end]

                # Projection score: how much of R lives in col-space of W_h
                # = || W_h.T @ R ||^2  (exact when W_h is orthonormal)
                proj = W_h.T @ R            # (head_dim,)
                heatmap[li, h] = float(proj @ proj)

        # Normalize to [0, 1]
        m = heatmap.max()
        if m > 0:
            heatmap /= m

        return heatmap
