"""Shared utilities for Wittgenstein's Monster."""

from __future__ import annotations

import numpy as np
from scipy import stats


def to_numpy(x) -> np.ndarray:
    """Convert MLX array to numpy, forcing evaluation first."""
    import mlx.core as mx

    if isinstance(x, np.ndarray):
        return x
    mx.eval(x)
    return np.asarray(x)


def normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalize a vector or batch of vectors."""
    if v.ndim == 1:
        n = np.linalg.norm(v)
        return v / n if n > 0 else v
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return v / norms


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(normalize(a) @ normalize(b))


def spearman_rho(predicted_order: list, ground_truth_order: list) -> tuple[float, float]:
    """Spearman rank correlation between two orderings.

    Returns (rho, p_value).
    """
    rank_map = {item: i for i, item in enumerate(ground_truth_order)}
    pred_ranks = [rank_map[item] for item in predicted_order if item in rank_map]
    true_ranks = list(range(len(pred_ranks)))
    if len(pred_ranks) < 3:
        return 0.0, 1.0
    rho, p = stats.spearmanr(pred_ranks, true_ranks)
    return float(rho), float(p)


def causal_mask(seq_len: int, dtype=None):
    """Create a standard causal attention mask."""
    import mlx.core as mx

    if dtype is None:
        dtype = mx.float16
    return mx.triu(mx.full((seq_len, seq_len), -mx.inf, dtype=dtype), k=1)
