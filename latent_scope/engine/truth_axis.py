"""Truth Axis — does the model know what's true?

Calibrate a truth-vs-falsehood axis using known true, false, and gibberish
statements.  Then project any new statement onto this axis at every layer.
The layer trajectory reveals where in the network the model develops its
understanding of truth — and whether it can distinguish fact from plausible
fiction from nonsense.

The "lasagna ordering" test: at each layer, a well-calibrated model should
place true statements above gibberish, and gibberish above false:
    true > gibberish > false
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from latent_scope.core import HookedModel


TRUTH_DOMAINS_PATH = Path(__file__).parent.parent / "data" / "truth_domains.json"


DEFAULT_DATASET: dict[str, list[str]] = {
    "true": [
        "Paris is the capital of France.",
        "Berlin is the capital of Germany.",
        "Tokyo is the capital of Japan.",
        "Ottawa is the capital of Canada.",
        "Rome is the capital of Italy.",
        "Madrid is the capital of Spain.",
    ],
    "false": [
        "London is the capital of France.",
        "Rome is the capital of Germany.",
        "Seoul is the capital of Japan.",
        "Sydney is the capital of Canada.",
        "Paris is the capital of Italy.",
        "Lisbon is the capital of Spain.",
    ],
    "gibberish": [
        "Bicycle is the capital of France.",
        "Pancake is the capital of Germany.",
        "Lantern is the capital of Japan.",
        "Volcano is the capital of Canada.",
        "Sandbox is the capital of Italy.",
        "Cupcake is the capital of Spain.",
    ],
}


def load_truth_domains() -> list[dict]:
    """Load pre-built truth domains from JSON."""
    with open(TRUTH_DOMAINS_PATH, encoding="utf-8") as f:
        return json.load(f)


def get_truth_domain(key: str) -> dict[str, list[str]]:
    """Load a specific truth domain by key, returning {true, false, gibberish}."""
    domains = load_truth_domains()
    domain = next(d for d in domains if d["key"] == key)
    return {
        "true": domain["true"],
        "false": domain["false"],
        "gibberish": domain["gibberish"],
    }


@dataclass
class TruthCalibration:
    """Centroids per label per layer, built from calibration statements."""
    # centroids[label][layer] = np.ndarray (hidden_dim,)
    centroids: dict[str, dict[int, np.ndarray]] = field(default_factory=dict)
    num_layers: int = 0

    def project(self, vec: np.ndarray, layer: int) -> float:
        """Signed projection onto the true-vs-false axis at a given layer."""
        ct = self.centroids["true"].get(layer)
        cf = self.centroids["false"].get(layer)
        if ct is None or cf is None:
            return 0.0
        mid = 0.5 * (ct + cf)
        axis = ct - cf
        norm = np.linalg.norm(axis)
        if norm == 0:
            return 0.0
        axis = axis / norm
        return float(np.dot(vec - mid, axis))


@dataclass
class TrajectoryResult:
    """Per-layer projection of a statement onto the truth axis."""
    text: str
    label: str
    # points[layer_index] = signed projection float
    points: dict[int, float] = field(default_factory=dict)


class TruthAxis:
    """Truth axis calibration and projection tool.

    Usage::

        tool = TruthAxis(model)
        cal = tool.calibrate(DEFAULT_DATASET)
        result = tool.project_statement("The Earth orbits the Sun.", cal)
        # result.points[20] > 0 means the model thinks it's true at layer 20
    """

    def __init__(self, model: "HookedModel"):
        self.model = model

    def calibrate(
        self,
        dataset: dict[str, list[str]],
        callback=None,
    ) -> TruthCalibration:
        """Build truth axis from labeled calibration statements.

        dataset keys: "true", "false", "gibberish"
        Each value: list of statements.
        """
        num_layers = self.model.num_layers
        cal = TruthCalibration(num_layers=num_layers)

        # Collect last-token vectors per label per layer
        per_label: dict[str, dict[int, list[np.ndarray]]] = {
            label: {l: [] for l in range(num_layers)}
            for label in dataset
        }

        total = sum(len(stmts) for stmts in dataset.values())
        done = 0
        for label, statements in dataset.items():
            for stmt in statements:
                vecs = self._last_token_all_layers(stmt)
                for layer, vec in vecs.items():
                    per_label[label][layer].append(vec)
                done += 1
                if callback:
                    callback("calibrate", done, total, stmt)

        # Compute centroids
        cal.centroids = {}
        for label in dataset:
            cal.centroids[label] = {}
            for layer in range(num_layers):
                vecs = per_label[label][layer]
                if vecs:
                    cal.centroids[label][layer] = np.mean(vecs, axis=0)

        return cal

    def project_statement(
        self,
        text: str,
        calibration: TruthCalibration,
        label: str = "test",
    ) -> TrajectoryResult:
        """Project a single statement onto the calibrated truth axis."""
        vecs = self._last_token_all_layers(text)
        points = {
            layer: calibration.project(vec, layer)
            for layer, vec in vecs.items()
        }
        return TrajectoryResult(text=text, label=label, points=points)

    def _last_token_all_layers(self, text: str) -> dict[int, np.ndarray]:
        """Extract last-token vector at every layer (single forward pass)."""
        all_layers = list(range(self.model.num_layers))
        cache = self.model.run_with_cache(text, layers=all_layers)

        result = {}
        for layer in all_layers:
            path = f"model.layers.{layer}"
            if path not in cache:
                continue
            acts = cache[path]  # (1, seq_len, hidden_dim)
            vec = np.asarray(acts[0, -1, :], dtype=np.float64)
            vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
            result[layer] = vec
        return result
