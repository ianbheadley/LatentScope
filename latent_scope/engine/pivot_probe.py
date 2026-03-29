"""Pivot Axis Probe — sort anything along a semantic dimension.

Define two semantic poles (e.g. "small" → "large"), then project arbitrary
concepts onto the axis the model has learned.  The result is a rank ordering
that emerges purely from the geometry of the residual stream — no chat, no
prompting tricks, just linear algebra on hidden states.

Three validation gates ensure the axis is real:
    Gate A  Spearman ρ significant (p < 0.05)
    Gate B  Δρ vs null axis > 0.30
    Gate C  Intruder variance ratio > 3×
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from latent_scope.utils import spearman_rho, normalize

if TYPE_CHECKING:
    from latent_scope.core import HookedModel


DOMAINS_PATH = Path(__file__).parent.parent / "data" / "domains.json"


@dataclass
class ProbeResult:
    """Result of a pivot axis probe run."""

    domain: str
    scores: dict[str, float]           # concept → projection score
    rho: float = 0.0                   # Spearman ρ vs ground truth
    p_value: float = 1.0
    null_rho: float = 0.0
    delta_rho: float = 0.0
    intruder_ratio: float = 0.0
    gate_a: bool = False               # significance
    gate_b: bool = False               # beats null
    gate_c: bool = False               # domain specific
    verdict: str = "NONE"

    @property
    def ranked(self) -> list[str]:
        return list(self.scores.keys())


class PivotProbe:
    """Use an LLM as a semantic sorting machine.

    Usage::

        probe = PivotProbe(model)

        # Quick custom axis
        result = probe.run(
            pole_low="tiny", pole_high="enormous",
            concepts=["ant", "dog", "whale", "elephant", "mouse"],
        )
        print(result.ranked)  # ['ant', 'mouse', 'dog', 'elephant', 'whale']

        # Run a pre-built domain with full validation
        result = probe.run_domain("biological_size")
    """

    def __init__(self, model: "HookedModel", layer: int = 20):
        self.model = model
        self.layer = layer

    def run(
        self,
        pole_low: str | list[str],
        pole_high: str | list[str],
        concepts: list[str],
        ground_truth: Optional[list[str]] = None,
        null_pole_low: Optional[str] = None,
        null_pole_high: Optional[str] = None,
        intruders: Optional[list[str]] = None,
        domain_name: str = "custom",
        callback=None,
    ) -> ProbeResult:
        """Run pivot axis probe with optional validation gates."""
        axis = self.model.get_axis_vector(pole_low, pole_high, self.layer)
        scores = self.model.project_concepts(concepts, axis, self.layer, callback=callback)
        result = ProbeResult(domain=domain_name, scores=scores)

        # Gate A: Spearman significance
        if ground_truth:
            ranked = list(scores.keys())
            result.rho, result.p_value = spearman_rho(ranked, ground_truth)
            result.gate_a = result.p_value < 0.05

        # Gate B: beats null axis
        if null_pole_low and null_pole_high and ground_truth:
            null_axis = self.model.get_axis_vector(
                null_pole_low, null_pole_high, self.layer
            )
            null_scores = self.model.project_concepts(concepts, null_axis, self.layer)
            null_ranked = list(null_scores.keys())
            result.null_rho, _ = spearman_rho(null_ranked, ground_truth)
            result.delta_rho = result.rho - abs(result.null_rho)
            result.gate_b = result.delta_rho > 0.30

        # Gate C: intruder specificity
        if intruders:
            in_domain = np.array(list(scores.values()))
            intruder_vecs = [
                self.model.get_concept_vector(c, self.layer) for c in intruders
            ]
            intruder_scores = np.array([float(v @ axis) for v in intruder_vecs])
            var_in = np.var(in_domain) if len(in_domain) > 1 else 1e-8
            var_out = np.var(intruder_scores) if len(intruder_scores) > 1 else 1e-8
            result.intruder_ratio = float(var_in / max(var_out, 1e-8))
            result.gate_c = result.intruder_ratio > 3.0

        # Verdict
        if result.gate_a and result.gate_b:
            result.verdict = "CONFIRMED" if result.rho > 0.70 else "PARTIAL"
        elif result.gate_a:
            result.verdict = "WEAK"

        return result

    def run_domain(self, domain_key: str, callback=None) -> ProbeResult:
        """Run a pre-built domain from domains.json."""
        domains = self.load_domains()
        domain = next(d for d in domains if d["key"] == domain_key)
        return self.run(
            pole_low=domain["pole_low"],
            pole_high=domain["pole_high"],
            concepts=domain["concepts"],
            ground_truth=domain.get("ground_truth_order"),
            null_pole_low=domain.get("null_pole_low"),
            null_pole_high=domain.get("null_pole_high"),
            intruders=domain.get("intruders"),
            domain_name=domain["name"],
            callback=callback,
        )

    def list_domains(self) -> list[dict]:
        """Return list of available pre-built domains."""
        return [
            {"key": d["key"], "name": d["name"], "category": d["category"],
             "description": d["axis_description"]}
            for d in self.load_domains()
        ]

    @staticmethod
    def load_domains() -> list[dict]:
        with open(DOMAINS_PATH, encoding="utf-8") as f:
            return json.load(f)
