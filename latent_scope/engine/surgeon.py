"""Activation Surgeon — intervene on model generation in real time.

Inject a contrastive steering vector into the residual stream during
generation. Watch how the model's outputs change as you dial the
intervention up from 0 to 5×.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np

from latent_scope.utils import normalize

if TYPE_CHECKING:
    from latent_scope.core import HookedModel


@dataclass
class SurgeryResult:
    """Comparison of baseline vs. intervened generation."""

    prompt: str
    baseline: str
    intervened: str
    operator_name: str
    layer: int
    scale: float


class ActivationSurgeon:
    """Steer model generation by injecting operators into the residual stream.

    Usage::

        surgeon = ActivationSurgeon(model)

        op = surgeon.contrastive_operator(
            "formality",
            positive="distinguished formal academic prose",
            negative="casual everyday slang",
        )

        result = surgeon.operate(
            prompt="The weather today is",
            operator=op, layer=16, scale=3.0,
        )
        print("Baseline:", result.baseline)
        print("Steered: ", result.intervened)
    """

    def __init__(self, model: "HookedModel"):
        self.model = model

    def contrastive_operator(
        self,
        name: str,
        positive: str,
        negative: str,
        layer: int = 20,
    ) -> "Operator":
        """Build a steering operator from a contrastive pair.

        vector = normalize(h_pos[-1] - h_neg[-1]) at the given layer.
        Uses the final token's hidden state — captures the overall "style"
        of each prompt.
        """
        from latent_scope.engine.operator_miner import Operator

        path = f"model.layers.{layer}"
        cache_pos = self.model.run_with_cache(positive, layers=[layer])
        cache_neg = self.model.run_with_cache(negative, layers=[layer])

        # Last token hidden state at this layer
        vec_pos = cache_pos[path][0, -1, :].astype(np.float32)
        vec_neg = cache_neg[path][0, -1, :].astype(np.float32)

        diff = normalize(vec_pos - vec_neg)
        return Operator(name=name, vector=diff, layer=layer)

    def operate(
        self,
        prompt: str,
        operator,
        layer: int | None = None,
        scale: float = 1.0,
        max_tokens: int = 100,
        temp: float = 0.0,
    ) -> SurgeryResult:
        """Generate text with and without operator intervention."""
        target_layer = layer if layer is not None else operator.layer
        if target_layer < 0:
            target_layer = self.model.num_layers // 2

        baseline = self.model.generate_text(prompt, max_tokens=max_tokens, temp=temp)

        if scale == 0.0:
            return SurgeryResult(
                prompt=prompt, baseline=baseline, intervened=baseline,
                operator_name=operator.name, layer=target_layer, scale=scale,
            )

        intervened = self._generate_steered(
            prompt, operator.vector, target_layer, scale, max_tokens, temp,
        )

        return SurgeryResult(
            prompt=prompt,
            baseline=baseline,
            intervened=intervened,
            operator_name=operator.name,
            layer=target_layer,
            scale=scale,
        )

    def _generate_steered(
        self,
        prompt: str,
        op_vector: np.ndarray,
        layer: int,
        scale: float,
        max_tokens: int,
        temp: float,
    ) -> str:
        """Token-by-token generation with a residual stream injection."""
        shift = mx.array((op_vector * scale).astype(np.float16))
        hook_path = f"model.layers.{layer}"

        def inject(inputs, output, wrapper):
            if isinstance(output, tuple):
                h, *rest = output
                return (h + shift, *rest)
            return output + shift

        tokens = self.model.tokenize(prompt)  # (1, seq_len)
        generated_ids = []

        for _ in range(max_tokens):
            logits = self.model.run_with_hooks(
                tokens, post_hooks={hook_path: inject},
            )  # (1, seq, vocab)

            last_logits = logits[0, -1, :]  # (vocab,)

            if temp <= 0:
                next_id = int(mx.argmax(last_logits).item())
            else:
                probs = mx.softmax(last_logits / temp)
                next_id = int(mx.random.categorical(mx.log(probs + 1e-10)).item())

            if next_id == self.model.tokenizer.eos_token_id:
                break

            generated_ids.append(next_id)
            next_token = mx.array([[next_id]])          # (1, 1)
            tokens = mx.concatenate([tokens, next_token], axis=1)
            mx.eval(tokens)

        return self.model.tokenizer.decode(generated_ids)

    def sweep_scales(
        self,
        prompt: str,
        operator,
        scales: list[float] | None = None,
        layer: int | None = None,
        max_tokens: int = 60,
    ) -> list[SurgeryResult]:
        """Run the operator at multiple scales to show the gradual effect."""
        if scales is None:
            scales = [0.0, 1.0, 2.0, 3.0, 5.0]
        return [
            self.operate(prompt, operator, layer=layer, scale=s, max_tokens=max_tokens)
            for s in scales
        ]
