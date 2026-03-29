"""Core model wrapper providing hook-based access to transformer internals.

Follows the same philosophy as mlux: wrap modules transparently so that
running with hooks or caches produces numerically identical logits to
running without them.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load, generate


# ---------------------------------------------------------------------------
# HookWrapper – transparent module wrapper that captures activations
# ---------------------------------------------------------------------------

class HookWrapper:
    """Wraps an nn.Module to optionally capture inputs/outputs and run hooks."""

    def __init__(self, wrapped: nn.Module, name: str):
        self.wrapped = wrapped
        self.hook_name = name
        self.last_input: Optional[tuple] = None
        self.last_output = None
        self._capture = False
        self._pre_hooks: list[Callable] = []
        self._post_hooks: list[Callable] = []

    def __call__(self, *args, **kwargs):
        # Pre-hooks may modify inputs
        for hook in self._pre_hooks:
            result = hook(args, kwargs, self)
            if result is not None:
                args, kwargs = result

        if self._capture:
            self.last_input = args

        output = self.wrapped(*args, **kwargs)

        if self._capture:
            self.last_output = output

        # Post-hooks may modify output
        for hook in self._post_hooks:
            result = hook(args, output, self)
            if result is not None:
                output = result

        return output

    def __getattr__(self, name: str):
        # Delegate unknown attributes to the wrapped module for transparency.
        return getattr(self.wrapped, name)


# ---------------------------------------------------------------------------
# Module tree utilities
# ---------------------------------------------------------------------------

def _iter_modules(module: nn.Module, prefix: str = ""):
    """Yield (dotted_path, child_module) for every submodule."""
    for name, child in module.children().items():
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Module):
            yield path, child
            yield from _iter_modules(child, path)
        elif isinstance(child, list):
            for i, item in enumerate(child):
                if isinstance(item, nn.Module):
                    item_path = f"{path}.{i}"
                    yield item_path, item
                    yield from _iter_modules(item, item_path)


def _get_nested(module: nn.Module, path: str):
    parts = path.split(".")
    obj = module
    for p in parts:
        if p.isdigit():
            obj = obj[int(p)]
        elif isinstance(obj, dict):
            obj = obj[p]
        else:
            obj = getattr(obj, p)
    return obj


def _set_nested(module: nn.Module, path: str, value):
    parts = path.split(".")
    obj = module
    for p in parts[:-1]:
        if p.isdigit():
            obj = obj[int(p)]
        elif isinstance(obj, dict):
            obj = obj[p]
        else:
            obj = getattr(obj, p)
    last = parts[-1]
    if last.isdigit():
        obj[int(last)] = value
    elif isinstance(obj, dict):
        obj[last] = value
    else:
        setattr(obj, last, value)


# ---------------------------------------------------------------------------
# HookedModel – main interface
# ---------------------------------------------------------------------------

class HookedModel:
    """Wraps an MLX language model for interpretability research.

    Usage::

        model = HookedModel.from_pretrained("mlx-community/Llama-3.2-3B-Instruct-4bit")

        # Capture activations
        cache = model.run_with_cache("The capital of France is", layers=[0, 12, 24])

        # Run with intervention hooks
        def amplify(inputs, output, wrapper):
            return output * 1.5

        model.run_with_hooks("Hello", post_hooks={"model.layers.12": amplify})
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self._lock = threading.Lock()
        self._wrappers: dict[str, HookWrapper] = {}

        # Discover architecture
        self.num_layers = len(model.model.layers)
        cfg = getattr(model.model, "args", None) or getattr(model, "config", None)
        self.hidden_dim = getattr(cfg, "hidden_size", None)
        self.num_heads = getattr(cfg, "num_attention_heads", None)
        self.num_kv_heads = getattr(cfg, "num_key_value_heads", self.num_heads)

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "HookedModel":
        model, tokenizer = load(model_path, **kwargs)
        return cls(model, tokenizer)

    # -- Tokenization helpers -----------------------------------------------

    def tokenize(self, text: str, add_bos: bool = True) -> mx.array:
        tokens = self.tokenizer.encode(text)
        if add_bos and hasattr(self.tokenizer, "bos_token_id"):
            bos = self.tokenizer.bos_token_id
            if bos is not None and (len(tokens) == 0 or tokens[0] != bos):
                tokens = [bos] + tokens
        return mx.array([tokens])

    def decode(self, token_ids) -> str:
        if isinstance(token_ids, mx.array):
            mx.eval(token_ids)
            token_ids = token_ids.tolist()
        if isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        return self.tokenizer.decode(token_ids)

    # -- Wrapping / unwrapping ----------------------------------------------

    def _wrap(self, paths: list[str]):
        """Wrap specified module paths with HookWrappers."""
        for path in paths:
            if path in self._wrappers:
                continue
            module = _get_nested(self.model, path)
            wrapper = HookWrapper(module, name=path)
            _set_nested(self.model, path, wrapper)
            self._wrappers[path] = wrapper

    def _unwrap_all(self):
        """Restore all wrapped modules to originals."""
        for path, wrapper in self._wrappers.items():
            _set_nested(self.model, path, wrapper.wrapped)
        self._wrappers.clear()

    def layer_paths(self, layers: Optional[list[int]] = None) -> list[str]:
        """Return dotted paths to transformer layer modules."""
        if layers is None:
            layers = list(range(self.num_layers))
        return [f"model.layers.{i}" for i in layers]

    # -- Core run methods ---------------------------------------------------

    def run_with_cache(
        self,
        text_or_tokens,
        layers: Optional[list[int]] = None,
    ) -> dict[str, np.ndarray]:
        """Forward pass capturing layer outputs.

        Returns dict mapping layer path -> numpy array of activations.
        """
        if layers is None:
            layers = list(range(self.num_layers))
        paths = self.layer_paths(layers)

        with self._lock:
            cache = {}
            self._wrap(paths)
            try:
                for w in self._wrappers.values():
                    w._capture = True
                    w.last_output = None

                tokens = self._to_tokens(text_or_tokens)
                logits = self.model(tokens)
                mx.eval(logits)

                for path in paths:
                    out = self._wrappers[path].last_output
                    if out is not None:
                        if isinstance(out, tuple):
                            out = out[0]
                        mx.eval(out)
                        cache[path] = np.asarray(out)
            finally:
                for w in self._wrappers.values():
                    w._capture = False
                    w.last_input = None
                    w.last_output = None
                self._unwrap_all()

        return cache

    def run_with_hooks(
        self,
        text_or_tokens,
        pre_hooks: Optional[dict[str, Callable]] = None,
        post_hooks: Optional[dict[str, Callable]] = None,
    ) -> mx.array:
        """Forward pass with intervention hooks. Returns logits."""
        pre_hooks = pre_hooks or {}
        post_hooks = post_hooks or {}
        all_paths = list(set(pre_hooks) | set(post_hooks))

        with self._lock:
            self._wrap(all_paths)
            try:
                for path, fn in pre_hooks.items():
                    self._wrappers[path]._pre_hooks.append(fn)
                for path, fn in post_hooks.items():
                    self._wrappers[path]._post_hooks.append(fn)

                tokens = self._to_tokens(text_or_tokens)
                logits = self.model(tokens)
                mx.eval(logits)
            finally:
                for w in self._wrappers.values():
                    w._pre_hooks.clear()
                    w._post_hooks.clear()
                self._unwrap_all()

        return logits

    # -- Vector extraction --------------------------------------------------

    @staticmethod
    def _finite_array(values) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)

    def _find_word_in_sentence(self, word: str, sentence: str) -> slice:
        """Find token positions of `word` within `sentence`.

        Encodes the prefix (text before the word) to count its tokens,
        then encodes the word in-context to get its token count.
        Accurate to ±1 token for most BPE vocabularies.
        """
        idx = sentence.lower().find(word.lower())
        if idx == -1:
            raise ValueError(f"'{word}' not found in '{sentence}'")

        prefix = sentence[:idx]
        prefix_ids = self.tokenizer.encode(prefix)
        bos = getattr(self.tokenizer, "bos_token_id", None)
        if bos is not None and prefix_ids and prefix_ids[0] == bos:
            prefix_ids = prefix_ids[1:]

        word_ids = self.tokenizer.encode(" " + sentence[idx:idx + len(word)].strip())
        if bos is not None and word_ids and word_ids[0] == bos:
            word_ids = word_ids[1:]
        n_word = max(len(word_ids), 1)

        bos_offset = 1 if bos is not None else 0
        start = len(prefix_ids) + bos_offset
        full_len = self.tokenize(sentence).shape[1]
        return slice(start, min(start + n_word, full_len))

    def _concept_span(self, concept: str, template: str) -> slice:
        """Return the token positions of the concept in the model input.

        Robust to trailing-space tokenization ambiguity: encode the concept
        with a leading space to match in-context tokenization, then take the
        last N tokens of the full prompt.
        """
        prompt = template.format(concept=concept)
        model_tokens = self.tokenize(prompt)   # (1, seq_len) — what the model sees
        seq_len = model_tokens.shape[1]

        # Encode " <concept>" to get how many tokens the concept occupies
        # in context (leading space simulates the space before the word).
        concept_raw = self.tokenizer.encode(" " + concept.strip())
        bos = getattr(self.tokenizer, "bos_token_id", None)
        if bos is not None and concept_raw and concept_raw[0] == bos:
            concept_raw = concept_raw[1:]
        n_concept = max(len(concept_raw), 1)

        return slice(seq_len - n_concept, seq_len)

    def get_concept_vector(
        self,
        concept: str,
        layer: int = 20,
        template: str = "Concept: {concept}",
    ) -> np.ndarray:
        """Extract a normalized concept vector from a specific layer.

        Uses mean-pooling over concept token positions.
        """
        prompt = template.format(concept=concept)
        concept_span = self._concept_span(concept, template)

        path = f"model.layers.{layer}"
        cache = self.run_with_cache(prompt, layers=[layer])
        activations = cache[path]  # (1, seq_len, hidden_dim)

        vec = self._finite_array(activations[0, concept_span, :]).mean(axis=0)
        vec = self._finite_array(vec)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return self._finite_array(vec)

    def get_axis_vector(
        self,
        pole_low: str | list[str],
        pole_high: str | list[str],
        layer: int = 20,
        template: str = "Concept: {concept}",
    ) -> np.ndarray:
        """Build a normalized directional axis from two semantic poles."""
        if isinstance(pole_low, str):
            pole_low = [pole_low]
        if isinstance(pole_high, str):
            pole_high = [pole_high]

        low_vecs = [self.get_concept_vector(c, layer, template) for c in pole_low]
        high_vecs = [self.get_concept_vector(c, layer, template) for c in pole_high]

        low_centroid = self._finite_array(np.mean(low_vecs, axis=0))
        high_centroid = self._finite_array(np.mean(high_vecs, axis=0))
        axis = self._finite_array(high_centroid - low_centroid)
        norm = np.linalg.norm(axis)
        if norm > 0:
            axis = axis / norm
        return self._finite_array(axis)

    def get_all_layer_vectors(
        self,
        concept: str,
        template: str = "Concept: {concept}",
    ) -> dict[int, np.ndarray]:
        """Extract concept vectors from every layer in a single forward pass.

        Returns dict mapping layer_index -> normalized (hidden_dim,) vector.
        This is the efficient path for layer trajectory analysis.
        """
        prompt = template.format(concept=concept)
        concept_span = self._concept_span(concept, template)

        all_layers = list(range(self.num_layers))
        cache = self.run_with_cache(prompt, layers=all_layers)

        result = {}
        for layer in all_layers:
            path = f"model.layers.{layer}"
            if path not in cache:
                continue
            acts = self._finite_array(cache[path][0, concept_span, :]).mean(axis=0)
            acts = self._finite_array(acts)
            norm = np.linalg.norm(acts)
            result[layer] = (acts / norm) if norm > 0 else acts
        return result

    def project_concepts(
        self,
        concepts: list[str],
        axis: np.ndarray,
        layer: int = 20,
        template: str = "Concept: {concept}",
        callback=None,
    ) -> dict[str, float]:
        """Project concepts onto an axis. Returns {concept: score} sorted.

        Optional callback(concept, score) is called after each concept,
        enabling streaming progress in the web UI.
        """
        scores = {}
        safe_axis = self._finite_array(axis)
        for c in concepts:
            vec = self.get_concept_vector(c, layer, template)
            score = float(self._finite_array(vec) @ safe_axis)
            scores[c] = score if np.isfinite(score) else 0.0
            if callback:
                callback(c, scores[c])
        return dict(sorted(scores.items(), key=lambda x: x[1]))

    # -- Generation ---------------------------------------------------------

    def generate_text(
        self,
        prompt: str,
        max_tokens: int = 100,
        temp: float = 0.0,
    ) -> str:
        """Simple text generation wrapper."""
        with self._lock:
            try:
                from mlx_lm.sample_utils import make_sampler
                sampler = make_sampler(temp=temp)
                return generate(
                    self.model, self.tokenizer, prompt=prompt,
                    max_tokens=max_tokens, sampler=sampler, verbose=False,
                )
            except ImportError:
                return generate(
                    self.model, self.tokenizer, prompt=prompt,
                    max_tokens=max_tokens, temp=temp, verbose=False,
                )

    # -- Internal -----------------------------------------------------------

    def _to_tokens(self, text_or_tokens) -> mx.array:
        if isinstance(text_or_tokens, str):
            return self.tokenize(text_or_tokens)
        if isinstance(text_or_tokens, mx.array):
            return text_or_tokens
        return mx.array(text_or_tokens)
