"""Shared runtime helpers for the command center and explorers."""

from __future__ import annotations

import gc
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional

if TYPE_CHECKING:
    from latent_scope.core import HookedModel


DEFAULT_MODEL_CANDIDATES = [
    "mlx-community/Qwen2.5-3B-Instruct-4bit",
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mlx-community/Phi-3.5-mini-instruct-4bit",
]

# Orgs / name fragments that indicate non-LLM models (audio, vision encoders, etc.)
_NON_LLM_PATTERNS = [
    "musicgen", "encodec", "speaker-diarization", "segmentation",
    "wespeaker", "pyannote", "sentence-transformers", "t5-base",
    "whisper-", "clip-", "stable-diffusion", "vae", "unet",
]


def _looks_like_llm(repo_id: str) -> bool:
    low = repo_id.lower()
    return not any(p in low for p in _NON_LLM_PATTERNS)


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        value = item.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _looks_like_model_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    child_names = {child.name for child in path.iterdir()}
    if "config.json" in child_names:
        return True
    return any(name.endswith(".safetensors") for name in child_names)


def discover_mlx_models(extra_models: Optional[list[str]] = None) -> list[str]:
    """Discover likely-usable MLX model identifiers and local paths."""
    candidates: list[str] = []

    env_default = os.environ.get("WM_DEFAULT_MODEL", "")
    if env_default:
        candidates.append(env_default)

    env_models = os.environ.get("WM_MODELS", "")
    if env_models:
        candidates.extend(env_models.split(","))

    candidates.extend(DEFAULT_MODEL_CANDIDATES)

    model_roots = [
        Path.cwd() / "models",
        Path.home() / "models",
    ]
    for root in model_roots:
        if not root.exists():
            continue
        for child in sorted(root.iterdir()):
            if _looks_like_model_dir(child):
                candidates.append(str(child))

    # Scan all downloaded HuggingFace hub models
    hf_home = os.environ.get("HF_HOME", "")
    hf_cache = Path(hf_home) / "hub" if hf_home else Path.home() / ".cache" / "huggingface" / "hub"
    if hf_cache.exists():
        for repo_dir in sorted(hf_cache.glob("models--*")):
            # Convert "models--org--name" → "org/name"
            repo_id = repo_dir.name[len("models--"):].replace("--", "/", 1)
            if _looks_like_llm(repo_id):
                candidates.append(repo_id)

    if extra_models:
        candidates.extend(extra_models)

    return _unique(candidates)


@dataclass(frozen=True)
class ModelSession:
    model: Optional["HookedModel"]
    model_id: Optional[str]
    version: int


class SharedModelRuntime:
    """Thread-safe shared model loader for the command center."""

    def __init__(self, discovered_models: Optional[list[str]] = None):
        self._condition = threading.Condition()
        self._model: Optional["HookedModel"] = None
        self._model_id: Optional[str] = None
        self._version = 0
        self._loading = False
        self._active_sessions = 0
        self._last_error: Optional[str] = None
        self._available_models = discover_mlx_models(discovered_models or [])

    def list_models(self) -> list[str]:
        with self._condition:
            return list(self._available_models)

    def refresh_models(self) -> list[str]:
        with self._condition:
            remembered = list(self._available_models)
        discovered = discover_mlx_models(remembered)
        with self._condition:
            self._available_models = discovered
            return list(self._available_models)

    def snapshot(self) -> dict:
        with self._condition:
            return {
                "current_model": self._model_id,
                "has_model": self._model is not None,
                "loading": self._loading,
                "version": self._version,
                "active_sessions": self._active_sessions,
                "last_error": self._last_error,
                "available_models": list(self._available_models),
            }

    def get_model_with_version(self) -> tuple[Optional["HookedModel"], int, Optional[str]]:
        with self._condition:
            return self._model, self._version, self._model_id

    @contextmanager
    def model_session(self) -> Iterator[ModelSession]:
        with self._condition:
            while self._loading:
                self._condition.wait()
            self._active_sessions += 1
            session = ModelSession(
                model=self._model,
                model_id=self._model_id,
                version=self._version,
            )
        try:
            yield session
        finally:
            with self._condition:
                self._active_sessions -= 1
                if self._active_sessions == 0:
                    self._condition.notify_all()

    # ── Convenience helpers for app.py ──────────────────────────────────────

    @property
    def model(self):
        """Direct access to the loaded model (None if not loaded)."""
        with self._condition:
            return self._model

    def current_model_name(self) -> Optional[str]:
        """Return the currently loaded model id, or None."""
        with self._condition:
            return self._model_id

    def load(self, model_id: str) -> "HookedModel":
        """Alias for load_model()."""
        return self.load_model(model_id)

    def unload(self):
        """Alias for unload_model()."""
        self.unload_model()

    def load_model(self, model_id: str) -> "HookedModel":
        model_id = model_id.strip()
        if not model_id:
            raise ValueError("Model id is required.")

        with self._condition:
            while self._loading:
                self._condition.wait()
            self._loading = True
            self._last_error = None
            while self._active_sessions > 0:
                self._condition.wait()

        try:
            from latent_scope.core import HookedModel

            loaded_model = HookedModel.from_pretrained(model_id)
        except Exception as exc:
            with self._condition:
                self._loading = False
                self._last_error = str(exc)
                self._condition.notify_all()
            raise

        with self._condition:
            previous_model = self._model
            self._model = loaded_model
            self._model_id = model_id
            self._version += 1
            self._loading = False
            self._available_models = _unique([model_id, *self._available_models])
            self._condition.notify_all()

        if previous_model is not None:
            del previous_model
            gc.collect()

        return loaded_model

    def unload_model(self):
        with self._condition:
            while self._loading:
                self._condition.wait()
            self._loading = True
            self._last_error = None
            while self._active_sessions > 0:
                self._condition.wait()
            previous_model = self._model
            self._model = None
            self._model_id = None
            self._version += 1
            self._loading = False
            self._condition.notify_all()

        if previous_model is not None:
            del previous_model
            gc.collect()
