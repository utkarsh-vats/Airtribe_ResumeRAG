"""Embedding backends behind one ``Embedder`` contract, plus a lazy registry.

``get_embedder(name, **params)`` imports a backend only when it is requested, so importing
``rag.embedders`` never pulls in torch, sentence-transformers or google-genai.
``from_env()`` builds the embedder configured by ``EMBEDDER`` and the backend's settings
(``HF_EMBEDDING_MODEL``; ``GEMINI_API_KEY``, ``GEMINI_EMBEDDING_MODEL``,
``GEMINI_EMBEDDING_DIM``).
"""

import importlib
import os
from collections.abc import Mapping

from rag.embedders.base import Embedder

DEFAULT_EMBEDDER = "huggingface"
DEFAULT_HF_MODEL = "BAAI/bge-small-en-v1.5"

# name -> "module:class", imported on demand.
_REGISTRY: dict[str, str] = {
    "fake": "rag.embedders.fake:FakeEmbedder",
    "gemini": "rag.embedders.gemini:GeminiEmbedder",
    "huggingface": "rag.embedders.huggingface:HuggingFaceEmbedder",
}


def available_embedders() -> list[str]:
    """Registered backend names."""
    return sorted(_REGISTRY)


def get_embedder(name: str, **params) -> Embedder:
    """An embedder by registry name, e.g. ``get_embedder("fake", dim=128)``."""
    try:
        target = _REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown embedder {name!r}; available: {available_embedders()}") from None
    module, _, cls = target.partition(":")
    return getattr(importlib.import_module(module), cls)(**params)


def _setting(env: Mapping[str, str], key: str) -> str:
    return (env.get(key) or "").strip()


def from_env(environ: Mapping[str, str] | None = None) -> Embedder:
    """The embedder configured by ``EMBEDDER`` (default huggingface) and its settings."""
    env = os.environ if environ is None else environ
    name = _setting(env, "EMBEDDER") or DEFAULT_EMBEDDER
    if name == "huggingface":
        return get_embedder(name, model=_setting(env, "HF_EMBEDDING_MODEL") or DEFAULT_HF_MODEL)
    if name == "gemini":
        model = _setting(env, "GEMINI_EMBEDDING_MODEL")
        if not model:
            raise ValueError(
                "GEMINI_EMBEDDING_MODEL is required when EMBEDDER=gemini "
                "(e.g. gemini-embedding-001 or gemini-embedding-2)"
            )
        dim = _setting(env, "GEMINI_EMBEDDING_DIM")
        if dim and not dim.isdigit():
            raise ValueError(f"GEMINI_EMBEDDING_DIM must be a whole number, got {dim!r}")
        return get_embedder(
            name,
            model=model,
            api_key=_setting(env, "GEMINI_API_KEY"),
            output_dimensionality=int(dim) if dim else None,
        )
    return get_embedder(name)


__all__ = ["Embedder", "available_embedders", "from_env", "get_embedder"]
