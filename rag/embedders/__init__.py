"""Embedding backends behind one ``Embedder`` contract, plus a lazy registry.

``get_embedder(name, **params)`` imports a backend only when it is requested, so importing
``rag.embedders`` never pulls in torch, sentence-transformers or google-genai.
``from_env()`` builds the embedder configured by ``EMBEDDER`` and ``HF_EMBEDDING_MODEL``.
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


def from_env(environ: Mapping[str, str] | None = None) -> Embedder:
    """The embedder configured by ``EMBEDDER`` (default huggingface) and its settings."""
    env = os.environ if environ is None else environ
    name = (env.get("EMBEDDER") or DEFAULT_EMBEDDER).strip()
    if name == "huggingface":
        return get_embedder(name, model=(env.get("HF_EMBEDDING_MODEL") or DEFAULT_HF_MODEL).strip())
    return get_embedder(name)


__all__ = ["Embedder", "available_embedders", "from_env", "get_embedder"]
