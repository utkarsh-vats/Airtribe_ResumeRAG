"""The ``Embedder`` contract shared by every backend.

- Every returned vector is L2-normalised, so cosine similarity equals the dot product. The
  base class enforces this on whatever a backend returns.
- Documents and queries may be encoded differently (an instruction prefix for queries, a task
  type), so the two methods stay separate.
- ``embed_documents`` batches internally (``batch_size``).
- ``embedder_id`` is a stable string encoding everything that changes the vectors (backend,
  model, dimension); step 8 derives collection names from it, so vectors from different
  embedders never share a collection.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence

DEFAULT_BATCH_SIZE = 32


def l2_normalise(vector: Sequence[float]) -> list[float]:
    """``vector`` scaled to unit length (a zero vector is returned unchanged)."""
    norm = math.sqrt(sum(x * x for x in vector))
    return [float(x) / norm for x in vector] if norm else [float(x) for x in vector]


class Embedder(ABC):
    """Turns text into L2-normalised vectors."""

    def __init__(self, batch_size: int = DEFAULT_BATCH_SIZE):
        """``batch_size`` caps how many documents go to the backend in one call."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size

    @property
    @abstractmethod
    def name(self) -> str:
        """Registry name of the backend, e.g. ``huggingface``."""

    @property
    @abstractmethod
    def model(self) -> str:
        """Model identifier within the backend."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Vector dimension."""

    @property
    @abstractmethod
    def max_tokens(self) -> int:
        """Longest input, in the backend's tokens, before it truncates."""

    @property
    @abstractmethod
    def embedder_id(self) -> str:
        """Stable id covering everything that changes the vectors, e.g. ``fake-d64``."""

    @abstractmethod
    def token_count(self, text: str) -> int:
        """Tokens in ``text`` as the backend counts them."""

    @abstractmethod
    def _embed_document_batch(self, texts: list[str]) -> Sequence[Sequence[float]]:
        """Vectors for one batch of documents (normalised or not)."""

    @abstractmethod
    def _embed_query(self, text: str) -> Sequence[float]:
        """The vector for one query (normalised or not)."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Unit vectors for ``texts``, in order, sent to the backend in batches."""
        vectors: list[list[float]] = []
        texts = list(texts)
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors.extend(l2_normalise(v) for v in self._embed_document_batch(batch))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """The unit vector for a search query."""
        return l2_normalise(self._embed_query(text))

    def __repr__(self) -> str:
        """``ClassName('embedder_id')``."""
        return f"{type(self).__name__}({self.embedder_id!r})"
