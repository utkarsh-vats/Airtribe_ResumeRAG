"""``FakeEmbedder``: deterministic feature hashing for tests and CI (no model, no network).

Each lowercased word token is hashed with SHA-256 (never Python's salted ``hash()``) to a
bucket and a sign; the signed counts are L2-normalised. The same text gives the same vector
in every process and on every platform, and texts that share words score higher.
"""

import hashlib
import re
from collections.abc import Sequence
from typing import override

from rag.embedders.base import DEFAULT_BATCH_SIZE, Embedder

TOKEN = re.compile(r"\w+")


class FakeEmbedder(Embedder):
    """Signed feature hashing of word tokens into ``dim`` buckets."""

    def __init__(self, dim: int = 64, max_tokens: int = 512, batch_size: int = DEFAULT_BATCH_SIZE):
        """``dim`` buckets; ``max_tokens`` is reported, not enforced."""
        super().__init__(batch_size)
        if dim < 1:
            raise ValueError("dim must be positive")
        self._dim = dim
        self._max_tokens = max_tokens

    @property
    @override
    def name(self) -> str:
        return "fake"

    @property
    @override
    def model(self) -> str:
        return "feature-hashing"

    @property
    @override
    def dim(self) -> int:
        return self._dim

    @property
    @override
    def max_tokens(self) -> int:
        return self._max_tokens

    @property
    @override
    def embedder_id(self) -> str:
        return f"fake-d{self._dim}"

    @staticmethod
    def tokens(text: str) -> list[str]:
        """Lowercased word tokens."""
        return TOKEN.findall(text.lower())

    @override
    def token_count(self, text: str) -> int:
        return len(self.tokens(text))

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self._dim
        # A text with no words hashes the empty token, so it still gets a unit vector.
        for token in self.tokens(text) or [""]:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:8], "big") % self._dim
            vector[bucket] += 1.0 if digest[8] & 1 else -1.0
        return vector

    @override
    def _embed_document_batch(self, texts: list[str]) -> Sequence[Sequence[float]]:
        return [self._vector(text) for text in texts]

    @override
    def _embed_query(self, text: str) -> Sequence[float]:
        return self._vector(text)
