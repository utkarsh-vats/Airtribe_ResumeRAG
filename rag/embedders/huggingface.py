"""``HuggingFaceEmbedder``: local sentence-transformers models.

``torch`` and ``sentence_transformers`` are imported only when the model is first used, so
importing ``rag`` or this module stays fast and works without them. Known models carry their
query/document prefixes, dimension and maximum sequence length in ``MODEL_CONFIGS``; other
models work too, without a prefix and with a warning. Vectors are requested normalised.

Inputs longer than the model's ``max_seq_length`` are truncated by the model without notice,
so every over-long text is logged with its token overflow.
"""

import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, override

from rag.embedders.base import DEFAULT_BATCH_SIZE, Embedder

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True)
class ModelConfig:
    """What we know about a model without loading it."""

    query_prefix: str = ""
    document_prefix: str = ""
    dim: int | None = None
    max_seq_length: int | None = None
    # Fallback semantic calibration (rag.calibration) for a collection that has none: the
    # floor and ceiling of candidate-level semantic aggregates. None = generic fallback.
    semantic_floor: float | None = None
    semantic_ceiling: float | None = None


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "BAAI/bge-small-en-v1.5": ModelConfig(BGE_QUERY_PREFIX, "", 384, 512, 0.82, 1.01),
    "BAAI/bge-base-en-v1.5": ModelConfig(BGE_QUERY_PREFIX, "", 768, 512),
    "sentence-transformers/all-MiniLM-L6-v2": ModelConfig("", "", 384, 256),
}


def _auto_device() -> str:
    """The best available device: cuda, then mps, then cpu (imports torch)."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _load_model(model: str, device: str) -> Any:
    """Load a SentenceTransformer (imports sentence_transformers and torch)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model, device=device)


class HuggingFaceEmbedder(Embedder):
    """A sentence-transformers model, loaded lazily on first use."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        loader: Callable[[str, str], Any] | None = None,
    ):
        """``device`` None auto-selects cuda / mps / cpu; ``loader`` is for tests."""
        super().__init__(batch_size)
        self._model_name = model
        self._requested_device = device
        self._loader = loader or _load_model
        self._model: Any = None
        self.config = MODEL_CONFIGS.get(model)
        if self.config is None:
            logger.warning(
                "Unknown embedding model %r: using no query/document prefix. Add it to "
                "MODEL_CONFIGS if it expects one.",
                model,
            )
            self.config = ModelConfig()

    # --- Model --------------------------------------------------------------------------

    @property
    def device(self) -> str:
        """The device the model runs on (auto-selected unless overridden)."""
        if self._requested_device is None:
            self._requested_device = _auto_device()
        return self._requested_device

    def _st(self) -> Any:
        if self._model is None:
            self._model = self._loader(self._model_name, self.device)
        return self._model

    # --- Embedder properties ------------------------------------------------------------

    @property
    @override
    def name(self) -> str:
        return "huggingface"

    @property
    @override
    def model(self) -> str:
        return self._model_name

    @property
    @override
    def dim(self) -> int:
        if self.config.dim is not None:
            return self.config.dim
        return int(self._st().get_sentence_embedding_dimension())

    @property
    @override
    def max_tokens(self) -> int:
        if self.config.max_seq_length is not None and self._model is None:
            return self.config.max_seq_length
        return int(self._st().max_seq_length)

    @property
    @override
    def embedder_id(self) -> str:
        short = self._model_name.rsplit("/", 1)[-1]
        return "hf-" + re.sub(r"[^A-Za-z0-9._-]+", "-", short)

    # --- Encoding -----------------------------------------------------------------------

    @override
    def token_count(self, text: str) -> int:
        """Tokens including the model's special tokens, as counted against max_seq_length."""
        return len(self._st().tokenizer.encode(text, add_special_tokens=True))

    def _warn_if_truncated(self, texts: list[str]) -> None:
        limit = int(self._st().max_seq_length)  # the loaded model's real limit
        for text in texts:
            count = self.token_count(text)
            if count > limit:
                logger.warning(
                    "Text of %d tokens exceeds max_seq_length %d by %d tokens and will be "
                    "truncated: %r",
                    count,
                    limit,
                    count - limit,
                    text[:80],
                )

    def _encode(self, texts: list[str]) -> Sequence[Sequence[float]]:
        self._warn_if_truncated(texts)
        vectors = self._st().encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist() if hasattr(vectors, "tolist") else vectors

    @override
    def _embed_document_batch(self, texts: list[str]) -> Sequence[Sequence[float]]:
        return self._encode([self.config.document_prefix + text for text in texts])

    @override
    def _embed_query(self, text: str) -> Sequence[float]:
        return self._encode([self.config.query_prefix + text])[0]
