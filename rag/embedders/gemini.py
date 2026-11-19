"""``GeminiEmbedder``: Google's Gemini embedding API through ``google-genai``.

Two model families are supported (Gemini API docs, "Embeddings" page, Nov 2026):

- ``gemini-embedding-001`` (text only, 2,048 input tokens). Tasks go in ``task_type``:
  ``RETRIEVAL_DOCUMENT`` for documents and ``RETRIEVAL_QUERY`` for queries ("Task types with
  Embeddings 1"). Several texts go in one request, as a list of strings.
- ``gemini-embedding-2`` (8,192 input tokens). ``task_type`` isn't accepted; the docs
  recommend putting the task in the text ("Task types with Embeddings 2", asymmetric
  retrieval format): ``task: search result | query: {content}`` for queries and
  ``title: none | text: {content}`` for documents. Several inputs in one request are
  *aggregated into one embedding* ("Embedding aggregation"), so each text is sent in its
  own request.

Both default to 3072 dimensions; ``output_dimensionality`` (128-3072) reduces it. Any other
model name is refused. ``google.genai`` is imported only when the first request is made.

Requests that fail with 429 or 5xx are retried with exponential backoff and jitter; other
4xx errors aren't retried. ``EmbeddingError`` names the model and the HTTP status. An optional
requests-per-minute throttle spaces requests out. Sleep, randomness and the clock are
injectable so tests never wait.
"""

import logging
import math
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, override

from rag.embedders.base import Embedder

logger = logging.getLogger(__name__)

DEFAULT_DIM = 3072
MIN_DIM = 128
# Not stated on the embeddings page; a conservative per-request cap for the list form.
DEFAULT_001_BATCH = 100
# Offline estimate: the Gemini token docs give roughly 4 characters per token.
CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class GeminiFamily:
    """What differs between the supported model families."""

    task_types: bool  # accepts task_type in the request config
    max_batch: int  # texts per request (1 where inputs would be aggregated)
    max_input_tokens: int
    query_prefix: str = ""
    document_prefix: str = ""


FAMILIES: dict[str, GeminiFamily] = {
    "gemini-embedding-001": GeminiFamily(
        task_types=True, max_batch=DEFAULT_001_BATCH, max_input_tokens=2048
    ),
    "gemini-embedding-2": GeminiFamily(
        task_types=False,
        max_batch=1,
        max_input_tokens=8192,
        query_prefix="task: search result | query: ",
        document_prefix="title: none | text: ",
    ),
}


class GeminiConfigError(ValueError):
    """The Gemini embedder is misconfigured (no API key, unknown model, bad dimension)."""


class EmbeddingError(RuntimeError):
    """An embedding request failed for good (after any retries)."""

    def __init__(self, model: str, status: int | None, attempts: int, cause: BaseException):
        """Record which model failed, with what status, after how many attempts."""
        self.model = model
        self.status = status
        self.attempts = attempts
        super().__init__(
            f"Gemini embedding request to {model!r} failed with status {status} after "
            f"{attempts} attempt(s): {cause}"
        )


def _status(exc: BaseException) -> int | None:
    """The HTTP status of an API error (google.genai's APIError carries it as ``code``)."""
    for attribute in ("code", "status_code"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int):
            return value
    return None


def _retryable(status: int | None) -> bool:
    return status == 429 or (status is not None and 500 <= status < 600)


def _make_client(api_key: str) -> Any:
    from google import genai

    return genai.Client(api_key=api_key)


class GeminiEmbedder(Embedder):
    """Gemini embeddings for ``gemini-embedding-001`` or ``gemini-embedding-2``."""

    def __init__(
        self,
        model: str,
        api_key: str | None,
        output_dimensionality: int | None = None,
        batch_size: int | None = None,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        requests_per_minute: float | None = None,
        client: Any = None,
        sleep: Callable[[float], None] = time.sleep,
        rand: Callable[[], float] = random.random,
        clock: Callable[[], float] = time.monotonic,
    ):
        """Validate the configuration; the client is created on the first request."""
        model = model.strip().removeprefix("models/") if model else ""
        if model not in FAMILIES:
            raise GeminiConfigError(
                f"Unsupported Gemini embedding model {model!r}; supported: {sorted(FAMILIES)}"
            )
        if not api_key or not api_key.strip():
            raise GeminiConfigError(
                "GEMINI_API_KEY is not set; it is required for EMBEDDER=gemini "
                "(get one at https://aistudio.google.com/apikey)"
            )
        dim = DEFAULT_DIM if output_dimensionality is None else output_dimensionality
        if not MIN_DIM <= dim <= DEFAULT_DIM:
            raise GeminiConfigError(f"output_dimensionality must be {MIN_DIM}-{DEFAULT_DIM}")
        self.family = FAMILIES[model]
        super().__init__(min(batch_size or self.family.max_batch, self.family.max_batch))
        if max_retries < 0:
            raise GeminiConfigError("max_retries must be >= 0")
        self._model = model
        self._api_key = api_key.strip()
        self._requested_dim = output_dimensionality
        self._dim = dim
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.min_interval = 60.0 / requests_per_minute if requests_per_minute else 0.0
        self._client = client
        self._sleep = sleep
        self._rand = rand
        self._clock = clock
        self._last_request: float | None = None

    # --- Embedder properties ------------------------------------------------------------

    @property
    @override
    def name(self) -> str:
        return "gemini"

    @property
    @override
    def model(self) -> str:
        return self._model

    @property
    @override
    def dim(self) -> int:
        return self._dim

    @property
    @override
    def max_tokens(self) -> int:
        return self.family.max_input_tokens

    @property
    @override
    def embedder_id(self) -> str:
        return f"gemini-{self._model}-d{self._dim}"

    @override
    def token_count(self, text: str) -> int:
        """Offline estimate, about 4 characters per token (no API call)."""
        return math.ceil(len(text) / CHARS_PER_TOKEN)

    # --- Requests -----------------------------------------------------------------------

    def _api(self) -> Any:
        if self._client is None:
            self._client = _make_client(self._api_key)
        return self._client

    def _throttle(self) -> None:
        if self.min_interval and self._last_request is not None:
            wait = self._last_request + self.min_interval - self._clock()
            if wait > 0:
                self._sleep(wait)
        self._last_request = self._clock()

    def backoff(self, retry: int) -> float:
        """Delay before retry number ``retry`` (0-based): exponential, capped, with jitter."""
        ceiling = min(self.max_delay, self.base_delay * 2**retry)
        return ceiling / 2 + self._rand() * ceiling / 2

    def _request(self, contents: str | list[str], task_type: str) -> list[list[float]]:
        config: dict[str, Any] = {}
        if self.family.task_types:
            config["task_type"] = task_type
        if self._requested_dim is not None:
            config["output_dimensionality"] = self._requested_dim
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                response = self._api().models.embed_content(
                    model=self._model, contents=contents, config=config or None
                )
                return [list(e.values) for e in response.embeddings]
            except Exception as exc:  # classified by HTTP status below
                status = _status(exc)
                if not _retryable(status) or attempt == self.max_retries:
                    raise EmbeddingError(self._model, status, attempt + 1, exc) from exc
                delay = self.backoff(attempt)
                logger.warning(
                    "Gemini %s returned %s; retry %d/%d in %.1fs",
                    self._model,
                    status,
                    attempt + 1,
                    self.max_retries,
                    delay,
                )
                self._sleep(delay)
        raise AssertionError("unreachable")

    def _warn_if_too_long(self, texts: Sequence[str]) -> None:
        for text in texts:
            estimate = self.token_count(text)
            if estimate > self.family.max_input_tokens:
                logger.warning(
                    "Text of about %d tokens exceeds the %s input limit of %d by about %d "
                    "tokens and may be truncated: %r",
                    estimate,
                    self._model,
                    self.family.max_input_tokens,
                    estimate - self.family.max_input_tokens,
                    text[:80],
                )

    # Vectors come back raw; the base class L2-normalises them. That matters for reduced
    # dimensions: gemini-embedding-001 returns unnormalised vectors below 3072 dimensions.
    @override
    def _embed_document_batch(self, texts: list[str]) -> Sequence[Sequence[float]]:
        inputs = [self.family.document_prefix + text for text in texts]
        self._warn_if_too_long(inputs)
        if self.family.max_batch == 1:
            return [self._request(text, "RETRIEVAL_DOCUMENT")[0] for text in inputs]
        vectors = self._request(inputs, "RETRIEVAL_DOCUMENT")
        if len(vectors) != len(inputs):
            raise EmbeddingError(
                self._model, None, 1, ValueError(f"{len(vectors)} vectors for {len(inputs)} texts")
            )
        return vectors

    @override
    def _embed_query(self, text: str) -> Sequence[float]:
        query = self.family.query_prefix + text
        self._warn_if_too_long([query])
        return self._request(query, "RETRIEVAL_QUERY")[0]
