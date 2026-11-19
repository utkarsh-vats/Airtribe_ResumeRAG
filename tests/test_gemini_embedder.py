"""GeminiEmbedder against a fake google-genai client (no network, no sleeping)."""

import logging
import math
from types import SimpleNamespace

import pytest

from rag.embedders import from_env, get_embedder
from rag.embedders.gemini import (
    DEFAULT_001_BATCH,
    FAMILIES,
    EmbeddingError,
    GeminiConfigError,
    GeminiEmbedder,
    gemini_embedder_id,
)


class FakeAPIError(Exception):
    """Shaped like google.genai.errors.APIError: the HTTP status is ``code``."""

    def __init__(self, code: int):
        super().__init__(f"{code} error")
        self.code = code


class FakeModels:
    def __init__(self, failures: list[int] | None = None, dim: int = 4):
        self.failures = list(failures or [])
        self.dim = dim
        self.calls: list[dict] = []

    def embed_content(self, *, model, contents, config=None):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if self.failures:
            raise FakeAPIError(self.failures.pop(0))
        texts = [contents] if isinstance(contents, str) else contents
        # Unnormalised vectors that depend on the text, like a reduced-dimension 001 output.
        return SimpleNamespace(
            embeddings=[
                SimpleNamespace(values=[float(len(t))] + [1.0] * (self.dim - 1)) for t in texts
            ]
        )


class Sleeper:
    def __init__(self):
        self.delays: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def make(model: str = "gemini-embedding-001", failures=None, **kwargs):
    models = FakeModels(failures)
    sleeper = Sleeper()
    embedder = GeminiEmbedder(
        model,
        api_key="test-key",
        client=SimpleNamespace(models=models),
        sleep=sleeper,
        rand=lambda: 0.5,
        **kwargs,
    )
    return embedder, models, sleeper


def norm(v) -> float:
    return math.sqrt(sum(x * x for x in v))


# --- Request shape per family -------------------------------------------------------------


def test_001_task_types_for_documents_and_queries():
    embedder, models, _ = make()
    embedder.embed_documents(["Python and Django", "Kubernetes"])
    embedder.embed_query("python developer")
    docs, query = models.calls
    assert docs["model"] == "gemini-embedding-001"
    assert docs["contents"] == ["Python and Django", "Kubernetes"]  # one request, list form
    assert docs["config"] == {"task_type": "RETRIEVAL_DOCUMENT"}
    assert query["contents"] == "python developer"  # no text prefix on 001
    assert query["config"] == {"task_type": "RETRIEVAL_QUERY"}


def test_001_batches_up_to_the_limit():
    embedder, models, _ = make()
    assert embedder.batch_size == DEFAULT_001_BATCH
    embedder.embed_documents([f"doc {i}" for i in range(DEFAULT_001_BATCH * 2 + 5)])
    assert [len(c["contents"]) for c in models.calls] == [DEFAULT_001_BATCH] * 2 + [5]
    small, small_models, _ = make(batch_size=3)
    small.embed_documents(["a", "b", "c", "d"])
    assert [len(c["contents"]) for c in small_models.calls] == [3, 1]
    capped, _, _ = make(batch_size=500)
    assert capped.batch_size == DEFAULT_001_BATCH


def test_embedding_2_one_input_per_request_with_task_prefixes():
    embedder, models, _ = make("gemini-embedding-2", batch_size=10)
    assert embedder.batch_size == 1  # several inputs would be aggregated into one vector
    vectors = embedder.embed_documents(["Python and Django", "Kubernetes"])
    embedder.embed_query("python developer")
    assert len(vectors) == 2
    assert [c["contents"] for c in models.calls] == [
        "title: none | text: Python and Django",
        "title: none | text: Kubernetes",
        "task: search result | query: python developer",
    ]
    assert all(c["config"] is None for c in models.calls)  # no task_type on embedding-2


def test_output_dimensionality_passed_through_and_in_id():
    embedder, models, _ = make(output_dimensionality=768)
    embedder.embed_query("x")
    assert models.calls[0]["config"] == {
        "task_type": "RETRIEVAL_QUERY",
        "output_dimensionality": 768,
    }
    assert (embedder.dim, embedder.embedder_id) == (768, "gemini-embedding-001-d768")
    two, two_models, _ = make("gemini-embedding-2", output_dimensionality=1536)
    two.embed_documents(["y"])
    assert two_models.calls[0]["config"] == {"output_dimensionality": 1536}
    assert two.embedder_id == "gemini-embedding-2-d1536"


def test_embedder_id_never_doubles_the_prefix():
    assert gemini_embedder_id("gemini-embedding-001", 768) == "gemini-embedding-001-d768"
    assert gemini_embedder_id("gemini-embedding-2", 3072) == "gemini-embedding-2-d3072"
    assert gemini_embedder_id("text-embedding-x", 256) == "gemini-text-embedding-x-d256"
    assert make("gemini-embedding-2")[0].embedder_id == "gemini-embedding-2-d3072"


def test_default_dimension_and_properties():
    embedder, _, _ = make()
    assert (embedder.name, embedder.model, embedder.dim) == ("gemini", "gemini-embedding-001", 3072)
    assert embedder.embedder_id == "gemini-embedding-001-d3072"
    assert embedder.max_tokens == 2048
    assert make("models/gemini-embedding-2")[0].max_tokens == 8192  # "models/" prefix accepted


def test_reduced_dimension_vectors_are_normalised():
    embedder, _, _ = make(output_dimensionality=128)
    for vector in [*embedder.embed_documents(["abc", "abcdef"]), embedder.embed_query("q")]:
        assert norm(vector) == pytest.approx(1.0)


def test_token_estimate_and_limit_warning(caplog):
    embedder, _, _ = make()
    assert embedder.token_count("abcd" * 10) == 10
    assert embedder.token_count("abcde") == 2  # rounds up
    with caplog.at_level(logging.WARNING, logger="rag.embedders.gemini"):
        embedder.embed_documents(["x" * (2048 * 4 + 40)])
    assert "about 2058 tokens exceeds the gemini-embedding-001 input limit of 2048" in caplog.text


# --- Retries and throttle -----------------------------------------------------------------


@pytest.mark.parametrize("status", [429, 503, 500])
def test_retry_then_success(status):
    embedder, models, sleeper = make(failures=[status, status], base_delay=1.0)
    vector = embedder.embed_query("q")
    assert len(models.calls) == 3 and len(vector) == 4
    assert sleeper.delays == [0.75, 1.5]  # ceiling 1, 2 with jitter fixed at 0.5


def test_backoff_sequence_is_exponential_capped_and_jittered():
    embedder, _, _ = make(base_delay=1.0, max_delay=10.0)
    assert [embedder.backoff(n) for n in range(6)] == [0.75, 1.5, 3.0, 6.0, 7.5, 7.5]
    low, _, _ = make()
    low._rand = lambda: 0.0
    high, _, _ = make()
    high._rand = lambda: 1.0
    assert (low.backoff(2), high.backoff(2)) == (2.0, 4.0)  # jitter spans [ceiling/2, ceiling]


def test_gives_up_after_max_retries():
    embedder, models, sleeper = make(failures=[429] * 10, max_retries=3)
    with pytest.raises(EmbeddingError) as info:
        embedder.embed_documents(["a"])
    error = info.value
    assert (error.model, error.status, error.attempts) == ("gemini-embedding-001", 429, 4)
    assert "gemini-embedding-001" in str(error) and "429" in str(error)
    assert len(models.calls) == 4 and len(sleeper.delays) == 3


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_no_retry_on_other_client_errors(status):
    embedder, models, sleeper = make(failures=[status])
    with pytest.raises(EmbeddingError) as info:
        embedder.embed_query("q")
    assert info.value.status == status and info.value.attempts == 1
    assert len(models.calls) == 1 and sleeper.delays == []


@pytest.mark.parametrize(
    ("kind", "code", "retried"),
    [("ClientError", 429, True), ("ServerError", 503, True), ("ClientError", 400, False)],
)
def test_real_sdk_errors_are_classified(kind, code, retried):
    from google.genai import errors

    error = getattr(errors, kind)(code, {"error": {"code": code, "message": "m", "status": "X"}})
    embedder, models, sleeper = make()

    def fail_once(**kwargs):
        models.embed_content = FakeModels().embed_content
        raise error

    models.embed_content = fail_once
    if retried:
        embedder.embed_query("q")
        assert len(sleeper.delays) == 1
    else:
        with pytest.raises(EmbeddingError) as info:
            embedder.embed_query("q")
        assert info.value.status == code and sleeper.delays == []


def test_errors_without_a_status_are_not_retried():
    embedder, models, _ = make()

    def boom(**_kwargs):
        raise ConnectionError("network down")

    models.embed_content = boom
    with pytest.raises(EmbeddingError) as info:
        embedder.embed_query("q")
    assert info.value.status is None


def test_throttle_spaces_requests():
    now = [100.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    embedder = GeminiEmbedder(
        "gemini-embedding-2",
        api_key="k",
        requests_per_minute=60,  # one request per second
        client=SimpleNamespace(models=FakeModels()),
        sleep=sleep,
        clock=lambda: now[0],
    )
    embedder.embed_documents(["a", "b", "c"])  # three requests
    assert sleeps == [1.0, 1.0]
    now[0] += 5.0
    embedder.embed_query("q")  # enough time passed: no wait
    assert sleeps == [1.0, 1.0]


def test_throttle_off_by_default():
    embedder, _, sleeper = make("gemini-embedding-2")
    embedder.embed_documents(["a", "b", "c"])
    assert embedder.min_interval == 0.0 and sleeper.delays == []


# --- Configuration ------------------------------------------------------------------------


@pytest.mark.parametrize("key", [None, "", "   "])
def test_missing_api_key_fails_at_construction(key):
    with pytest.raises(GeminiConfigError, match="GEMINI_API_KEY"):
        GeminiEmbedder("gemini-embedding-001", api_key=key)


@pytest.mark.parametrize(
    "model", ["text-embedding-004", "gemini-embedding-exp", "", "embedding-001"]
)
def test_unsupported_model_lists_families(model):
    with pytest.raises(GeminiConfigError) as info:
        GeminiEmbedder(model, api_key="k")
    for family in FAMILIES:
        assert family in str(info.value)


@pytest.mark.parametrize("dim", [64, 4096])
def test_dimension_range_checked(dim):
    with pytest.raises(GeminiConfigError, match="128-3072"):
        GeminiEmbedder("gemini-embedding-001", api_key="k", output_dimensionality=dim)


def test_registry_and_from_env():
    embedder = get_embedder("gemini", model="gemini-embedding-2", api_key="k")
    assert isinstance(embedder, GeminiEmbedder)
    env = {
        "EMBEDDER": "gemini",
        "GEMINI_API_KEY": "k",
        "GEMINI_EMBEDDING_MODEL": "gemini-embedding-001",
        "GEMINI_EMBEDDING_DIM": "768",
    }
    configured = from_env(env)
    assert configured.embedder_id == "gemini-embedding-001-d768"
    assert from_env({**env, "GEMINI_EMBEDDING_DIM": ""}).dim == 3072
    with pytest.raises(ValueError, match="GEMINI_EMBEDDING_MODEL is required"):
        from_env({**env, "GEMINI_EMBEDDING_MODEL": ""})
    with pytest.raises(GeminiConfigError, match="GEMINI_API_KEY"):
        from_env({**env, "GEMINI_API_KEY": ""})
    with pytest.raises(ValueError, match="whole number"):
        from_env({**env, "GEMINI_EMBEDDING_DIM": "lots"})


def test_client_created_lazily_with_the_key(monkeypatch):
    from rag.embedders import gemini

    created: list[str] = []

    def factory(api_key: str):
        created.append(api_key)
        return SimpleNamespace(models=FakeModels())

    monkeypatch.setattr(gemini, "_make_client", factory)
    embedder = GeminiEmbedder("gemini-embedding-001", api_key=" secret ")
    assert created == []
    embedder.embed_query("q")
    embedder.embed_query("r")
    assert created == ["secret"]
