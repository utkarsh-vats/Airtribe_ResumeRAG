"""Embedders: the base contract, FakeEmbedder, HuggingFaceEmbedder (mocked) and the registry.

Offline: no model is downloaded and sentence-transformers is never imported here.
"""

import json
import logging
import math
import subprocess
import sys
from collections.abc import Sequence

import numpy as np
import pytest

from rag.embedders import available_embedders, from_env, get_embedder
from rag.embedders import huggingface as hf_module
from rag.embedders.base import Embedder, l2_normalise
from rag.embedders.fake import FakeEmbedder
from rag.embedders.huggingface import BGE_QUERY_PREFIX, MODEL_CONFIGS, HuggingFaceEmbedder
from scripts.persona_schema import ROOT


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def norm(v: Sequence[float]) -> float:
    return math.sqrt(dot(v, v))


# --- Base contract ------------------------------------------------------------------------


class CountingEmbedder(Embedder):
    """Returns unnormalised vectors and records batch sizes."""

    def __init__(self, batch_size: int = 3):
        super().__init__(batch_size)
        self.batches: list[int] = []

    name = model = embedder_id = "counting"  # type: ignore[assignment]
    dim = 2  # type: ignore[assignment]
    max_tokens = 10  # type: ignore[assignment]

    def token_count(self, text: str) -> int:
        return len(text)

    def _embed_document_batch(self, texts):
        self.batches.append(len(texts))
        return [[3.0, 4.0] for _ in texts]

    def _embed_query(self, text):
        return [0.0, 2.0]


def test_base_batches_and_normalises():
    embedder = CountingEmbedder(batch_size=3)
    vectors = embedder.embed_documents([f"t{i}" for i in range(7)])
    assert embedder.batches == [3, 3, 1]
    assert vectors == [[0.6, 0.8]] * 7
    assert embedder.embed_query("q") == [0.0, 1.0]
    assert embedder.embed_documents([]) == []
    with pytest.raises(ValueError):
        CountingEmbedder(batch_size=0)


def test_l2_normalise():
    assert l2_normalise([3, 4]) == [0.6, 0.8]
    assert l2_normalise([0, 0]) == [0.0, 0.0]


# --- FakeEmbedder -------------------------------------------------------------------------

TEXTS = [
    "Senior Python developer: Django, FastAPI and PostgreSQL",
    "Python backend services with Django and Postgres",
    "Kubernetes clusters provisioned with Terraform on AWS",
    "",
]


def test_fake_is_deterministic_across_processes():
    script = (
        "import json; from rag.embedders.fake import FakeEmbedder; "
        f"print(json.dumps(FakeEmbedder().embed_documents({TEXTS!r})))"
    )
    outputs = []
    for seed in ("0", "12345"):  # different hash() salts: SHA-256 must not care
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
            env={**_base_env(), "PYTHONHASHSEED": seed},
        )
        outputs.append(json.loads(result.stdout))
    assert outputs[0] == outputs[1] == FakeEmbedder().embed_documents(TEXTS)


def _base_env() -> dict[str, str]:
    import os

    return {k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"}


def test_fake_vectors_are_unit_length():
    embedder = FakeEmbedder()
    for vector in [*embedder.embed_documents(TEXTS), embedder.embed_query("python django")]:
        assert len(vector) == 64
        assert norm(vector) == pytest.approx(1.0)


def test_fake_shared_words_are_closer():
    embedder = FakeEmbedder(dim=256)
    python_a, python_b, devops, _ = embedder.embed_documents(TEXTS)
    assert dot(python_a, python_b) > dot(python_a, devops)
    query = embedder.embed_query("python django")
    assert dot(query, python_a) > dot(query, devops)


def test_fake_is_case_insensitive_and_identical_for_queries():
    embedder = FakeEmbedder()
    assert embedder.embed_query("Python DJANGO") == embedder.embed_documents(["python django"])[0]


def test_fake_dim_and_id():
    embedder = FakeEmbedder(dim=128, max_tokens=50)
    assert len(embedder.embed_query("x")) == 128
    assert (embedder.name, embedder.dim, embedder.max_tokens) == ("fake", 128, 50)
    assert embedder.embedder_id == "fake-d128"
    assert embedder.token_count("Hello, wide world!") == 3
    with pytest.raises(ValueError):
        FakeEmbedder(dim=0)


# --- HuggingFaceEmbedder (mocked SentenceTransformer) -------------------------------------


class MockTokenizer:
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [0] * (len(text.split()) + (2 if add_special_tokens else 0))


class MockSentenceTransformer:
    def __init__(self, max_seq_length: int = 512, dim: int = 4):
        self.max_seq_length = max_seq_length
        self.dim = dim
        self.tokenizer = MockTokenizer()
        self.calls: list[tuple[list[str], dict]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def encode(self, sentences, **kwargs):
        self.calls.append((list(sentences), kwargs))
        return np.array([[float(len(s)), 1.0, 0.0, 0.0] for s in sentences])


class Loader:
    def __init__(self, model: MockSentenceTransformer | None = None):
        self.model = model or MockSentenceTransformer()
        self.loaded: list[tuple[str, str]] = []

    def __call__(self, name: str, device: str) -> MockSentenceTransformer:
        self.loaded.append((name, device))
        return self.model


def make(model: str = "BAAI/bge-small-en-v1.5", **kwargs) -> tuple[HuggingFaceEmbedder, Loader]:
    loader = Loader(kwargs.pop("mock", None))
    return HuggingFaceEmbedder(
        model, loader=loader, device=kwargs.pop("device", "cpu"), **kwargs
    ), loader


def test_bge_prefix_applies_to_queries_only():
    embedder, loader = make()
    embedder.embed_documents(["Python and Django"])
    embedder.embed_query("python developer")
    (docs, doc_kwargs), (queries, _) = loader.model.calls
    assert docs == ["Python and Django"]
    assert queries == [BGE_QUERY_PREFIX + "python developer"]
    assert doc_kwargs["normalize_embeddings"] is True


def test_minilm_has_no_prefix():
    embedder, loader = make("sentence-transformers/all-MiniLM-L6-v2")
    embedder.embed_query("python developer")
    assert loader.model.calls[0][0] == ["python developer"]
    assert loader.model.calls[0][1]["normalize_embeddings"] is True


def test_hf_vectors_normalised_and_listed():
    embedder, _ = make()
    (vector,) = embedder.embed_documents(["abc"])
    assert isinstance(vector, list) and norm(vector) == pytest.approx(1.0)


def test_hf_batching():
    embedder, loader = make(batch_size=2)
    embedder.embed_documents([f"doc {i}" for i in range(5)])
    assert [len(texts) for texts, _ in loader.model.calls] == [2, 2, 1]
    assert all(kwargs["batch_size"] == 2 for _, kwargs in loader.model.calls)


def test_hf_lazy_load_and_config_without_loading():
    embedder, loader = make()
    assert (embedder.dim, embedder.max_tokens) == (384, 512)
    assert embedder.embedder_id == "hf-bge-small-en-v1.5"
    assert loader.loaded == []  # nothing loaded yet
    embedder.embed_query("x")
    embedder.embed_query("y")
    assert loader.loaded == [("BAAI/bge-small-en-v1.5", "cpu")]


def test_hf_device_override_and_auto(monkeypatch):
    embedder, loader = make(device="mps")
    embedder.embed_query("x")
    assert loader.loaded[0][1] == "mps"
    monkeypatch.setattr(hf_module, "_auto_device", lambda: "cuda")
    auto = HuggingFaceEmbedder(loader=(auto_loader := Loader()))
    auto.embed_query("x")
    assert auto_loader.loaded == [("BAAI/bge-small-en-v1.5", "cuda")]
    assert auto.device == "cuda"


def test_hf_token_count_uses_the_tokenizer():
    embedder, _ = make()
    assert embedder.token_count("one two three") == 5  # + [CLS] and [SEP]


def test_hf_truncation_warning(caplog):
    embedder, _ = make(mock=MockSentenceTransformer(max_seq_length=6))
    with caplog.at_level(logging.WARNING, logger="rag.embedders.huggingface"):
        embedder.embed_documents(["short text", "one two three four five six seven"])
    messages = [r.getMessage() for r in caplog.records]
    assert len(messages) == 1
    assert "9 tokens exceeds max_seq_length 6 by 3 tokens" in messages[0]
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="rag.embedders.huggingface"):
        embedder.embed_query("one two three")  # the prefix pushes the query over
    assert "exceeds max_seq_length 6" in caplog.records[0].getMessage()


def test_hf_unknown_model_warns_and_has_no_prefix(caplog):
    with caplog.at_level(logging.WARNING, logger="rag.embedders.huggingface"):
        embedder, loader = make("my-org/my-model@v2", mock=MockSentenceTransformer(dim=7))
    assert "Unknown embedding model 'my-org/my-model@v2'" in caplog.text
    assert embedder.embedder_id == "hf-my-model-v2"
    assert embedder.dim == 7  # not in the config table: asks the loaded model
    embedder.embed_query("q")
    assert loader.model.calls[0][0] == ["q"]


def test_model_config_table():
    assert set(MODEL_CONFIGS) == {
        "BAAI/bge-small-en-v1.5",
        "BAAI/bge-base-en-v1.5",
        "sentence-transformers/all-MiniLM-L6-v2",
    }
    assert MODEL_CONFIGS["BAAI/bge-small-en-v1.5"].query_prefix == BGE_QUERY_PREFIX
    assert all(c.document_prefix == "" for c in MODEL_CONFIGS.values())


# --- Lazy imports, registry and env -------------------------------------------------------


def test_importing_rag_does_not_import_torch():
    script = (
        "import sys, rag, rag.embedders, rag.embedders.huggingface, rag.embedders.fake\n"
        "from rag.embedders import get_embedder, from_env\n"
        "get_embedder('huggingface', device='cpu'); from_env({}); get_embedder('fake')\n"
        "print(sorted(m for m in ('torch', 'sentence_transformers', 'google.genai') "
        "if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"


def test_registry_round_trip():
    assert available_embedders() == ["fake", "huggingface"]
    fake = get_embedder("fake", dim=32)
    assert isinstance(fake, FakeEmbedder) and fake.embedder_id == "fake-d32"
    hf = get_embedder("huggingface", model="sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    assert isinstance(hf, HuggingFaceEmbedder) and hf.embedder_id == "hf-all-MiniLM-L6-v2"
    with pytest.raises(ValueError, match="available"):
        get_embedder("gemini")


def test_from_env():
    default = from_env({})
    assert isinstance(default, HuggingFaceEmbedder) and default.model == "BAAI/bge-small-en-v1.5"
    blank = from_env({"EMBEDDER": "", "HF_EMBEDDING_MODEL": ""})
    assert blank.model == "BAAI/bge-small-en-v1.5"
    minilm = from_env(
        {"EMBEDDER": "huggingface", "HF_EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2"}
    )
    assert minilm.embedder_id == "hf-all-MiniLM-L6-v2"
    assert isinstance(from_env({"EMBEDDER": "fake"}), FakeEmbedder)
    with pytest.raises(ValueError):
        from_env({"EMBEDDER": "word2vec"})


def test_env_example_defaults_match():
    env = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "EMBEDDER=huggingface" in env
    assert "HF_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5" in env
