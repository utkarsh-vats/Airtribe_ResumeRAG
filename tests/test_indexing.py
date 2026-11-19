"""rag.store and rag.indexer: Chroma collections, the sidecar and incremental indexing."""

import json
import os
import shutil
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

import fs_tools
from rag import indexer
from rag.chunking import FixedSizeChunker, SectionChunker
from rag.embedders.fake import FakeEmbedder
from rag.indexer import content_hash, default_workspace_root, index_directory
from rag.metadata import extract_metadata
from rag.sections import parse_resume
from rag.store import (
    MAX_NAME,
    DimensionMismatchError,
    Sidecar,
    SidecarError,
    VectorStore,
    chunk_metadata,
    collection_name,
)
from scripts.persona_schema import ROOT

AS_OF = date(2026, 11, 1)
SAMPLES = ("john_doe.pdf", "elena_rodriguez.docx", "noah_williams.txt", "zoe_turner.txt")
FIXED_NOW = datetime(2026, 11, 19, 12, 0, tzinfo=UTC)


class SpyEmbedder(FakeEmbedder):
    """FakeEmbedder that counts embed_documents calls and texts."""

    def __init__(self, *args, fail_on: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = 0
        self.texts = 0
        self.fail_on = fail_on

    def embed_documents(self, texts):
        self.calls += 1
        self.texts += len(texts)
        if self.fail_on and any(self.fail_on in t for t in texts):
            raise RuntimeError("embedding backend exploded")
        return super().embed_documents(texts)


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "workspace"
    (root / "resumes").mkdir(parents=True)
    for name in SAMPLES:
        shutil.copy(ROOT / "data" / "resumes" / name, root / "resumes" / name)
    monkeypatch.setenv("WORKSPACE_ROOT", str(root))
    return root


@pytest.fixture
def chroma_dir(tmp_path: Path) -> Path:
    return tmp_path / "chroma"


def make_store(chroma_dir: Path, chunker=None, embedder=None):
    chunker = chunker or SectionChunker()
    embedder = embedder or SpyEmbedder()
    return (
        chunker,
        embedder,
        VectorStore(chroma_dir, chunker.chunker_id, embedder.embedder_id, embedder.dim),
    )


def run(store, chunker, embedder, **kwargs):
    return index_directory(
        "resumes",
        chunker,
        embedder,
        store,
        kwargs.pop("as_of", AS_OF),
        now=lambda: FIXED_NOW,
        **kwargs,
    )


def expected_chunks(workspace: Path, name: str, chunker) -> list:
    text = fs_tools.read_file(f"resumes/{name}")["data"]["content"]
    return chunker.chunk(f"resumes/{name}", parse_resume(text))


# --- Collection names and metadata --------------------------------------------------------


def test_collection_name_rules():
    assert (
        collection_name("section-v1", "hf-bge-small-en-v1.5")
        == "resumes__section-v1__hf-bge-small-en-v1.5"
    )
    assert collection_name("fixed w120/o30", "x..y") == "resumes__fixed-w120-o30__x.y"
    long = collection_name("section-v1", "e" * 600)
    assert len(long) == MAX_NAME and long[-13] == "-" and long[-12:].isalnum()
    assert long != collection_name("section-v1", "e" * 601)  # the hash keeps them apart


def test_chunk_metadata_has_no_none_or_empty_lists(workspace):
    chunker = SectionChunker()
    text = fs_tools.read_file("resumes/zoe_turner.txt")["data"]["content"]
    parsed = parse_resume(text)
    meta = extract_metadata("resumes/zoe_turner.txt", parsed, AS_OF)
    summary = chunk_metadata(chunker.chunk("resumes/zoe_turner.txt", parsed)[0], meta)
    assert "role_index" not in summary and "role_title" not in summary  # None -> omitted
    empty = chunk_metadata(
        chunker.chunk("resumes/zoe_turner.txt", parsed)[0],
        type(meta)(**{**meta.__dict__, "skills": (), "highest_education": None}),
    )
    assert "skills" not in empty and "highest_degree_level" not in empty


# --- Indexing runs ------------------------------------------------------------------------


def test_first_run_adds_everything(workspace, chroma_dir):
    chunker, embedder, store = make_store(chroma_dir)
    report = run(store, chunker, embedder)
    assert sorted(report.added) == sorted(f"resumes/{n}" for n in SAMPLES)
    assert (report.updated, report.unchanged, report.removed, report.failed) == ([], [], [], [])
    expected = sum(len(expected_chunks(workspace, n, chunker)) for n in SAMPLES)
    assert report.chunks_written == store.count() == expected
    assert store.list_candidates() == sorted(Path(n).stem for n in SAMPLES)
    assert embedder.calls == len(SAMPLES)
    assert set(report.file_timings) == set(report.added)
    assert report.timings.embed >= 0 and report.timings.total > 0


def test_second_run_is_unchanged_with_no_embedding(workspace, chroma_dir):
    chunker, embedder, store = make_store(chroma_dir)
    run(store, chunker, embedder)
    embedder.calls = 0
    report = run(store, chunker, embedder)
    assert sorted(report.unchanged) == sorted(f"resumes/{n}" for n in SAMPLES)
    assert (report.added, report.updated, report.chunks_written, embedder.calls) == ([], [], 0, 0)
    assert report.metadata_refreshed == []


def test_editing_a_file_replaces_its_chunks(workspace, chroma_dir):
    chunker, embedder, store = make_store(chroma_dir)
    run(store, chunker, embedder)
    old_ids = store.chunk_ids("noah_williams")
    path = workspace / "resumes" / "noah_williams.txt"
    path.write_bytes(path.read_bytes().replace(b"Canoe tripping", b"Canoe tripping, Chess"))
    embedder.calls = 0
    report = run(store, chunker, embedder)
    assert report.updated == ["resumes/noah_williams.txt"] and embedder.calls == 1
    new_ids = store.chunk_ids("noah_williams")
    assert new_ids == sorted(
        c.chunk_id for c in expected_chunks(workspace, "noah_williams.txt", chunker)
    )
    assert set(old_ids) - set(new_ids)  # the hobbies chunk changed
    assert not (set(old_ids) - set(new_ids)) & set(store.get(include=[])["ids"])  # old ones gone


def test_deleted_file_is_pruned(workspace, chroma_dir):
    chunker, embedder, store = make_store(chroma_dir)
    run(store, chunker, embedder)
    (workspace / "resumes" / "zoe_turner.txt").unlink()
    report = run(store, chunker, embedder)
    assert report.removed == ["zoe_turner"]
    assert store.chunk_ids("zoe_turner") == []
    assert "zoe_turner" not in store.load_sidecar().candidates


def test_pipeline_version_bump_reindexes_everything(workspace, chroma_dir, monkeypatch):
    chunker, embedder, store = make_store(chroma_dir)
    run(store, chunker, embedder)
    monkeypatch.setattr(indexer, "PIPELINE_VERSION", "test-bump")
    embedder.calls = 0
    report = run(store, chunker, embedder)
    assert sorted(report.updated) == sorted(f"resumes/{n}" for n in SAMPLES)
    assert embedder.calls == len(SAMPLES)
    assert store.load_sidecar().pipeline_version == "test-bump"


def test_new_as_of_refreshes_metadata_without_embedding(workspace, chroma_dir):
    chunker, embedder, store = make_store(chroma_dir)
    run(store, chunker, embedder)
    embedder.calls = 0
    report = run(store, chunker, embedder, as_of=date(2027, 11, 1))
    assert embedder.calls == 0 and len(report.unchanged) == len(SAMPLES)
    assert "resumes/zoe_turner.txt" in report.metadata_refreshed  # her role is ongoing
    sidecar = store.load_sidecar()
    assert sidecar.as_of == "2027-11-01"
    months = sidecar.candidates["zoe_turner"].metadata["total_months"]
    assert months == 31  # May 2025 .. Nov 2027
    stored = store.get(where={"candidate_id": "zoe_turner"}, include=["metadatas"])["metadatas"]
    assert {m["total_months"] for m in stored} == {31}


def test_bad_files_fail_or_skip_without_aborting(workspace, chroma_dir):
    (workspace / "resumes" / "broken.pdf").write_bytes(b"%PDF-1.4 not really a pdf")
    (workspace / "resumes" / "empty.txt").write_bytes(b"")
    (workspace / "resumes" / "photo.png").write_bytes(b"\x89PNG")
    (workspace / "resumes" / "notes.docx").write_bytes(b"not a zip")
    chunker, embedder, store = make_store(chroma_dir)
    report = run(store, chunker, embedder)
    failed = {f.path: f.code for f in report.failed}
    assert failed == {
        "resumes/broken.pdf": "EXTRACTION_FAILED",
        "resumes/empty.txt": "EMPTY_TEXT",
        "resumes/notes.docx": "EXTRACTION_FAILED",
    }
    assert next(f for f in report.failed if f.code == "EMPTY_TEXT").message == "File is empty."
    assert [(s.path, s.code) for s in report.skipped] == [
        ("resumes/photo.png", "UNSUPPORTED_FORMAT")
    ]
    assert len(report.added) == len(SAMPLES)


def test_embedding_failure_is_recorded_and_others_continue(workspace, chroma_dir):
    chunker = SectionChunker()
    embedder = SpyEmbedder(fail_on="Coral Coast Web")
    _, _, store = make_store(chroma_dir, chunker, embedder)
    report = run(store, chunker, embedder)
    assert [(f.path, f.code) for f in report.failed] == [("resumes/zoe_turner.txt", "RuntimeError")]
    assert len(report.added) == len(SAMPLES) - 1
    assert store.chunk_ids("zoe_turner") == []


def test_no_sections_fails(workspace, chroma_dir):
    (workspace / "resumes" / "blank.txt").write_bytes(b"Just a name\n")
    chunker, embedder, store = make_store(chroma_dir)
    report = run(store, chunker, embedder)
    assert ("resumes/blank.txt", "NO_SECTIONS") in [(f.path, f.code) for f in report.failed]


def test_rebuild_drops_and_reindexes(workspace, chroma_dir):
    chunker, embedder, store = make_store(chroma_dir)
    run(store, chunker, embedder)
    embedder.calls = 0
    report = run(store, chunker, embedder, rebuild=True)
    assert len(report.added) == len(SAMPLES) and embedder.calls == len(SAMPLES)
    assert store.count() == report.chunks_written


def test_separate_collections_per_chunker_and_embedder(workspace, chroma_dir):
    configs = [
        (SectionChunker(), SpyEmbedder()),
        (FixedSizeChunker(), SpyEmbedder()),
        (SectionChunker(), SpyEmbedder(dim=32)),
    ]
    names = set()
    for chunker, embedder in configs:
        _, _, store = make_store(chroma_dir, chunker, embedder)
        run(store, chunker, embedder)
        names.add(store.name)
        assert store.count() > 0
    assert names == {
        "resumes__section-v1__fake-d64",
        "resumes__fixed-w120-o30__fake-d64",
        "resumes__section-v1__fake-d32",
    }
    assert len(list((chroma_dir / "sidecar").glob("*.json"))) == 3


def test_dimension_mismatch(chroma_dir):
    VectorStore(chroma_dir, "section-v1", "fake-d64", 64)
    with pytest.raises(DimensionMismatchError, match="64-dimensional"):
        VectorStore(chroma_dir, "section-v1", "fake-d64", 32)


def test_wrong_store_for_chunker_is_refused(workspace, chroma_dir):
    _, embedder, store = make_store(chroma_dir)
    with pytest.raises(ValueError, match="not fixed-w120-o30"):
        index_directory("resumes", FixedSizeChunker(), embedder, store, AS_OF)


def test_stored_metadata_and_documents(workspace, chroma_dir):
    chunker, embedder, store = make_store(chroma_dir)
    run(store, chunker, embedder)
    stored = store.get(include=["metadatas", "documents"])
    for metadata in stored["metadatas"]:
        assert all(v is not None and v != [] for v in metadata.values())
        assert {
            "candidate_id",
            "resume_path",
            "name",
            "section",
            "chunker_id",
            "total_months",
        } <= set(metadata)
    chunks = {c.chunk_id: c for n in SAMPLES for c in expected_chunks(workspace, n, chunker)}
    for chunk_id, document in zip(stored["ids"], stored["documents"], strict=True):
        assert document == chunks[chunk_id].text  # text, not embed_text
    john = store.get(where={"candidate_id": "john_doe"}, include=["metadatas"])["metadatas"]
    assert all(m["highest_degree_level"] == 3 and "Python" in m["skills"] for m in john)
    roles = [m for m in john if "role_index" in m]
    assert sorted({m["role_index"] for m in roles}) == [0, 1, 2]


def test_query_returns_nearest_chunk(workspace, chroma_dir):
    chunker, embedder, store = make_store(chroma_dir)
    run(store, chunker, embedder)
    target = expected_chunks(workspace, "elena_rodriguez.docx", chunker)[2]
    result = store.query(embedder.embed_query(target.embed_text), n=3)
    assert result["ids"][0][0] == target.chunk_id
    assert result["distances"][0][0] == pytest.approx(0.0, abs=1e-5)
    only_noah = store.query(
        embedder.embed_query("Postgres"), n=2, where={"candidate_id": "noah_williams"}
    )
    assert {m["candidate_id"] for m in only_noah["metadatas"][0]} == {"noah_williams"}


# --- Sidecar ------------------------------------------------------------------------------


def test_sidecar_round_trip_and_as_of(workspace, chroma_dir):
    chunker, embedder, store = make_store(chroma_dir)
    run(store, chunker, embedder)
    raw = json.loads(store.sidecar_path.read_text(encoding="utf-8"))
    assert raw["as_of"] == "2026-11-01" and raw["schema_version"] == 1
    assert raw["pipeline_version"] == indexer.PIPELINE_VERSION
    assert (raw["chunker_id"], raw["embedder_id"]) == ("section-v1", "fake-d64")
    sidecar = store.load_sidecar()
    assert Sidecar.from_dict(sidecar.to_dict()).to_dict() == raw
    record = sidecar.candidates["noah_williams"]
    text = fs_tools.read_file("resumes/noah_williams.txt")["data"]["content"]
    assert record.content_hash == content_hash(text)
    assert record.indexed_at == "2026-11-19T12:00:00+00:00"
    assert [p.name for p in store.sidecar_path.parent.iterdir()] == [
        store.sidecar_path.name
    ]  # no temp files


def test_sidecar_schema_version_checked(workspace, chroma_dir):
    chunker, embedder, store = make_store(chroma_dir)
    run(store, chunker, embedder)
    raw = json.loads(store.sidecar_path.read_text(encoding="utf-8"))
    store.sidecar_path.write_text(json.dumps({**raw, "schema_version": 99}), encoding="utf-8")
    with pytest.raises(SidecarError, match="schema version 99"):
        store.load_sidecar()
    with pytest.raises(SidecarError, match="malformed"):
        Sidecar.from_dict({"schema_version": 1})


# --- Chroma's default embedder and the workspace default ----------------------------------


def test_chroma_default_embedding_function_never_built(workspace, chroma_dir, monkeypatch):
    import chromadb.api.types as chroma_types
    from chromadb.utils import embedding_functions

    def refuse(*_args, **_kwargs):
        raise AssertionError("Chroma's default embedding function was used")

    for cls in {chroma_types.DefaultEmbeddingFunction, embedding_functions.ONNXMiniLM_L6_V2}:
        monkeypatch.setattr(cls, "__init__", refuse)
        monkeypatch.setattr(cls, "__call__", refuse)
    chunker, embedder, store = make_store(chroma_dir)
    run(store, chunker, embedder)
    reopened = VectorStore(chroma_dir, chunker.chunker_id, embedder.embedder_id, embedder.dim)
    reopened.query(embedder.embed_query("Python"), n=1)
    run(reopened, chunker, embedder, rebuild=True)


def test_default_workspace_root(monkeypatch):
    monkeypatch.delenv("WORKSPACE_ROOT", raising=False)
    with default_workspace_root():
        assert os.environ["WORKSPACE_ROOT"] == "./data"
    assert "WORKSPACE_ROOT" not in os.environ
    monkeypatch.setenv("WORKSPACE_ROOT", "/somewhere")
    with default_workspace_root():
        assert os.environ["WORKSPACE_ROOT"] == "/somewhere"
    monkeypatch.setenv("WORKSPACE_ROOT", "")
    with default_workspace_root():
        assert os.environ["WORKSPACE_ROOT"] == "./data"
    assert os.environ["WORKSPACE_ROOT"] == ""


# --- Full corpus --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def corpus_index(tmp_path_factory):
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("WORKSPACE_ROOT", str(ROOT / "data"))
        chunker, embedder = SectionChunker(), SpyEmbedder()
        store = VectorStore(
            tmp_path_factory.mktemp("chroma"),
            chunker.chunker_id,
            embedder.embedder_id,
            embedder.dim,
        )
        report = index_directory("resumes", chunker, embedder, store, AS_OF, now=lambda: FIXED_NOW)
        texts = {
            e["path"]: fs_tools.read_file(e["path"])["data"]["content"]
            for e in fs_tools.list_files("resumes")
        }
    return report, store, chunker, texts


def test_full_corpus_index(corpus_index):
    report, store, chunker, texts = corpus_index
    assert len(report.added) == 35 and report.failed == [] and report.skipped == []
    assert report.chunks_written == store.count() == 228
    assert len(store.list_candidates()) == 35
    sidecar = store.load_sidecar()
    assert len(sidecar.candidates) == 35
    for path, text in texts.items():
        parsed = parse_resume(text)
        expected = extract_metadata(path, parsed, AS_OF)
        assert sidecar.candidates[expected.candidate_id].metadata == expected.to_dict()
        chunks = chunker.chunk(path, parsed)
        assert store.chunk_ids(expected.candidate_id) == sorted(c.chunk_id for c in chunks)
