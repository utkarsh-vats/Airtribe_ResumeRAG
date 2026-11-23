"""rag.calibration: percentiles, label-free computation, sidecar persistence, fallbacks, CLI."""

import builtins
import json
import pathlib
import shutil
from datetime import UTC, date, datetime

import pytest

import resume_rag
from rag.calibration import (
    Calibration,
    compute_calibration,
    fallback_calibration,
    load_calibration,
    percentile,
    save_calibration,
)
from rag.chunking import SectionChunker
from rag.embedders.fake import FakeEmbedder
from rag.embedders.huggingface import HuggingFaceEmbedder
from rag.indexer import index_directory
from rag.jd_parser import parse_job_description
from rag.keyword_index import KeywordIndex
from rag.store import VectorStore
from scripts.persona_schema import ROOT

JD_DIR = ROOT / "data" / "job_descriptions"
FIXED_NOW = datetime(2026, 11, 23, 9, 30, tzinfo=UTC)


def jobs() -> dict:
    return {
        f"job_descriptions/{p.name}": parse_job_description(p.read_text("utf-8"))
        for p in sorted(JD_DIR.glob("*.txt"))
    }


# --- Maths --------------------------------------------------------------------------------


def test_percentile_linear_interpolation():
    assert percentile([4, 1, 3, 2], 50) == 2.5
    assert percentile([1, 2, 3, 4], 95) == pytest.approx(3.85)
    assert percentile([1, 2, 3, 4], 0) == 1 and percentile([1, 2, 3, 4], 100) == 4
    assert percentile([7.0], 95) == 7.0
    assert percentile(list(range(101)), 95) == 95
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1], 101)


def test_normalise_clamps():
    cal = Calibration(0.6, 0.8, "collection")
    assert cal.normalise(0.5) == 0.0
    assert cal.normalise(0.7) == pytest.approx(0.5)
    assert cal.normalise(0.9) == 1.0
    flat = Calibration(0.7, 0.7, "collection")
    assert (flat.normalise(0.69), flat.normalise(0.7)) == (0.0, 1.0)
    with pytest.raises(ValueError):
        Calibration(float("nan"), 1.0, "collection")


def test_round_trip():
    cal = Calibration(0.6, 0.8, "collection", "fake-d64", ("a.txt",), 3, "2026-11-23", {"x": 1})
    assert Calibration.from_dict(json.loads(json.dumps(cal.to_dict()))) == cal


# --- Computation --------------------------------------------------------------------------


def _forbid_qrels(monkeypatch):
    def guard(original):
        def wrapper(self_or_path, *args, **kwargs):
            if "qrels" in str(self_or_path):
                raise AssertionError(f"calibration read {self_or_path}")
            return original(self_or_path, *args, **kwargs)

        return wrapper

    for name in ("open", "read_text", "read_bytes"):
        monkeypatch.setattr(pathlib.Path, name, guard(getattr(pathlib.Path, name)))
    monkeypatch.setattr(builtins, "open", guard(builtins.open))


def test_compute_reads_no_labels(fake_corpus, monkeypatch):
    all_jobs = jobs()
    _forbid_qrels(monkeypatch)
    with pytest.raises(AssertionError, match="qrels"):  # the guard works
        (ROOT / "data" / "ground_truth" / "qrels.json").read_text()
    cal = compute_calibration(
        all_jobs,
        fake_corpus.store,
        fake_corpus.keyword_index,
        fake_corpus.embedder,
        now=lambda: FIXED_NOW,
    )
    assert cal.source == "collection" and cal.embedder_id == "fake-d64"
    assert cal.floor <= cal.ceiling
    assert 0 < cal.pairs <= 6 * 35
    assert cal.jd_files == tuple(sorted(all_jobs))
    assert cal.computed_at == "2026-11-23T09:30:00+00:00"
    assert cal.retrieval["second_chunk_weight"] == 0.3


def test_compute_requires_jobs(fake_corpus):
    with pytest.raises(ValueError, match="at least one JD"):
        compute_calibration({}, fake_corpus.store, None, fake_corpus.embedder)


# --- Persistence and fallbacks ------------------------------------------------------------


@pytest.fixture
def small_store(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    (workspace / "resumes").mkdir(parents=True)
    for name in ("john_doe.pdf", "noah_williams.txt", "elena_rodriguez.docx"):
        shutil.copy(ROOT / "data" / "resumes" / name, workspace / "resumes" / name)
    shutil.copytree(JD_DIR, workspace / "job_descriptions")
    monkeypatch.setenv("WORKSPACE_ROOT", str(workspace))
    chunker, embedder = SectionChunker(), FakeEmbedder()
    store = VectorStore(tmp_path / "chroma", chunker.chunker_id, embedder.embedder_id, 64)
    index_directory("resumes", chunker, embedder, store, date(2026, 11, 1))
    return workspace, store, chunker, embedder


def test_saved_in_the_sidecar_and_cleared_by_changes(small_store):
    workspace, store, chunker, embedder = small_store
    cal = compute_calibration(
        jobs(), store, KeywordIndex.from_store(store, mode="skills"), embedder
    )
    save_calibration(store, cal)
    raw = json.loads(store.sidecar_path.read_text("utf-8"))
    assert raw["calibration"]["floor"] == cal.floor
    loaded, warnings = load_calibration(store.load_sidecar(), embedder)
    assert loaded == cal and warnings == []

    report = index_directory("resumes", chunker, embedder, store, date(2026, 11, 1))
    assert not report.calibration_cleared and store.load_sidecar().calibration is not None
    shutil.copy(ROOT / "data" / "resumes" / "zoe_turner.txt", workspace / "resumes")
    report = index_directory("resumes", chunker, embedder, store, date(2026, 11, 1))
    assert report.calibration_cleared and store.load_sidecar().calibration is None


def test_fallback_warning(fake_corpus):
    cal, warnings = load_calibration(fake_corpus.sidecar, FakeEmbedder())
    assert cal.source == "generic" and (cal.floor, cal.ceiling) == (0.0, 1.0)
    assert len(warnings) == 1 and "no semantic calibration" in warnings[0]
    assert "resume_rag.py calibrate" in warnings[0]


def test_model_fallback_from_the_config_table():
    embedder = HuggingFaceEmbedder("BAAI/bge-small-en-v1.5", loader=lambda *a: None)
    cal, warning = fallback_calibration(embedder)
    assert cal.source == "model"
    assert (cal.floor, cal.ceiling) == (
        embedder.config.semantic_floor,
        embedder.config.semantic_ceiling,
    )
    assert "BAAI/bge-small-en-v1.5 fallback" in warning


# --- CLI ----------------------------------------------------------------------------------


def test_calibrate_command(small_store, monkeypatch, capsys):
    _, store, _, _ = small_store
    monkeypatch.setenv("CHROMA_DIR", str(store.chroma_dir))
    code = resume_rag.main(["calibrate", "--embedder", "fake", "--json"], load_env=False)
    out = capsys.readouterr().out
    assert code == 0
    data = json.loads(out)
    assert len(data["jd_files"]) == 6 and data["source"] == "collection"
    assert store.load_sidecar().calibration == data
    code = resume_rag.main(["calibrate", "--embedder", "fake"], load_env=False)
    assert code == 0 and "floor (median)" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["calibrate", "--embedder", "fake", "--chunker", "fixed"], "no index for collection"),
        (["calibrate", "--embedder", "fake", "--jd-dir", "resumes/none"], "does not exist"),
    ],
)
def test_calibrate_errors(small_store, monkeypatch, capsys, argv, message):
    _, store, _, _ = small_store
    monkeypatch.setenv("CHROMA_DIR", str(store.chroma_dir))
    assert resume_rag.main(argv, load_env=False) == 2
    assert message in capsys.readouterr().err
