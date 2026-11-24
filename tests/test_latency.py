"""rag.evaluation.latency: percentiles, query and index latency, environment info."""

import shutil
import subprocess
from datetime import UTC, date, datetime
from types import SimpleNamespace

import pytest

import rag.evaluation.latency as latency
from rag.chunking import SectionChunker
from rag.embedders.fake import FakeEmbedder
from rag.evaluation.latency import (
    INDEX_STAGES,
    QUERY_STAGES,
    environment_info,
    git_commit,
    measure_index_latency,
    measure_query_latency,
    percentile,
)
from scripts.persona_schema import ROOT

PY_TEXT = (ROOT / "data" / "job_descriptions" / "python_backend.txt").read_text("utf-8")


class StepTimer:
    """Each call advances one millisecond, so every timed interval is a count of calls."""

    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 0.001
        return self.now


# --- Percentiles --------------------------------------------------------------------------


def test_percentile_by_hand():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([4, 1, 3, 2], 0) == 1 and percentile([1, 2, 3, 4], 100) == 4
    assert percentile([7.5], 50) == percentile([7.5], 95) == 7.5
    assert percentile(list(range(1, 21)), 95) == pytest.approx(19.05)  # 19 + 0.05 * (20 - 19)
    assert percentile([10, 20], 25) == 12.5
    with pytest.raises(ValueError):
        percentile([], 50)


# --- Query latency ------------------------------------------------------------------------


def run(fake_corpus, **kwargs):
    kwargs.setdefault("timer", StepTimer())
    return measure_query_latency(
        {"python_backend": PY_TEXT},
        fake_corpus.store,
        fake_corpus.keyword_index,
        fake_corpus.embedder,
        **kwargs,
    )


def test_exact_stage_numbers_with_a_fake_timer(fake_corpus):
    result = run(fake_corpus, modes=("hybrid", "semantic"), repeat=2, warmup=1)
    assert result.model_load_ms == 1.0  # one step around the warm-up embed
    hybrid = result.macro[("hybrid", "strict")]
    # Every stage is one step; total spans parse + 14 timer calls in match() + output.
    assert {stage: hybrid[stage]["p50"] for stage in QUERY_STAGES} == {
        "parse": 1.0,
        "filter": 1.0,
        "embed_query": 1.0,
        "semantic_search": 1.0,
        "keyword_search": 1.0,
        "aggregate_fuse": 1.0,
        "backfill": 1.0,
        "scoring": 1.0,
        "output": 1.0,
        "total": 17.0,
    }
    semantic = result.macro[("semantic", "soft")]
    assert semantic["keyword_search"]["p95"] == 0.0 and semantic["total"]["max"] == 15.0
    assert len(result.rows) == 2 * 2 * len(QUERY_STAGES)
    assert result.to_dict()["macro"]["hybrid/strict"]["total"]["mean"] == 17.0


def test_warmup_runs_are_executed_but_not_counted(fake_corpus, monkeypatch):
    calls = []

    def fake_query(*args):
        calls.append(len(calls))
        index = len(calls) - 1
        warm = index % 4 == 0  # warmup=1, repeat=3 per combination
        seconds = 9.9 if warm else 0.001 * (index % 4)
        return {stage: seconds for stage in QUERY_STAGES}

    monkeypatch.setattr(latency, "_one_query", fake_query)
    result = run(fake_corpus, modes=("hybrid",), repeat=3, warmup=1)
    assert len(calls) == 2 * (1 + 3)  # strict and soft, each 1 warm-up + 3 runs
    stats = result.macro[("hybrid", "soft")]["total"]
    assert stats == {"p50": 2.0, "p95": 2.9, "mean": 2.0, "max": 3.0}


def test_match_is_called_warmup_plus_repeat_times(fake_corpus, monkeypatch):
    calls = []
    original = latency.match

    def spy(*args, **kwargs):
        calls.append(kwargs["clock"])
        return original(*args, **kwargs)

    monkeypatch.setattr(latency, "match", spy)
    timer = StepTimer()
    run(fake_corpus, modes=("keyword",), filter_modes=("strict",), repeat=4, warmup=2, timer=timer)
    assert len(calls) == 6 and all(c is timer for c in calls)


def test_bad_arguments(fake_corpus):
    with pytest.raises(ValueError):
        run(fake_corpus, repeat=0)
    with pytest.raises(ValueError):
        run(fake_corpus, filter_modes=("loose",))


# --- Index latency ------------------------------------------------------------------------


@pytest.fixture
def small_workspace(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    (workspace / "resumes").mkdir(parents=True)
    for name in ("john_doe.pdf", "noah_williams.txt", "elena_rodriguez.docx"):
        shutil.copy(ROOT / "data" / "resumes" / name, workspace / "resumes" / name)
    monkeypatch.setenv("WORKSPACE_ROOT", str(workspace))
    real = tmp_path / "real_chroma"
    monkeypatch.setenv("CHROMA_DIR", str(real))
    return tmp_path, real


def test_index_latency_uses_a_temporary_directory(small_workspace):
    tmp_path, real = small_workspace
    temp_root = tmp_path / "temp"
    temp_root.mkdir()
    result = measure_index_latency(
        "resumes", SectionChunker(), FakeEmbedder(), date(2026, 11, 1), temp_root=temp_root
    )
    assert result.files == 3 and result.failed == 0 and result.chunks > 0
    assert result.rerun_unchanged == 3
    assert set(result.totals_ms) == set(INDEX_STAGES) == set(result.per_file)
    assert result.per_file["total"]["p95"] >= result.per_file["total"]["p50"] >= 0
    assert result.first_run_ms > 0 and result.rerun_ms > 0
    assert result.embed_chunks_per_second is None or result.embed_chunks_per_second > 0
    assert list(temp_root.iterdir()) == []  # removed afterwards
    assert not real.exists()  # the real CHROMA_DIR was never touched


def test_index_latency_cleans_up_when_indexing_raises(small_workspace, monkeypatch):
    tmp_path, real = small_workspace
    temp_root = tmp_path / "temp"
    temp_root.mkdir()

    def boom(*args, **kwargs):
        raise RuntimeError("indexing exploded")

    monkeypatch.setattr(latency, "index_directory", boom)
    with pytest.raises(RuntimeError, match="exploded"):
        measure_index_latency(
            "resumes", SectionChunker(), FakeEmbedder(), date(2026, 11, 1), temp_root=temp_root
        )
    assert list(temp_root.iterdir()) == [] and not real.exists()


# --- Environment --------------------------------------------------------------------------


def test_environment_info():
    fixed = datetime(2026, 11, 25, 8, 0, tzinfo=UTC)
    done = SimpleNamespace(returncode=0, stdout="abc1234\n")
    info = environment_info(FakeEmbedder(), now=lambda: fixed, run=lambda *a, **k: done)
    assert set(info) == {
        "python",
        "platform",
        "processor",
        "torch_device",
        "embedder_id",
        "model",
        "chromadb",
        "sentence_transformers",
        "git_commit",
        "timestamp",
    }
    assert info["git_commit"] == "abc1234" and info["timestamp"] == "2026-11-25T08:00:00+00:00"
    assert info["torch_device"] is None and info["embedder_id"] == "fake-d64"
    assert info["chromadb"] == "1.5.9" and info["python"].startswith("3.14")


def test_git_missing_or_failing():
    def missing(*args, **kwargs):
        raise FileNotFoundError("git")

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("git", 10)

    assert git_commit(missing) is None and git_commit(timeout) is None
    assert git_commit(lambda *a, **k: SimpleNamespace(returncode=128, stdout="")) is None
    real = git_commit()
    assert real is None or (real.isalnum() and 4 <= len(real) <= 12)
