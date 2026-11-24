"""rag.evaluation.experiments and display, plus scripts.build_experiment_collections."""

import os
from collections import defaultdict

import pytest

import rag.evaluation.evaluate as evaluate_module
import rag.evaluation.experiments as experiments
from rag.chunking import SectionChunker
from rag.embedders.fake import FakeEmbedder
from rag.evaluation.display import pivot_text, text_table, to_frame
from rag.evaluation.experiments import (
    TRAP_NAMES,
    ExperimentCollection,
    calibration_variants,
    chunk_stats,
    compare_collections,
    compare_retrieval_configs,
    compare_scoring,
    recall_curve,
    trap_cases,
)
from rag.retrieval import RetrievalConfig
from rag.scoring import ScoreWeights
from rag.sections import parse_resume
from scripts import build_experiment_collections as build_script
from scripts.persona_schema import ROOT

METRICS = {"P@5_strict", "R@10_lenient", "MRR_strict", "nDCG@10", "intrusion@10"}


@pytest.fixture
def collection(fake_corpus):
    return ExperimentCollection("fake", fake_corpus.store, fake_corpus.embedder)


@pytest.fixture
def spy(monkeypatch):
    """Record (keyword_mode, weights, index mode) of every match() made by the runners."""
    calls = []
    for module in (evaluate_module, experiments):
        original = module.match

        def wrapper(job, store, keyword_index, embedder, sidecar, config, *args, _o=original, **kw):
            calls.append((config.keyword_mode.value, kw.get("weights"), keyword_index.mode.value))
            return _o(job, store, keyword_index, embedder, sidecar, config, *args, **kw)

        monkeypatch.setattr(module, "match", wrapper)
    return calls


def test_compare_collections_columns_and_determinism(collection):
    rows = compare_collections([collection], modes=("hybrid", "keyword"))
    assert [(r["collection"], r["mode"], r["order"]) for r in rows] == [
        ("fake", "hybrid", "retrieval"),
        ("fake", "hybrid", "scored"),
        ("fake", "keyword", "retrieval"),
        ("fake", "keyword", "scored"),
    ]
    assert set(rows[0]) >= METRICS
    assert rows == compare_collections([collection], modes=("hybrid", "keyword"))
    only = compare_collections([collection], modes=("semantic",), orders=("scored",))
    assert [r["order"] for r in only] == ["scored"]


@pytest.mark.parametrize("threshold", ["strict", "lenient"])
def test_recall_curve_is_non_decreasing(collection, threshold):
    rows = recall_curve(collection, modes=("hybrid", "keyword"), threshold=threshold)
    assert set(rows[0]) == {"collection", "mode", "threshold", "k", "recall"}
    by_mode = defaultdict(list)
    for r in rows:
        by_mode[r["mode"]].append((r["k"], r["recall"]))
    for points in by_mode.values():
        assert [k for k, _ in points] == list(range(1, 16))
        values = [v for _, v in points]
        assert values == sorted(values) and all(0 <= v <= 1 for v in values)
    with pytest.raises(ValueError):
        recall_curve(collection, threshold="loose")


def test_retrieval_variants_are_applied(collection, spy):
    variants = {
        "skills": RetrievalConfig(),
        "words+skills": RetrievalConfig(keyword_mode="words+skills"),
    }
    rows = compare_retrieval_configs(collection, variants, modes=("keyword",))
    assert {r["variant"] for r in rows} == {"skills", "words+skills"}
    assert {"variant", "collection", "mode", "order"} | METRICS <= set(rows[0])
    assert {(mode, index) for mode, _, index in spy} == {
        ("skills", "skills"),
        ("words+skills", "words+skills"),
    }


def test_scoring_variants_are_applied(collection, spy):
    semantic_only = ScoreWeights(1, 0, 0, 0)
    result = compare_scoring(collection, {"default": ScoreWeights(), "semantic": semantic_only})
    assert [r["variant"] for r in result.metrics] == ["default", "semantic"]
    assert all(r["order"] == "scored" for r in result.metrics)
    assert {w for _, w, _ in spy} == {ScoreWeights(), semantic_only}
    assert set(result.top10[0]) == {
        "variant",
        "jd_id",
        "rank",
        "candidate_id",
        "label",
        "match_score",
    }
    per = defaultdict(int)
    for r in result.top10:
        per[(r["variant"], r["jd_id"])] += 1
    assert len(per) == 12 and set(per.values()) == {10}


def test_calibration_variants_in_memory(collection, fake_corpus):
    before = fake_corpus.store.sidecar_path.read_bytes()
    rows = calibration_variants(collection)
    assert fake_corpus.store.sidecar_path.read_bytes() == before  # nothing written
    assert [(r["floor_pct"], r["ceiling_pct"]) for r in rows] == [(50, 95), (50, 99), (50, 100)]
    ceilings = [r["ceiling"] for r in rows]
    assert ceilings == sorted(ceilings) and len({r["floor"] for r in rows}) == 1
    assert set(rows[0]) == {
        "collection",
        "floor_pct",
        "ceiling_pct",
        "floor",
        "ceiling",
        "eligible_pairs",
        "saturated",
        "score_min",
        "score_max",
        "score_std",
    }
    assert all(r["eligible_pairs"] == 18 and 0 <= r["saturated"] <= 18 for r in rows)
    saturated = [r["saturated"] for r in rows]
    assert saturated == sorted(saturated, reverse=True)  # a higher ceiling saturates fewer
    assert rows == calibration_variants(collection)


def test_every_trap_case_is_present(collection):
    variants = {
        "skills": RetrievalConfig(),
        "words+skills": RetrievalConfig(keyword_mode="words+skills"),
    }
    rows = trap_cases(collection, variants)
    assert len(TRAP_NAMES) == 7 and len(rows) == 14
    assert [r["trap"] for r in rows[:7]] == list(TRAP_NAMES)
    assert {r["variant"] for r in rows} == {"skills", "words+skills"}
    assert set(rows[0]) == {
        "trap",
        "variant",
        "jd_id",
        "candidate_id",
        "label",
        "order",
        "rank",
        "score",
        "detail",
    }
    hobby = {r["variant"]: r for r in rows if "hobbies" in r["trap"]}
    assert hobby["skills"]["rank"] is None  # hobbies carry no skill tokens
    assert (
        hobby["words+skills"]["rank"] is not None and "weighted" in hobby["words+skills"]["detail"]
    )
    emily = next(r for r in rows if r["candidate_id"] == "emily_clarke")
    assert emily["label"] == "D" and "Java" in emily["detail"]
    assert rows == trap_cases(collection, variants)


def test_chunk_stats_match_the_chunker(collection):
    import fs_tools

    words = defaultdict(list)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("WORKSPACE_ROOT", str(ROOT / "data"))
        for entry in fs_tools.list_files("resumes"):
            parsed = parse_resume(fs_tools.read_file(entry["path"])["data"]["content"])
            for chunk in SectionChunker().chunk(entry["path"], parsed):
                words[chunk.section].append(len(chunk.text.split()))
    rows = chunk_stats(collection)
    assert [r["section"] for r in rows] == sorted(words)
    for r in rows:
        counts = words[r["section"]]
        assert (r["chunks"], r["words_min"], r["words_max"]) == (
            len(counts),
            min(counts),
            max(counts),
        )
    hobbies = next(r for r in rows if r["section"] == "hobbies")
    assert hobbies["words_median"] == 3.5


# --- Display ------------------------------------------------------------------------------


def test_display_helpers_emit_no_ansi(collection):
    rows = [
        {"mode": "hybrid", "metric": "nDCG@10", "value": 0.98},
        {"mode": "keyword", "metric": "nDCG@10", "value": None},
    ]
    text = text_table(rows)
    assert "\x1b" not in text and text.splitlines()[0].split() == ["mode", "metric", "value"]
    assert "0.980" in text and "-" in text.splitlines()[-1]
    assert "\x1b" not in pivot_text(rows, "metric", "mode", "value")
    frame = to_frame(rows, ["mode", "value"])
    assert list(frame.columns) == ["mode", "value"]
    assert "\x1b" not in frame.to_string() and "\x1b" not in chunk_stats(collection).text()
    assert text_table([]) == "(no rows)"


# --- Build script -------------------------------------------------------------------------


def test_build_script_with_only_and_a_fake_embedder(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("WORKSPACE_ROOT", str(ROOT / "data"))
    argv = ["--only", "section-noctx-bge-small"]
    assert build_script.main(argv, embedder_override=lambda s: FakeEmbedder(), load_env=False) == 0
    out = capsys.readouterr().out
    assert "resumes__section-v1-noctx__fake-d64" in out and "35 added" in out
    assert "calibrate" in out and "floor" in out
    assert build_script.main(argv, embedder_override=lambda s: FakeEmbedder(), load_env=False) == 0
    out = capsys.readouterr().out
    assert "35 unchanged" in out and "calibration kept" in out
    assert os.listdir(tmp_path / "chroma")


def test_build_script_skips_gemini_without_a_key(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert build_script.main(["--only", "section-gemini"], load_env=False) == 0
    assert capsys.readouterr().out == "section-gemini: skipped (GEMINI_API_KEY is not set)\n"
    assert not (tmp_path / "chroma").exists()
