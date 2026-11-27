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


# --- Notebook runners (step 14b) ----------------------------------------------------------


def test_dataset_counts_and_labels():
    from rag.evaluation.experiments import dataset_counts, label_counts

    rows = dataset_counts()
    for dimension in ("format", "layout", "date_format"):
        assert sum(r["resumes"] for r in rows if r["dimension"] == dimension) == 35
    assert {r["value"] for r in rows if r["dimension"] == "format"} == {"pdf", "docx", "txt"}
    assert label_counts() == [
        {"jd_id": jd, "S": 3, "P": 3, "D": 2, "none": 27}
        for jd in (
            "data_engineer",
            "devops_sre",
            "frontend",
            "java_backend",
            "ml_engineer",
            "python_backend",
        )
    ]


def test_resume_lengths(monkeypatch):
    from rag.evaluation.experiments import resume_lengths

    monkeypatch.setenv("WORKSPACE_ROOT", str(ROOT / "data"))
    rows = resume_lengths()
    assert len(rows) == 35 and all(r["words"] > 50 for r in rows)
    assert {r["format"] for r in rows} == {"pdf", "docx", "txt"}


def test_extraction_summary_and_traps(collection, fake_corpus):
    from rag.evaluation.experiments import extraction_summary, extraction_traps

    summary = {r["metric"]: r["value"] for r in extraction_summary(fake_corpus.sidecar)}
    assert summary["candidates"] == 35 and summary["skills F1 (micro)"] == 1.0
    assert summary["candidates with a mismatch"] == 0
    traps = {r["candidate_id"]: r for r in extraction_traps(collection)}
    assert list(traps) == [
        "peter_hughes",
        "noah_williams",
        "mateo_garcia",
        "omar_haddad",
        "ishaan_gupta",
        "nina_kowalski",
    ]
    assert traps["peter_hughes"]["extracted"] == "Excel"
    assert "k8s -> Kubernetes" in traps["noah_williams"]["extracted"]
    assert traps["mateo_garcia"]["extracted"] == "Python 52 months (4.3 yrs)"
    assert "76 months" in traps["mateo_garcia"]["detail"]  # summed would over-count
    assert traps["omar_haddad"]["extracted"] == "Python 0 months, Django 0 months"
    assert traps["ishaan_gupta"]["extracted"].startswith("total 0 months")
    assert traps["nina_kowalski"]["detail"].endswith("from a skills section: none")


def test_candidate_chunk_tokens_show_the_fixed_chunker_leak(tmp_path, monkeypatch, collection):
    from rag.chunking import FixedSizeChunker
    from rag.evaluation.experiments import candidate_chunk_tokens
    from rag.indexer import index_directory
    from rag.store import VectorStore

    workspace = tmp_path / "workspace"
    (workspace / "resumes").mkdir(parents=True)
    import shutil
    from datetime import date

    shutil.copy(ROOT / "data" / "resumes" / "peter_hughes.pdf", workspace / "resumes")
    monkeypatch.setenv("WORKSPACE_ROOT", str(workspace))
    chunker, embedder = FixedSizeChunker(), FakeEmbedder()
    store = VectorStore(tmp_path / "chroma", chunker.chunker_id, embedder.embedder_id, 64)
    index_directory("resumes", chunker, embedder, store, date(2026, 11, 1))
    fixed = ExperimentCollection("fixed", store, embedder)

    (leaky,) = candidate_chunk_tokens(fixed, "peter_hughes", contains="Spark Energy")
    assert leaky["body_start"] == 0 and "skill:spark" in leaky["skill_tokens"]
    assert leaky["matching_lines"] and all("Spark Energy" in ln for ln in leaky["matching_lines"])
    (clean,) = candidate_chunk_tokens(collection, "peter_hughes", contains="Spark Energy")
    assert clean["body_start"] == 2 and "skill:spark" not in clean["skill_tokens"]
    assert len(candidate_chunk_tokens(collection, "peter_hughes")) > 1


def test_query_token_table():
    from rag.evaluation.experiments import query_token_table

    rows = query_token_table({"fake": FakeEmbedder()})
    assert len(rows) == 6 and {r["model"] for r in rows} == {"fake"}
    assert all(r["query_text"] < r["whole"] for r in rows)
    assert set(rows[0]) == {
        "jd_id",
        "model",
        "max_tokens",
        "whole",
        "query_text",
        "whole_over",
        "query_over",
    }


def test_per_jd_variant_rows(collection):
    variants = {
        "query_text": RetrievalConfig(),
        "full_text": RetrievalConfig(query_source="full_text"),
    }
    rows = compare_retrieval_configs(
        collection, variants, modes=("semantic",), orders=("retrieval",), per_jd=True
    )
    assert len(rows) == 2 * 6 and {r["variant"] for r in rows} == set(variants)
    assert {"jd_id", "mode", "order", "nDCG@10"} <= set(rows[0])


def test_open_collections_never_creates_a_missing_one(tmp_path, fake_corpus):
    from rag.evaluation.experiments import CollectionSpec, open_collections

    specs = (
        CollectionSpec("present", "section-v1", "m", "section-bge-small"),
        CollectionSpec("absent", "fixed-w120-o30", "m", "fixed-bge-small"),
    )
    chroma = fake_corpus.store.chroma_dir
    before = sorted(p.name for p in chroma.iterdir())
    opened, messages = open_collections(specs, chroma, embedder_factory=lambda m: FakeEmbedder())
    assert opened["present"] is not None and opened["absent"] is None
    assert messages[1].endswith(
        "python -m scripts.build_experiment_collections --only fixed-bge-small"
    )
    assert sorted(p.name for p in chroma.iterdir()) == before


def test_notebook_specs_match_the_build_script():
    from rag.evaluation.experiments import NOTEBOOK_COLLECTIONS

    assert {s.build_name for s in NOTEBOOK_COLLECTIONS} <= set(build_script.NAMES)


# --- Figures ------------------------------------------------------------------------------


def test_figures_are_saved_with_relative_paths(tmp_path, monkeypatch):
    for key in (
        "HF_HUB_DISABLE_SYMLINKS_WARNING",
        "HF_HUB_VERBOSITY",
        "HF_HUB_OFFLINE",
        "HF_HUB_DISABLE_PROGRESS_BARS",
        "TRANSFORMERS_VERBOSITY",
        "TOKENIZERS_PARALLELISM",
        "ANONYMIZED_TELEMETRY",
    ):
        monkeypatch.delenv(key, raising=False)  # restored after the test
    import matplotlib

    matplotlib.use("Agg")
    from rag.evaluation.display import (
        MODE_COLORS,
        bar_chart,
        histogram,
        line_chart,
        notebook_setup,
        save_figure,
    )

    notebook_setup()
    assert os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] == "1"
    assert os.environ["HF_HUB_VERBOSITY"] == "error" and "HF_HUB_OFFLINE" in os.environ
    rows = [
        {"metric": m, "mode": mode, "panel": p, "value": v}
        for m, v in (("nDCG@10", 0.9), ("P@5", 0.6))
        for mode in ("semantic", "hybrid")
        for p in ("a", "b")
    ]
    fig = bar_chart(
        rows,
        "metric",
        "mode",
        "value",
        title="t",
        ylabel="score",
        palette=MODE_COLORS,
        panel="panel",
    )
    labels = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert len(fig.axes) == 2 and labels.count("0.900") == 4  # every bar annotated
    assert fig.axes[0].get_ylim()[0] == 0.0
    assert fig.axes[0].patches[0].get_facecolor()[:3] == matplotlib.colors.to_rgb(
        MODE_COLORS["semantic"]
    )
    curve = [{"k": k, "r": k / 10, "mode": "hybrid"} for k in range(1, 6)]
    line = line_chart(curve, "k", "r", "mode", title="t", xlabel="k", ylabel="recall")
    hist = histogram([1, 2, 3], bins=3, title="t", xlabel="words")
    for fig_, name in ((fig, "bars"), (line, "line"), (hist, "hist")):
        rel = save_figure(fig_, "00_test", name, tmp_path)
        assert (tmp_path / f"00_test_{name}.png").exists() and ":" not in rel


def test_melt():
    from rag.evaluation.display import melt

    rows = [{"mode": "hybrid", "a": 0.5, "b": 0.25, "x": 1}]
    assert melt(rows, ["mode"], ["a", "b"]) == [
        {"mode": "hybrid", "metric": "a", "value": 0.5},
        {"mode": "hybrid", "metric": "b", "value": 0.25},
    ]


def test_to_frame_keeps_integer_columns_with_gaps():
    frame = to_frame([{"rank": 3, "score": 0.5}, {"rank": None, "score": None}])
    assert str(frame["rank"].dtype) == "Int64" and frame["rank"].isna().tolist() == [False, True]
    assert "3.0" not in frame.to_string()
