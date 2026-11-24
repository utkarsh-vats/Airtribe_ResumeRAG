"""rag.evaluation: metrics, ground-truth loading, the three layers, and label isolation."""

import ast
import builtins
import json
import math
import pathlib
from datetime import date
from types import SimpleNamespace

import pytest
from match_helpers import PY_JD, meta

from rag.evaluation.evaluate import (
    evaluate_end_to_end,
    evaluate_extraction,
    evaluate_filter,
    evaluate_ranking,
    load_jobs,
)
from rag.evaluation.ground_truth import (
    GROUND_TRUTH_DIR,
    GroundTruthError,
    load_manifest,
    load_qrels,
    parse_manifest,
    parse_qrels,
)
from rag.evaluation.metrics import (
    LENIENT,
    STRICT,
    distractor_intrusion_at_k,
    macro_average,
    mrr,
    ndcg_at_k,
    precision_at_k,
    r_precision,
    recall_at_k,
)
from rag.metadata import DegreeLevel, Education
from scripts.persona_schema import ROOT

JD_PATHS = sorted((ROOT / "data" / "job_descriptions").glob("*.txt"))
GRADES = {"a": 2, "b": 1, "c": 0, "d": 2}


# --- Metrics ------------------------------------------------------------------------------


def test_precision_and_recall_strict_vs_lenient():
    ranking = ["a", "b", "c", "d"]
    assert precision_at_k(ranking, GRADES, 2, STRICT) == 0.5
    assert precision_at_k(ranking, GRADES, 2, LENIENT) == 1.0
    assert recall_at_k(ranking, GRADES, 2, STRICT) == 0.5  # a of {a, d}
    assert recall_at_k(ranking, GRADES, 2, LENIENT) == pytest.approx(2 / 3)  # a, b of {a, b, d}
    assert r_precision(ranking, GRADES, STRICT) == 0.5  # R = 2
    assert r_precision(ranking, GRADES, LENIENT) == pytest.approx(2 / 3)  # R = 3


def test_mrr_strict_vs_lenient():
    assert mrr(["c", "b", "a"], GRADES, STRICT) == pytest.approx(1 / 3)
    assert mrr(["c", "b", "a"], GRADES, LENIENT) == 0.5
    assert mrr(["c"], GRADES, STRICT) == 0.0  # nothing relevant ranked


def test_graded_ndcg_by_hand():
    grades = {"a": 2, "b": 1, "c": 0}
    dcg = (2**1 - 1) / math.log2(2) + (2**2 - 1) / math.log2(3) + 0  # b, a, c
    ideal = (2**2 - 1) / math.log2(2) + (2**1 - 1) / math.log2(3)
    assert ndcg_at_k(["b", "a", "c"], grades, 3) == pytest.approx(dcg / ideal)
    assert ndcg_at_k(["b", "a", "c"], grades, 3) == pytest.approx(0.7967, abs=1e-4)
    assert ndcg_at_k(["a", "b", "c"], grades, 3) == pytest.approx(1.0)


def test_ndcg_ties_and_cutoff():
    grades = {"a": 2, "b": 2, "c": 1}
    assert ndcg_at_k(["a", "b", "c"], grades, 3) == pytest.approx(1.0)
    assert ndcg_at_k(["b", "a", "c"], grades, 3) == pytest.approx(1.0)  # equal grades swap
    assert ndcg_at_k(["c"], grades, 1) == pytest.approx(1 / 3)  # gain 1 vs ideal 3


def test_empty_relevant_set_is_none_and_skipped():
    grades = {"a": 0, "b": 0}
    assert precision_at_k(["a"], grades, 5) is None
    assert recall_at_k(["a"], grades, 5) is None
    assert mrr(["a"], grades) is None and r_precision(["a"], grades) is None
    assert ndcg_at_k(["a"], grades, 5) is None
    assert macro_average([0.5, None, 1.0]) == 0.75
    assert macro_average([None, None]) is None


def test_k_beyond_the_list():
    assert precision_at_k(["a"], GRADES, 10) == 0.1
    assert recall_at_k(["a", "d"], GRADES, 10) == 1.0
    assert ndcg_at_k([], GRADES, 5) == 0.0


def test_intrusion():
    labels = {"x": "D", "y": "D", "a": "S"}
    assert distractor_intrusion_at_k(["x", "a", "y"], labels, 2) == 0.5
    assert distractor_intrusion_at_k(["x"], labels, 5) == 0.2
    assert distractor_intrusion_at_k([], labels, 5) == 0.0


def test_duplicates_and_bad_k_raise():
    with pytest.raises(ValueError, match="duplicate candidate ids in ranking: \\['a'\\]"):
        precision_at_k(["a", "b", "a"], GRADES, 2)
    for fn in (mrr, r_precision):
        with pytest.raises(ValueError, match="duplicate"):
            fn(["a", "a"], GRADES)
    with pytest.raises(ValueError, match="k must be"):
        ndcg_at_k(["a"], GRADES, 0)


# --- Ground truth -------------------------------------------------------------------------


def test_load_the_real_files():
    qrels, manifest = load_qrels(), load_manifest()
    assert len(qrels.jds) == 6 and len(qrels.candidate_ids) == 35
    py = qrels.jds["python_backend"]
    assert py.with_label("S") == {"ananya_iyer", "john_doe", "noah_williams"}
    assert py.grades["john_doe"] == 2 and py.eligible["john_doe"] is True
    assert len(manifest.candidates) == 35
    assert manifest.candidates["noah_williams"].skill_years["Python"] == 6.1


def _qrels(**overrides):
    judgement = {
        "candidate_id": "x",
        "resume_path": "resumes/x.txt",
        "label": "S",
        "grade": 2,
        "eligible": True,
        "unmet": [],
    } | overrides
    jd = {"jd_id": "j", "title": "J", "jd_path": "job_descriptions/j.txt"}
    return {"as_of": "2026-11-01", "jds": [jd | {"judgements": [judgement]}]}


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"label": "Q"}, "unknown label"),
        ({"grade": 1}, "does not match label"),
        ({"eligible": "yes"}, "eligible must be"),
        ({"eligible": False}, "empty exactly when eligible"),
        ({"unmet": ["Python (missing)"]}, "empty exactly when eligible"),
    ],
)
def test_qrels_validation(overrides, message):
    assert parse_qrels(_qrels()).jds["j"].labels == {"x": "S"}
    with pytest.raises(GroundTruthError, match=message):
        parse_qrels(_qrels(**overrides))


def test_qrels_structure_errors(tmp_path):
    with pytest.raises(GroundTruthError, match="missing"):
        parse_qrels({"jds": []})
    data = _qrels()
    data["jds"].append(json.loads(json.dumps(data["jds"][0])) | {"jd_id": "k"})
    data["jds"][1]["judgements"][0]["candidate_id"] = "y"
    with pytest.raises(GroundTruthError, match="different set of candidates"):
        parse_qrels(data)
    with pytest.raises(GroundTruthError, match="not found"):
        load_qrels(tmp_path / "none.json")
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    with pytest.raises(GroundTruthError, match="invalid JSON"):
        load_qrels(tmp_path / "bad.json")


def test_manifest_validation():
    candidate = {
        "candidate_id": "x",
        "name": "X",
        "resume_path": "resumes/x.txt",
        "skills": ["Python"],
        "skill_years": {"Python": 2.0},
        "total_years": 2.0,
        "highest_education": None,
    }
    assert parse_manifest({"as_of": "d", "candidates": [candidate]}).candidates["x"].name == "X"
    bad = candidate | {"skill_years": {"Go": 1.0}}
    with pytest.raises(GroundTruthError, match="skill_years keys differ"):
        parse_manifest({"as_of": "d", "candidates": [bad]})
    with pytest.raises(GroundTruthError, match="duplicate candidate"):
        parse_manifest({"as_of": "d", "candidates": [candidate, candidate]})


# --- Synthetic filter and extraction ------------------------------------------------------


def _sidecar(*metas):
    return SimpleNamespace(
        candidates={m.candidate_id: SimpleNamespace(metadata=m.to_dict()) for m in metas}
    )


def test_filter_with_one_deliberate_disagreement():
    good = meta("good", {"Python": 85, "Django": 30, "PostgreSQL": 40})
    weak = meta("weak", {"Python": 30})
    qrels = {
        "as_of": "2026-11-01",
        "jds": [
            {
                "jd_id": "py",
                "title": "Py",
                "jd_path": "job_descriptions/py.txt",
                "judgements": [
                    {
                        "candidate_id": "good",
                        "label": "S",
                        "grade": 2,
                        "eligible": True,
                        "unmet": [],
                    },
                    # deliberately wrong: the qrels call "weak" eligible
                    {
                        "candidate_id": "weak",
                        "label": "P",
                        "grade": 1,
                        "eligible": True,
                        "unmet": [],
                    },
                ],
            }
        ],
    }
    result = evaluate_filter(_sidecar(good, weak), {"py": PY_JD}, parse_qrels(qrels))
    counts = result.per_jd["py"]
    assert (counts.tp, counts.fp, counts.fn, counts.tn) == (1, 0, 1, 0)
    assert (counts.precision, counts.recall, counts.accuracy) == (1.0, 0.5, 0.5)
    (d,) = result.disagreements
    assert (d.candidate_id, d.system_eligible, d.qrels_eligible) == ("weak", False, True)
    assert d.system_unmet == (
        "Python 5+ yrs (has 2.5)",
        "Django or FastAPI (missing)",
        "PostgreSQL (missing)",
    )
    assert d.qrels_unmet == ()
    assert result.to_dict()["overall"]["agreements"] == 1
    assert list(result.to_dataframe()["jd_id"]) == ["py", "overall"]


def _with_education(m, degree="B.Sc.", year=2016):
    edu = Education(
        degree, DegreeLevel.BACHELORS, "Computer Science", "Northshore University", year
    )
    return type(m)(**{**m.__dict__, "education": (edu,), "highest_education": edu})


def test_extraction_with_one_mismatch_per_field():
    right = _with_education(meta("right", {"Python": 85, "Docker": 30}))
    wrong = _with_education(meta("wrong", {"Python": 61, "Go": 12}, total_months=60), year=2015)
    entry = {
        "resume_path": "resumes/x.txt",
        "highest_education": {
            "degree": "B.Sc.",
            "field": "Computer Science",
            "institution": "Northshore University",
            "end_year": 2016,
        },
    }
    manifest = parse_manifest(
        {
            "as_of": "2026-11-01",
            "candidates": [
                entry
                | {
                    "candidate_id": "right",
                    "name": "Right",
                    "skills": ["Docker", "Python"],
                    "skill_years": {"Docker": 2.5, "Python": 7.1},
                    "total_years": 7.1,
                },
                entry
                | {
                    "candidate_id": "wrong",
                    "name": "Someone Else",  # name
                    "skills": ["Python", "Rust"],  # Go extra, Rust missing
                    "skill_years": {"Python": 5.0, "Rust": 1.0},  # 5.1 extracted
                    "total_years": 6.0,  # 5.0 extracted
                },
            ],
        }
    )
    result = evaluate_extraction(_sidecar(right, wrong), manifest)
    assert result.candidates == 2 and result.name_exact == 0.5
    assert result.skills_micro == pytest.approx({"precision": 0.75, "recall": 0.75, "f1": 0.75})
    assert result.skills_macro == pytest.approx({"precision": 0.75, "recall": 0.75, "f1": 0.75})
    assert result.total_years_exact == 0.5 and result.total_years_mae == pytest.approx(0.5)
    assert result.skill_years_pairs == 3
    assert result.skill_years_mae == pytest.approx(0.1 / 3)
    assert result.education_exact == 0.5
    notes = result.mismatches["wrong"]
    assert [n.split(":")[0] for n in notes] == [
        "name",
        "skills",
        "skills",
        "total_years",
        "skill_years[Python]",
        "highest_education",
    ]
    assert "right" not in result.mismatches
    assert json.loads(json.dumps(result.to_dict()))["mismatches"]["wrong"] == list(notes)
    assert len(result.to_dataframe()) == 6


# --- Corpus (FakeEmbedder) ----------------------------------------------------------------


def test_corpus_filter_agrees_on_all_pairs(fake_corpus):
    result = evaluate_filter(fake_corpus.sidecar, load_jobs(JD_PATHS), load_qrels())
    assert (result.agreements, result.pairs) == (210, 210)
    assert result.disagreements == () and result.missing == ()
    assert result.overall.accuracy == 1.0


def test_corpus_extraction_is_perfect(fake_corpus):
    result = evaluate_extraction(fake_corpus.sidecar, load_manifest())
    assert result.candidates == 35 and result.mismatches == {}
    assert result.name_exact == result.total_years_exact == result.education_exact == 1.0
    assert result.skills_micro["f1"] == result.skills_macro["f1"] == 1.0
    assert result.total_years_mae == 0.0 and result.skill_years_mae == 0.0


def test_corpus_ranking_runs_every_mode_and_order(fake_corpus):
    result = evaluate_ranking(
        JD_PATHS,
        load_qrels(),
        fake_corpus.store,
        fake_corpus.keyword_index,
        fake_corpus.embedder,
    )
    assert len(result.rows) == 6 * 3 * 2
    assert set(result.macro) == {
        (m, o) for m in ("semantic", "keyword", "hybrid") for o in ("retrieval", "scored")
    }
    for row in result.rows:
        for key, value in row.items():
            if key not in ("jd_id", "mode", "order", "ranked"):
                assert value is None or 0.0 <= value <= 1.0, (row["jd_id"], key, value)
    scored = [r for r in result.rows if r["order"] == "scored" and r["mode"] == "hybrid"]
    assert all(r["ranked"] == 35 for r in scored)  # unfiltered: everyone is ranked
    frame = result.macro_dataframe()
    assert len(frame) == 6 and "nDCG@10" in frame.columns
    json.dumps(result.to_dict())


def test_corpus_end_to_end(fake_corpus):
    result = evaluate_end_to_end(
        JD_PATHS, load_qrels(), fake_corpus.store, fake_corpus.keyword_index, fake_corpus.embedder
    )
    assert len(result.rows) == 6
    for row in result.rows:
        assert row["eligible"] == 3 and row["returned"] <= 3
        assert row["s_total"] == 3 and 0 <= row["recall_s"] <= 1
    assert len(result.to_dataframe()) == 6


# --- Label isolation ----------------------------------------------------------------------


def _forbid_ground_truth(monkeypatch):
    root = str(GROUND_TRUTH_DIR.resolve()).lower()

    def guard(original):
        def wrapper(target, *args, **kwargs):
            if str(pathlib.Path(str(target)).resolve()).lower().startswith(root):
                raise AssertionError(f"opened ground truth: {target}")
            return original(target, *args, **kwargs)

        return wrapper

    for name in ("open", "read_text", "read_bytes"):
        monkeypatch.setattr(pathlib.Path, name, guard(getattr(pathlib.Path, name)))
    monkeypatch.setattr(builtins, "open", guard(builtins.open))


def test_matching_and_calibration_never_open_the_ground_truth(fake_corpus, monkeypatch):
    from rag.calibration import compute_calibration
    from rag.matcher import match

    jobs = load_jobs(JD_PATHS)  # before the guard: JDs are not labels
    _forbid_ground_truth(monkeypatch)
    with pytest.raises(AssertionError, match="ground truth"):
        load_qrels()
    for filter_mode in ("strict", "soft"):
        match(
            jobs["python_backend"],
            fake_corpus.store,
            fake_corpus.keyword_index,
            fake_corpus.embedder,
            fake_corpus.sidecar,
            filter_mode=filter_mode,
            as_of=date(2026, 11, 1),
        )
    compute_calibration(jobs, fake_corpus.store, fake_corpus.keyword_index, fake_corpus.embedder)


@pytest.mark.parametrize("module", ["retrieval", "scoring", "calibration", "matcher"])
def test_pipeline_modules_do_not_import_evaluation(module):
    tree = ast.parse((ROOT / "rag" / f"{module}.py").read_text("utf-8"))
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not any(name.startswith("rag.evaluation") for name in imported)
    assert "ground_truth" not in (ROOT / "rag" / f"{module}.py").read_text("utf-8")
