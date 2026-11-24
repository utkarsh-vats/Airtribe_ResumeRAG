"""job_matcher.py end to end with FakeEmbedder, plus the Python-version guard of both CLIs."""

import ast
import json
import runpy
import shutil
import sys
from datetime import date
from pathlib import Path

import pytest

import job_matcher
import resume_rag
from rag.output import BRIEF_MATCH_KEYS
from scripts.persona_schema import ROOT

AS_OF = "2026-11-01"
PY_JD = "job_descriptions/python_backend.txt"
TEXT_JD = "Backend Engineer\n\nRequirements\n- 5+ years of Python\n- PostgreSQL\n"


def _env(mp, workspace: Path, chroma: Path) -> None:
    mp.setenv("WORKSPACE_ROOT", str(workspace))
    mp.setenv("CHROMA_DIR", str(chroma))
    mp.setenv("EMBEDDER", "fake")
    for name in ("AS_OF_DATE", "GEMINI_API_KEY", "GEMINI_EMBEDDING_MODEL", "GEMINI_EMBEDDING_DIM"):
        mp.delenv(name, raising=False)


def _workspace(base: Path, resumes=None) -> Path:
    workspace = base / "workspace"
    (workspace / "resumes").mkdir(parents=True)
    for path in sorted((ROOT / "data" / "resumes").iterdir()):
        if resumes is None or path.name in resumes:
            shutil.copy(path, workspace / "resumes" / path.name)
    shutil.copytree(ROOT / "data" / "job_descriptions", workspace / "job_descriptions")
    return workspace


@pytest.fixture(scope="module")
def indexed(tmp_path_factory) -> tuple[Path, Path]:
    """All 35 resumes indexed with the fake embedder; no calibration."""
    base = tmp_path_factory.mktemp("matcher_cli")
    workspace, chroma = _workspace(base), base / "chroma"
    with pytest.MonkeyPatch.context() as mp:
        _env(mp, workspace, chroma)
        assert resume_rag.main(["index", "--as-of", AS_OF], load_env=False) == 0
    return workspace, chroma


@pytest.fixture
def env(indexed, monkeypatch):
    _env(monkeypatch, *indexed)
    return indexed


def cli(capsys, *argv: str) -> tuple[int, str, str]:
    code = job_matcher.main(list(argv), today=lambda: date(2026, 11, 1), load_env=False)
    out, err = capsys.readouterr()
    return code, out, err


def match_json(capsys, *argv: str) -> dict:
    code, out, _ = cli(capsys, "match", *argv, "--as-of", AS_OF, "--json")
    assert code == 0
    return json.loads(out)


# --- match --------------------------------------------------------------------------------


def test_match_strict_json(env, capsys):
    out = match_json(capsys, "--jd", PY_JD)
    assert out["filter"]["mode"] == "strict" and out["filter"]["eligible_count"] == 3
    assert {m["candidate_id"] for m in out["top_matches"]} <= {
        "john_doe",
        "ananya_iyer",
        "noah_williams",
    }
    assert all("unmet" not in m for m in out["top_matches"])
    assert out["job"]["source"] == PY_JD
    assert out["config"]["collection"] == "resumes__section-v1__fake-d64"


def test_match_soft_keeps_unmet(env, capsys):
    out = match_json(capsys, "--jd", PY_JD, "--filter", "soft", "-k", "5")
    assert len(out["top_matches"]) == 5 and all("unmet" in m for m in out["top_matches"])
    assert out["filter"]["filtered_count"] == 0


def test_match_jd_text_and_brief_format(env, capsys):
    out = match_json(capsys, "--jd-text", TEXT_JD, "--format", "brief")
    assert list(out) == ["job_description", "top_matches"]
    assert out["job_description"] == TEXT_JD
    assert all(tuple(m) == BRIEF_MATCH_KEYS for m in out["top_matches"])


def test_match_out_creates_directories(env, capsys, tmp_path):
    target = tmp_path / "a" / "b" / "python.json"
    code, out, _ = cli(capsys, "match", "--jd", PY_JD, "--as-of", AS_OF, "--out", str(target))
    assert code == 0 and target.exists()
    assert json.loads(target.read_text("utf-8"))["job"]["title"].startswith("Senior Python")
    assert f"Wrote {target}" in out


def test_match_table_output(env, capsys):
    code, out, _ = cli(capsys, "match", "--jd", PY_JD, "--as-of", AS_OF)
    assert code == 0
    assert "Senior Python Backend Engineer" in out and "Top " in out and "matches" in out
    assert "Filter: strict - 3 of 35 candidates meet every must-have" in out
    assert "Not assessed: Comfortable owning a service" in out
    assert "Meets all 3 must-haves" in out


def test_missing_calibration_is_a_warning(env, capsys):
    code, _, err = cli(capsys, "match", "--jd", PY_JD, "--as-of", AS_OF, "--json")
    assert code == 0 and "warning: the collection has no semantic calibration" in err


def test_as_of_mismatch_warns(env, capsys):
    code, _, err = cli(capsys, "match", "--jd", PY_JD, "--as-of", "2027-01-01", "--json")
    assert code == 0 and "built with as_of 2026-11-01, not 2027-01-01" in err


def test_calibrated_collection_has_no_warning(tmp_path, monkeypatch, capsys):
    workspace = _workspace(tmp_path, {"john_doe.pdf", "noah_williams.txt", "zoe_turner.txt"})
    _env(monkeypatch, workspace, tmp_path / "chroma")
    assert resume_rag.main(["index", "--as-of", AS_OF], load_env=False) == 0
    assert resume_rag.main(["calibrate"], load_env=False) == 0
    capsys.readouterr()
    code, _, err = cli(capsys, "match", "--jd", PY_JD, "--as-of", AS_OF, "--json")
    assert code == 0 and "warning" not in err


# --- batch and parse ----------------------------------------------------------------------


def test_batch_writes_one_file_per_jd(env, capsys, tmp_path):
    out_dir = tmp_path / "results"
    code, out, err = cli(capsys, "batch", "--out-dir", str(out_dir), "--as-of", AS_OF)
    assert code == 0
    files = sorted(p.name for p in out_dir.glob("*.json"))
    assert files == [
        "data_engineer.json",
        "devops_sre.json",
        "frontend.json",
        "java_backend.json",
        "ml_engineer.json",
        "python_backend.json",
    ]
    assert "Batch summary" in out
    for name in files:
        assert name.removesuffix(".json") in out
    assert err.count("no semantic calibration") == 1  # the same warning once
    data = json.loads((out_dir / "devops_sre.json").read_text("utf-8"))
    assert data["job"]["min_total_years"] == 4


def test_batch_brief_soft(env, capsys, tmp_path):
    out_dir = tmp_path / "soft"
    argv = ["batch", "--out-dir", str(out_dir), "--filter", "soft", "--format", "brief"]
    code, _, _ = cli(capsys, *argv, "--as-of", AS_OF, "-k", "3")
    assert code == 0
    data = json.loads((out_dir / "frontend.json").read_text("utf-8"))
    assert list(data) == ["job_description", "top_matches"] and len(data["top_matches"]) == 3


def test_parse(env, capsys):
    code, out, _ = cli(capsys, "parse", "--jd", "job_descriptions/devops_sre.txt")
    data = json.loads(out)
    assert code == 0 and data["source"] == "job_descriptions/devops_sre.txt"
    assert data["min_total_years"] == 4 and data["critical_skills"] == [
        "Kubernetes",
        "Terraform",
        "AWS",
    ]
    code, out, _ = cli(capsys, "parse", "--jd-text", TEXT_JD)
    assert code == 0 and json.loads(out)["must_have"][0]["min_years"] == 5


# --- exit code 2 --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["match", "--jd", PY_JD, "--chunker", "fixed"], "no index for collection"),
        (["match", "--jd", "job_descriptions/missing.txt"], "cannot read the job description"),
        (["match", "--jd", "../outside.txt"], "cannot read the job description"),
        (["match", "--jd-text", "   "], "empty"),
        (["match", "--jd", PY_JD, "-k", "0"], "-k must be at least 1"),
        (["match", "--jd", PY_JD, "--filter", "loose"], "invalid choice"),
        (["match", "--jd", PY_JD, "--as-of", "2026-02-30"], "invalid as-of date"),
        (["match"], "one of the arguments --jd --jd-text is required"),
        (["match", "--jd", PY_JD, "--embedder", "gemini"], "GEMINI_EMBEDDING_MODEL is required"),
        (["batch", "--jd-dir", "nowhere", "--out-dir", "{tmp}"], "does not exist"),
        (["parse", "--jd", "job_descriptions/missing.txt"], "cannot read the job description"),
        ([], "the following arguments are required"),
    ],
)
def test_exit_2(env, capsys, tmp_path, argv, message):
    code, _, err = cli(capsys, *(a.replace("{tmp}", str(tmp_path)) for a in argv))
    assert code == 2
    assert message in err
    assert err.startswith("job_matcher.py: error: ") and "Traceback" not in err


def test_batch_folder_without_jds_exits_2(env, capsys, tmp_path):
    (env[0] / "empty").mkdir(exist_ok=True)
    code, _, err = cli(capsys, "batch", "--jd-dir", "empty", "--out-dir", str(tmp_path))
    assert code == 2 and "no .txt, .pdf or .docx job descriptions in 'empty'" in err


def test_batch_without_gemini_config_exits_2(env, capsys, tmp_path):
    code, _, err = cli(capsys, "batch", "--embedder", "gemini", "--out-dir", str(tmp_path))
    assert code == 2 and "GEMINI_EMBEDDING_MODEL is required" in err


# --- Python-version guard -----------------------------------------------------------------


@pytest.mark.parametrize("script", ["resume_rag.py", "job_matcher.py"])
def test_version_guard(script, monkeypatch, capsys):
    monkeypatch.setattr(sys, "version_info", (3, 11, 9, "final", 0))
    monkeypatch.setattr(sys, "argv", [script, "--help"])
    with pytest.raises(SystemExit) as raised:
        runpy.run_path(str(ROOT / script), run_name="__main__")
    assert raised.value.code == 2
    err = capsys.readouterr().err
    assert err.startswith(f"{script}: error: Python 3.14 or newer is required")
    assert "this is Python 3.11.9" in err and "Activate .venv" in err


@pytest.mark.parametrize("script", ["resume_rag.py", "job_matcher.py"])
def test_guarded_scripts_parse_on_older_pythons(script):
    source = (ROOT / script).read_text("utf-8")
    ast.parse(source, feature_version=(3, 8))
    tree = ast.parse(source)
    first_imports = [n for n in tree.body if isinstance(n, ast.Import | ast.ImportFrom)]
    assert ast.unparse(first_imports[0]) == "import sys"  # nothing else before the guard
    guard = next(n for n in tree.body if isinstance(n, ast.If))
    assert tree.body.index(guard) < tree.body.index(first_imports[1])


# --- evaluate -----------------------------------------------------------------------------


def test_evaluate_tables_out_and_indexing(env, capsys, tmp_path):
    out_path = tmp_path / "eval" / "report.json"
    argv = ["evaluate", "--repeat", "1", "--modes", "semantic,hybrid", "--measure-indexing"]
    code, out, err = cli(capsys, *argv, "--out", str(out_path))
    assert code == 0
    assert "Ranking before the filter" in out and "strict P@5 is at most 0.6" in out
    assert "Filter: 210/210 pairs agree" in out
    assert "Extraction (35 candidates): name 1.000" in out
    assert "End to end (strict, top 10)" in out
    assert "Query latency, hybrid, ms" in out and "strict p50" in out
    assert "Index latency: 35 files" in out
    assert "unchanged re-run" in out and f"Wrote {out_path}" in out
    assert "no semantic calibration" in err
    report = json.loads(out_path.read_text("utf-8"))
    assert set(report) == {
        "environment",
        "config",
        "calibration",
        "warnings",
        "notes",
        "ranking",
        "filter",
        "extraction",
        "end_to_end",
        "latency",
    }
    assert len(report["ranking"]["rows"]) == 6 * 2 * 2  # JDs x modes x orders
    assert set(report["ranking"]["macro"]) == {
        "semantic/retrieval",
        "semantic/scored",
        "hybrid/retrieval",
        "hybrid/scored",
    }
    assert report["config"]["collection"] == "resumes__section-v1__fake-d64"
    assert report["calibration"]["source"] == "generic"
    assert report["latency"]["index"]["rerun_unchanged"] == 35
    assert set(report["latency"]["query"]["macro"]) == {
        "semantic/strict",
        "semantic/soft",
        "hybrid/strict",
        "hybrid/soft",
    }
    assert report["environment"]["embedder_id"] == "fake-d64"


def test_evaluate_json(env, capsys):
    code, out, _ = cli(capsys, "evaluate", "--repeat", "1", "--modes", "keyword", "--json")
    assert code == 0
    report = json.loads(out)
    assert report["latency"]["index"] is None
    assert report["filter"]["overall"]["accuracy"] == 1.0
    assert report["end_to_end"]["macro"]["recall_s"] <= 1.0


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["evaluate", "--modes", "dense"], "--modes takes a comma-separated subset"),
        (["evaluate", "--modes", ","], "--modes takes a comma-separated subset"),
        (["evaluate", "--repeat", "0"], "--repeat must be at least 1"),
        (["evaluate", "--chunker", "fixed"], "no index for collection"),
    ],
)
def test_evaluate_exit_2(env, capsys, argv, message):
    code, _, err = cli(capsys, *argv)
    assert code == 2 and message in err
