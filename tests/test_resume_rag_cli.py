"""resume_rag.py: the Part A CLI, end to end with FakeEmbedder and temporary directories."""

import json
import os
import shutil
from datetime import date
from pathlib import Path

import pytest

import resume_rag
from scripts.persona_schema import ROOT

AS_OF = "2026-11-01"
INJECTED_TODAY = date(2030, 1, 15)


@pytest.fixture(scope="module")
def corpus_dirs(tmp_path_factory) -> tuple[Path, Path]:
    """A workspace with all 35 resumes, and a CHROMA_DIR indexed once with the fake embedder."""
    base = tmp_path_factory.mktemp("cli")
    workspace = base / "workspace"
    shutil.copytree(ROOT / "data" / "resumes", workspace / "resumes")
    chroma = base / "chroma"
    with pytest.MonkeyPatch.context() as mp:
        _env(mp, workspace, chroma)
        assert (
            resume_rag.main(["index", "--embedder", "fake", "--as-of", AS_OF], load_env=False) == 0
        )
    return workspace, chroma


def _env(mp, workspace: Path, chroma: Path) -> None:
    mp.setenv("WORKSPACE_ROOT", str(workspace))
    mp.setenv("CHROMA_DIR", str(chroma))
    mp.setenv("EMBEDDER", "fake")
    for name in ("AS_OF_DATE", "GEMINI_API_KEY", "GEMINI_EMBEDDING_MODEL", "GEMINI_EMBEDDING_DIM"):
        mp.delenv(name, raising=False)


@pytest.fixture
def env(corpus_dirs, monkeypatch):
    workspace, chroma = corpus_dirs
    _env(monkeypatch, workspace, chroma)
    return workspace, chroma


def cli(capsys, *argv: str) -> tuple[int, str, str]:
    code = resume_rag.main(list(argv), today=lambda: INJECTED_TODAY, load_env=False)
    out, err = capsys.readouterr()
    return code, out, err


@pytest.fixture
def small_env(tmp_path, monkeypatch):
    """A fresh workspace with three resumes and an empty CHROMA_DIR."""
    workspace = tmp_path / "workspace"
    (workspace / "resumes").mkdir(parents=True)
    for name in ("john_doe.pdf", "noah_williams.txt", "elena_rodriguez.docx"):
        shutil.copy(ROOT / "data" / "resumes" / name, workspace / "resumes" / name)
    _env(monkeypatch, workspace, tmp_path / "chroma")
    return workspace


# --- index --------------------------------------------------------------------------------


def test_index_then_unchanged(env, capsys):
    code, out, _ = cli(capsys, "index", "--as-of", AS_OF, "--json")
    assert code == 0
    report = json.loads(out)
    assert report["collection"] == "resumes__section-v1__fake-d64"
    assert report["as_of"] == AS_OF
    assert (
        len(report["unchanged"]) == 35 and report["added"] == [] and report["chunks_written"] == 0
    )
    assert set(report["timings"]) == {"read", "parse_extract", "chunk", "embed", "upsert"}


def test_first_index_adds_everything(small_env, capsys):
    code, out, _ = cli(capsys, "index", "--as-of", AS_OF, "--json")
    assert code == 0 and len(json.loads(out)["added"]) == 3
    code, out, _ = cli(capsys, "index", "--as-of", AS_OF)
    assert code == 0
    assert "Collection: resumes__section-v1__fake-d64" in out
    assert "Timings (seconds)" in out and "unchanged" in out


def test_index_table_output(env, capsys):
    code, out, _ = cli(capsys, "index", "--as-of", AS_OF)
    assert code == 0
    assert "as_of: 2026-11-01" in out
    assert "Files" in out and "chunks written" in out


def test_corrupt_file_exits_1_with_its_code(small_env, capsys):
    (small_env / "resumes" / "broken.pdf").write_bytes(b"%PDF-1.4 nope")
    code, out, _ = cli(capsys, "index", "--as-of", AS_OF)
    assert code == 1
    assert "resumes/broken.pdf" in out and "EXTRACTION_FAILED" in out
    code, out, _ = cli(capsys, "index", "--as-of", AS_OF, "--json")
    assert code == 1
    assert json.loads(out)["failed"][0]["code"] == "EXTRACTION_FAILED"


def test_blank_as_of_uses_injected_today(small_env, capsys, monkeypatch):
    monkeypatch.setenv("AS_OF_DATE", "")
    code, out, _ = cli(capsys, "index", "--json")
    assert code == 0 and json.loads(out)["as_of"] == "2030-01-15"
    monkeypatch.setenv("AS_OF_DATE", "2025-06-30")
    _, out, _ = cli(capsys, "index", "--json")
    assert json.loads(out)["as_of"] == "2025-06-30"  # env beats the default
    _, out, _ = cli(capsys, "index", "--json", "--as-of", AS_OF)
    assert json.loads(out)["as_of"] == AS_OF  # flag beats env


def test_fixed_chunker_gets_its_own_collection(small_env, capsys):
    assert cli(capsys, "index", "--as-of", AS_OF)[0] == 0
    code, out, _ = cli(
        capsys,
        "index",
        "--as-of",
        AS_OF,
        "--chunker",
        "fixed",
        "--chunker-opt",
        "size=80",
        "--json",
    )
    assert code == 0 and json.loads(out)["collection"] == "resumes__fixed-w80-o30__fake-d64"
    code, out, _ = cli(capsys, "stats")
    assert "resumes__section-v1__fake-d64" in out and "resumes__fixed-w80-o30__fake-d64" in out


def test_rebuild_flag(small_env, capsys):
    cli(capsys, "index", "--as-of", AS_OF)
    code, out, _ = cli(capsys, "index", "--as-of", AS_OF, "--rebuild", "--json")
    assert code == 0 and len(json.loads(out)["added"]) == 3


# --- stats, candidates, show --------------------------------------------------------------


def test_stats_lists_the_collection(env, capsys):
    code, out, _ = cli(capsys, "stats")
    assert code == 0
    assert "resumes__section-v1__fake-d64" in out
    assert "228" in out and "35" in out and "2026-11-01" in out
    assert "experience 79" in out


def test_stats_with_nothing_indexed(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "none"))
    code, out, _ = cli(capsys, "stats")
    assert code == 0 and "No collections yet" in out
    assert not (tmp_path / "none").exists()


def test_candidates_lists_35(env, capsys):
    code, out, _ = cli(capsys, "candidates")
    assert code == 0
    assert "Candidates in resumes__section-v1__fake-d64 (35)" in out
    assert "john_doe" in out and "John Doe" in out and "priya_sharma_2" in out


def test_show_john_doe(env, capsys):
    code, out, _ = cli(capsys, "show", "john_doe")
    assert code == 0
    assert "John Doe" in out and "8.9 years" in out
    python = next(line for line in out.splitlines() if " Python " in line)
    django = next(line for line in out.splitlines() if " Django " in line)
    assert "8.9" in python and "experience:0" in python
    assert "6.9" in django  # 2020-01..2023-12 and 2023-01..present merged = 83 months
    assert "Senior Backend Engineer" in out and "Warnings: none" in out


def test_show_json(env, capsys):
    code, out, _ = cli(capsys, "show", "john_doe", "--json")
    assert code == 0
    data = json.loads(out)
    assert data["skill_years"]["Python"] == 8.9 and data["name"] == "John Doe"
    assert "john.doe@example.com" not in out


# --- exit code 2 --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["index", "--as-of", "2026-13-01"], "invalid as-of date"),
        (["index", "--embedder", "word2vec"], "invalid choice"),
        (["index", "--chunker", "semantic"], "invalid choice"),
        (["index", "--chunker-opt", "max_words"], "KEY=VALUE"),
        (["index", "--chunker-opt", "colour=blue"], "bad option for the section chunker"),
        (["index", "--chunker", "fixed", "--chunker-opt", "overlap=500"], "fixed chunker"),
        (["index", "--embedder", "gemini"], "GEMINI_EMBEDDING_MODEL is required"),
        (["index", "--dir", "nowhere"], "does not exist"),
        (["show", "nobody"], "unknown candidate 'nobody'"),
        (["candidates", "--chunker", "fixed"], "no index for collection"),
        ([], "the following arguments are required"),
    ],
)
def test_usage_and_config_errors_exit_2(env, capsys, argv, message):
    code, _, err = cli(capsys, *argv)
    assert code == 2
    assert message in err
    assert err.count("\n") == 1 and "Traceback" not in err  # one line


def test_gemini_without_key_exits_2(env, capsys, monkeypatch):
    monkeypatch.setenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
    code, _, err = cli(capsys, "index", "--embedder", "gemini")
    assert code == 2 and "GEMINI_API_KEY" in err


def test_unknown_embedder_from_env_exits_2(env, capsys, monkeypatch):
    monkeypatch.setenv("EMBEDDER", "word2vec")
    code, _, _ = cli(capsys, "stats")
    assert code == 0  # stats doesn't need an embedder
    code, _, err = cli(capsys, "candidates")
    assert code == 2 and "unknown embedder 'word2vec'" in err


def test_unset_workspace_root_defaults_to_data(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("WORKSPACE_ROOT", raising=False)
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.chdir(ROOT)
    code, _, _ = cli(capsys, "stats")
    assert code == 0 and os.environ["WORKSPACE_ROOT"] == "./data"
