"""Text files in the repo: LF line endings and no stray control characters.

Catches the CRLF endings and invisible control characters (a backspace where a regex ``\\b``
was meant) that editing scripts can leave behind. Only tab and newline are allowed.

Files come from ``git ls-files`` (tracked plus untracked-but-not-ignored). If the repo has a
``.git`` and git fails for any reason, the tests fail with git's stderr rather than silently
checking something else. Walking the folder is only for a source export with no ``.git``.
"""

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TEXT_SUFFIXES = {".py", ".md", ".json", ".toml", ".yml", ".yaml", ".txt", ".ipynb"}
TEXT_NAMES = {".gitignore", ".gitattributes", ".env.example"}
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    ".chroma",
    ".cache",
    "hf_cache",
    "node_modules",
}
BAD = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")  # every control character except \t and \n
GIT_LS_FILES = ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"]


def git_files(root: Path) -> list[Path]:
    """``git ls-files`` in ``root``; fails the test with git's own error if it can't run."""
    try:
        result = subprocess.run(GIT_LS_FILES, cwd=root, capture_output=True, text=True)
    except OSError as exc:
        pytest.fail(f"{root} has a .git but git could not be run: {exc}")
    if result.returncode != 0:
        pytest.fail(
            f"`{' '.join(GIT_LS_FILES)}` failed in {root} (exit {result.returncode}):\n"
            f"{result.stderr.strip()}"
        )
    return [root / name for name in result.stdout.split("\0") if name]


def walked_files(root: Path) -> list[Path]:
    return [
        path
        for path in root.rglob("*")
        if path.is_file() and not SKIP_DIRS & set(path.relative_to(root).parts)
    ]


def candidate_files(root: Path = REPO_ROOT) -> tuple[str, list[Path]]:
    """``("git", files)`` when ``root`` is a git checkout, else ``("walk", files)``."""
    if (root / ".git").exists():  # a directory, or a file in worktrees and submodules
        return "git", git_files(root)
    return "walk", walked_files(root)


def text_files(root: Path = REPO_ROOT) -> list[Path]:
    _, files = candidate_files(root)
    return sorted(
        p for p in files if p.is_file() and (p.suffix in TEXT_SUFFIXES or p.name in TEXT_NAMES)
    )


def problems(path: Path, root: Path = REPO_ROOT) -> list[str]:
    text = path.read_bytes().decode("utf-8", errors="replace")
    found = []
    for number, line in enumerate(text.split("\n"), 1):
        for match in BAD.finditer(line):
            what = "CR" if match.group() == "\r" else f"control character {match.group()!r}"
            found.append(f"{path.relative_to(root).as_posix()}:{number}: {what}")
    return found


def test_uses_git_in_a_checkout():
    source, files = candidate_files()
    if (REPO_ROOT / ".git").exists():
        assert source == "git"
    assert files


def test_text_files_found():
    files = {p.relative_to(REPO_ROOT).as_posix() for p in text_files()}
    assert {"pyproject.toml", "DESIGN.md", "rag/sections.py", ".gitattributes"} <= files
    assert not any(part in SKIP_DIRS for f in files for part in f.split("/"))


def test_no_cr_or_control_characters():
    found = [problem for path in text_files() for problem in problems(path)]
    assert found == [], "\n".join(found[:50])


def test_detector_reports_file_and_line(tmp_path: Path):
    sample = tmp_path / "probe.txt"
    sample.write_bytes(b"ok\r\nbad \x08 here\n")
    assert problems(sample, tmp_path) == [
        "probe.txt:1: CR",
        "probe.txt:2: control character '\\x08'",
    ]


def test_walk_only_without_git(tmp_path: Path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "b.py").write_text("y = 2\n", encoding="utf-8")
    source, files = candidate_files(tmp_path)
    assert source == "walk" and [p.name for p in files] == ["a.py"]


def test_git_failure_fails_with_stderr(tmp_path: Path):
    # A .git file pointing nowhere: git errors at once instead of searching parent folders.
    (tmp_path / ".git").write_text("gitdir: does-not-exist\n", encoding="utf-8")
    with pytest.raises(pytest.fail.Exception) as info:
        candidate_files(tmp_path)
    # Check that git's stderr is surfaced, not git's wording (it differs between versions).
    command_line, _, stderr = str(info.value).partition("\n")
    assert command_line.startswith(f"`{' '.join(GIT_LS_FILES)}` failed in ")
    assert re.search(r"\(exit [1-9]\d*\):$", command_line), command_line
    assert stderr.strip(), "git's stderr is missing from the failure message"
    assert any(line.startswith("fatal:") for line in stderr.splitlines()), stderr
