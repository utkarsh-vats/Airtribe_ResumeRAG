"""Text files in the repo: LF line endings and no stray control characters.

Catches the CRLF endings and invisible control characters (a backspace where a regex ``\\b``
was meant) that editing scripts can leave behind. Only tab and newline are allowed.
"""

import re
import subprocess
from pathlib import Path

from scripts.persona_schema import ROOT

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


def tracked_text_files() -> list[Path]:
    """Files git tracks (or would track), else a walk that skips caches and environments."""
    try:
        listed = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
        candidates = [ROOT / name for name in listed]
    except OSError, subprocess.CalledProcessError:
        candidates = [
            path
            for path in ROOT.rglob("*")
            if path.is_file() and not SKIP_DIRS & set(path.relative_to(ROOT).parts)
        ]
    return sorted(
        path
        for path in candidates
        if path.is_file() and (path.suffix in TEXT_SUFFIXES or path.name in TEXT_NAMES)
    )


def problems(path: Path) -> list[str]:
    text = path.read_bytes().decode("utf-8", errors="replace")
    found = []
    for number, line in enumerate(text.split("\n"), 1):
        for match in BAD.finditer(line):
            what = "CR" if match.group() == "\r" else f"control character {match.group()!r}"
            found.append(f"{path.relative_to(ROOT).as_posix()}:{number}: {what}")
    return found


def test_text_files_found():
    files = {p.relative_to(ROOT).as_posix() for p in tracked_text_files()}
    assert {"pyproject.toml", "DESIGN.md", "rag/sections.py", ".gitattributes"} <= files
    assert not any(part.startswith(".venv") for f in files for part in f.split("/"))


def test_no_cr_or_control_characters():
    found = [problem for path in tracked_text_files() for problem in problems(path)]
    assert found == [], "\n".join(found[:50])


def test_detector_reports_file_and_line():
    sample = ROOT / "tests" / "__hygiene_probe__.txt"
    try:
        sample.write_bytes(b"ok\r\nbad \x08 here\n")
        assert problems(sample) == [
            "tests/__hygiene_probe__.txt:1: CR",
            "tests/__hygiene_probe__.txt:2: control character '\\x08'",
        ]
    finally:
        sample.unlink()
