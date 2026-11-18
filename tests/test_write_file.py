import os
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from helpers import assert_err, assert_ok

import fs_tools
from fs_tools import read_file, write_file

DATA_KEYS = {"path", "name", "size_bytes", "overwritten", "created_dirs", "modified"}

# e-acute, right single quotation mark and an emoji, kept as code points so the source is ASCII.
UNICODE_TEXT = f"Caf{chr(0xE9)} r{chr(0x2019)}sum{chr(0xE9)} {chr(0x1F600)}\nsecond line\n\nend\n"


# --- Success paths ------------------------------------------------------------------------


def test_creates_new_file_in_root(workspace: Path) -> None:
    data = assert_ok(write_file("notes.md", "# Shortlist\n"))

    assert set(data) == DATA_KEYS
    assert data["path"] == "notes.md"
    assert data["name"] == "notes.md"
    assert data["size_bytes"] == len(b"# Shortlist\n")
    assert data["overwritten"] is False
    assert data["created_dirs"] == []
    assert (workspace / "notes.md").read_bytes() == b"# Shortlist\n"


def test_nested_path_creates_parent_dirs(workspace: Path) -> None:
    (workspace / "summaries").mkdir()

    data = assert_ok(write_file("summaries/2024/q1/jane.txt", "hello"))

    assert data["path"] == "summaries/2024/q1/jane.txt"
    assert data["created_dirs"] == ["summaries/2024", "summaries/2024/q1"]
    assert (workspace / "summaries" / "2024" / "q1" / "jane.txt").read_text("utf-8") == "hello"


def test_overwrite_replaces_content(workspace: Path) -> None:
    target = workspace / "notes.txt"
    target.write_text("old content that is longer", encoding="utf-8")

    data = assert_ok(write_file("notes.txt", "new"))

    assert data["overwritten"] is True
    assert data["size_bytes"] == 3
    assert target.read_bytes() == b"new"


def test_unicode_round_trip_keeps_lf_line_endings(workspace: Path) -> None:
    data = assert_ok(write_file("unicode.md", UNICODE_TEXT))

    raw = (workspace / "unicode.md").read_bytes()
    assert raw == UNICODE_TEXT.encode("utf-8")
    assert b"\r\n" not in raw
    assert data["size_bytes"] == len(raw)
    assert assert_ok(read_file("unicode.md"))["content"] == UNICODE_TEXT


def test_empty_content_creates_zero_byte_file(workspace: Path) -> None:
    data = assert_ok(write_file("empty.txt", ""))

    assert data["size_bytes"] == 0
    assert (workspace / "empty.txt").read_bytes() == b""


def test_uppercase_extension_is_accepted(workspace: Path) -> None:
    assert assert_ok(write_file("NOTES.MD", "x"))["name"] == "NOTES.MD"


def test_absolute_path_inside_workspace(workspace: Path) -> None:
    assert assert_ok(write_file(str(workspace / "abs.txt"), "x"))["path"] == "abs.txt"


def test_no_temp_files_remain_after_success(workspace: Path) -> None:
    (workspace / "out").mkdir()

    assert_ok(write_file("out/report.md", "first"))
    assert_ok(write_file("out/report.md", "second"))

    assert os.listdir(workspace / "out") == ["report.md"]


def test_written_file_is_listed_without_temp_files(workspace: Path) -> None:
    assert_ok(write_file("summaries/jane.md", "x"))

    assert [e["name"] for e in fs_tools.list_files("summaries")] == ["jane.md"]


def test_modified_is_iso8601_utc(workspace: Path) -> None:
    modified = assert_ok(write_file("notes.txt", "x"))["modified"]

    assert datetime.fromisoformat(modified).utcoffset() == timedelta(0)


# --- Argument errors ----------------------------------------------------------------------


@pytest.mark.parametrize("name", ["cv.pdf", "cv.docx", "script.py", "noextension"])
def test_unsupported_extension(workspace: Path, name: str) -> None:
    assert_err(write_file(name, "x"), "UNSUPPORTED_FORMAT")

    assert not (workspace / name).exists()


@pytest.mark.parametrize("content", [None, 123, b"bytes", ["list"]])
def test_non_string_content(workspace: Path, content: object) -> None:
    assert_err(write_file("notes.txt", content), "INVALID_ARGUMENT")  # type: ignore[arg-type]

    assert not (workspace / "notes.txt").exists()


def test_unencodable_content(workspace: Path) -> None:
    assert_err(write_file("notes.txt", f"bad {chr(0xD800)} surrogate"), "INVALID_ARGUMENT")


def test_content_over_cap(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fs_tools, "MAX_WRITE_BYTES", 10)

    assert_ok(write_file("at_cap.txt", "x" * 10))
    assert_err(write_file("over_cap.txt", "x" * 11), "CONTENT_TOO_LARGE")
    # The cap applies to encoded bytes: 6 characters, 12 bytes.
    assert_err(write_file("multibyte.txt", chr(0xE9) * 6), "CONTENT_TOO_LARGE")
    assert not (workspace / "over_cap.txt").exists()


def test_default_content_cap_is_1_mb() -> None:
    assert fs_tools.MAX_WRITE_BYTES == 1024 * 1024


@pytest.mark.parametrize("filepath", ["../escape.txt", "sub/../../escape.txt"])
def test_dotdot_escape(workspace: Path, tmp_path: Path, filepath: str) -> None:
    assert_err(write_file(filepath, "x"), "PATH_OUTSIDE_WORKSPACE")

    assert not (tmp_path / "escape.txt").exists()


def test_absolute_path_outside(workspace: Path, tmp_path: Path) -> None:
    target = tmp_path / "outside" / "escape.txt"

    assert_err(write_file(str(target), "x"), "PATH_OUTSIDE_WORKSPACE")

    assert not target.parent.exists()  # no directories created before the sandbox check


def test_invalid_path(workspace: Path) -> None:
    assert_err(write_file("", "x"), "INVALID_ARGUMENT")


# --- Filesystem errors --------------------------------------------------------------------


def test_target_is_a_directory(workspace: Path) -> None:
    (workspace / "folder.md").mkdir()

    assert_err(write_file("folder.md", "x"), "NOT_A_FILE")


@pytest.mark.parametrize("filepath", ["cv.txt/notes.md", "cv.txt/deeper/notes.md"])
def test_parent_component_is_a_file(workspace: Path, filepath: str) -> None:
    (workspace / "cv.txt").write_text("I am a file", encoding="utf-8")

    message = assert_err(write_file(filepath, "x"), "NOT_A_DIRECTORY")

    assert "cv.txt" in message
    assert str(workspace) not in message
    assert (workspace / "cv.txt").read_text("utf-8") == "I am a file"


def test_missing_workspace_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path / "missing"))

    assert_err(write_file("notes.txt", "x"), "NOT_FOUND")

    assert not (tmp_path / "missing").exists()


def test_failed_replace_leaves_target_untouched_and_no_temp_files(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (workspace / "out").mkdir()
    target = workspace / "out" / "report.md"
    target.write_text("original", encoding="utf-8")
    temp_names = []

    def failing_replace(src: str, dst: str) -> None:
        temp_names.append(Path(src).name)
        raise OSError(5, "simulated failure")

    monkeypatch.setattr(fs_tools.os, "replace", failing_replace)

    message = assert_err(write_file("out/report.md", "replacement"), "IO_ERROR")

    assert target.read_text("utf-8") == "original"
    assert os.listdir(workspace / "out") == ["report.md"]
    assert len(temp_names) == 1 and temp_names[0].startswith(".")  # hidden from list_files
    assert str(workspace) not in message


@pytest.mark.parametrize(
    ("exc", "code"),
    [(PermissionError(13, "denied"), "PERMISSION_DENIED"), (OSError(5, "disk"), "IO_ERROR")],
)
def test_os_errors_map_to_codes(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, exc: OSError, code: str
) -> None:
    def fail(path: Path, content: str) -> None:
        raise exc

    monkeypatch.setattr(fs_tools, "_atomic_write", fail)

    assert_err(write_file("notes.txt", "x"), code)


def test_unexpected_exception_is_internal_error(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(path: Path, content: str) -> None:
        raise RuntimeError("sensitive internal detail")

    monkeypatch.setattr(fs_tools, "_atomic_write", boom)

    message = assert_err(write_file("notes.txt", "x"), "INTERNAL_ERROR")
    assert "sensitive internal detail" not in message
