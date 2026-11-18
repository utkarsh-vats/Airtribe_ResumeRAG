import os
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import fs_tools
from fs_tools import FsToolError, list_files

ENTRY_KEYS = {"name", "path", "extension", "size_bytes", "modified", "supported"}


@pytest.fixture
def populated(workspace: Path) -> Path:
    for name in ("b.txt", "A.pdf", "D.PDF", "c.docx", "notes.md", "sheet.xlsx", ".gitkeep"):
        (workspace / name).write_bytes(b"x" * len(name))
    (workspace / "sub").mkdir()
    (workspace / "sub" / "inner.txt").write_text("inner", encoding="utf-8")
    return workspace


def _names(entries: list[dict]) -> list[str]:
    return [entry["name"] for entry in entries]


def _assert_raises(code: str, workspace: Path, *args: object) -> None:
    with pytest.raises(FsToolError) as excinfo:
        list_files(*args)  # type: ignore[arg-type]
    assert excinfo.value.code == code
    assert str(workspace) not in excinfo.value.message
    assert "Traceback" not in excinfo.value.message


# --- Listing ------------------------------------------------------------------------------


def test_lists_files_with_metadata(populated: Path) -> None:
    entries = list_files("")

    assert all(set(entry) == ENTRY_KEYS for entry in entries)
    entry = next(e for e in entries if e["name"] == "b.txt")
    assert entry["path"] == "b.txt"
    assert entry["extension"] == ".txt"
    assert entry["size_bytes"] == len("b.txt")
    assert entry["supported"] is True


def test_sorted_case_insensitively_excluding_subdirs_and_dotfiles(populated: Path) -> None:
    assert _names(list_files("")) == ["A.pdf", "b.txt", "c.docx", "D.PDF", "notes.md", "sheet.xlsx"]


@pytest.mark.parametrize("directory", ["", ".", "./"])
def test_root_spellings_list_the_workspace_root(populated: Path, directory: str) -> None:
    assert "b.txt" in _names(list_files(directory))


def test_nested_directory_returns_workspace_relative_posix_paths(populated: Path) -> None:
    entries = list_files("sub")

    assert _names(entries) == ["inner.txt"]
    assert entries[0]["path"] == "sub/inner.txt"


def test_absolute_directory_inside_workspace(populated: Path) -> None:
    assert _names(list_files(str(populated / "sub"))) == ["inner.txt"]


def test_empty_directory_returns_empty_list(workspace: Path) -> None:
    (workspace / "empty").mkdir()

    assert list_files("empty") == []


def test_supported_flag(populated: Path) -> None:
    supported = {e["name"]: e["supported"] for e in list_files("")}

    assert supported == {
        "A.pdf": True,
        "b.txt": True,
        "c.docx": True,
        "D.PDF": True,
        "notes.md": True,
        "sheet.xlsx": False,
    }


def test_extension_is_lowercased(populated: Path) -> None:
    entry = next(e for e in list_files("") if e["name"] == "D.PDF")

    assert entry["extension"] == ".pdf"


def test_modified_is_iso8601_utc(populated: Path) -> None:
    for entry in list_files(""):
        assert datetime.fromisoformat(entry["modified"]).utcoffset() == timedelta(0)


# --- Extension filter -----------------------------------------------------------------------


@pytest.mark.parametrize("extension", ["pdf", ".pdf", ".PDF", "*.pdf", "PDF", " pdf "])
def test_extension_filter_spellings(populated: Path, extension: str) -> None:
    assert _names(list_files("", extension)) == ["A.pdf", "D.PDF"]


@pytest.mark.parametrize("extension", [None, "", "   "])
def test_missing_or_blank_extension_means_no_filter(populated: Path, extension: str | None) -> None:
    assert len(list_files("", extension)) == 6


def test_filter_with_no_matches_returns_empty_list(populated: Path) -> None:
    assert list_files("", "csv") == []


@pytest.mark.parametrize("extension", ["p df", "docs/pdf", "..", ".", "*", "*.*", ".tar.gz", 5])
def test_invalid_extension_filter(populated: Path, extension: object) -> None:
    _assert_raises("INVALID_ARGUMENT", populated, "", extension)


# --- Errors -------------------------------------------------------------------------------


def test_missing_directory(workspace: Path) -> None:
    _assert_raises("NOT_FOUND", workspace, "nope")


def test_missing_workspace_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path / "missing"))

    _assert_raises("NOT_FOUND", tmp_path, "")


def test_file_is_not_a_directory(populated: Path) -> None:
    _assert_raises("NOT_A_DIRECTORY", populated, "b.txt")


@pytest.mark.parametrize("directory", ["..", "../", "sub/../.."])
def test_dotdot_escape(populated: Path, directory: str) -> None:
    _assert_raises("PATH_OUTSIDE_WORKSPACE", populated, directory)


def test_absolute_path_outside(populated: Path, tmp_path: Path) -> None:
    _assert_raises("PATH_OUTSIDE_WORKSPACE", populated, str(tmp_path))


@pytest.mark.parametrize("directory", [None, 7, "a\x00b"])
def test_invalid_directory_argument(workspace: Path, directory: object) -> None:
    _assert_raises("INVALID_ARGUMENT", workspace, directory)


def test_permission_denied(populated: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def deny(self: Path):
        raise PermissionError(13, "Permission denied", str(self))

    monkeypatch.setattr(Path, "iterdir", deny)

    _assert_raises("PERMISSION_DENIED", populated, "")


# --- Symlinks -----------------------------------------------------------------------------


def test_symlink_resolving_outside_is_skipped(workspace: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    (workspace / "real.txt").write_text("real", encoding="utf-8")
    try:
        os.symlink(outside / "secret.txt", workspace / "leak.txt")
        os.symlink(workspace / "real.txt", workspace / "alias.txt")
    except OSError as exc:  # e.g. Windows without Developer Mode / admin rights
        pytest.skip(f"symlinks not supported here: {exc}")

    entries = list_files("")

    assert _names(entries) == ["alias.txt", "real.txt"]
    assert all("secret" not in str(entry) for entry in entries)
    assert fs_tools.read_file("leak.txt")["error"]["code"] == "PATH_OUTSIDE_WORKSPACE"
