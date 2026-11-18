import os
from pathlib import Path

import pytest
from helpers import assert_err, assert_ok

import fs_tools
from fs_tools import ErrorCode, FsToolError, _display_path, _resolve_path


def _assert_rejected(user_path: str, code: ErrorCode = ErrorCode.PATH_OUTSIDE_WORKSPACE) -> None:
    with pytest.raises(FsToolError) as excinfo:
        _resolve_path(user_path)
    assert excinfo.value.code == code


def test_relative_path_inside_workspace_resolves(workspace: Path) -> None:
    assert _resolve_path("resumes/cv.txt") == (workspace / "resumes" / "cv.txt").resolve()


def test_dotdot_that_stays_inside_is_allowed(workspace: Path) -> None:
    assert _resolve_path("resumes/../cv.txt") == (workspace / "cv.txt").resolve()


@pytest.mark.parametrize(
    "user_path", ["../outside.txt", "resumes/../../outside.txt", "..", "a/b/../../../../etc"]
)
def test_dotdot_escape_is_rejected(workspace: Path, user_path: str) -> None:
    _assert_rejected(user_path)


def test_absolute_path_inside_workspace_is_allowed(workspace: Path) -> None:
    target = workspace / "cv.txt"
    assert _resolve_path(str(target)) == target.resolve()


def test_absolute_path_outside_workspace_is_rejected(workspace: Path, tmp_path: Path) -> None:
    _assert_rejected(str(tmp_path / "outside.txt"))


def test_sibling_directory_sharing_name_prefix_is_rejected(workspace: Path, tmp_path: Path) -> None:
    # A naive string-prefix check would accept ".../workspace_evil" for root ".../workspace".
    _assert_rejected(str(tmp_path / f"{workspace.name}_evil" / "cv.txt"))


def test_symlink_pointing_outside_is_rejected(workspace: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    try:
        os.symlink(outside, workspace / "link", target_is_directory=True)
    except OSError as exc:  # e.g. Windows without Developer Mode / admin rights
        pytest.skip(f"symlinks not supported here: {exc}")

    _assert_rejected("link/secret.txt")
    assert_err(fs_tools.read_file("link/secret.txt"), "PATH_OUTSIDE_WORKSPACE")


@pytest.mark.parametrize("user_path", ["", "   ", None, 42, "cv\x00.txt"])
def test_invalid_path_arguments_are_rejected(workspace: Path, user_path: object) -> None:
    _assert_rejected(user_path, ErrorCode.INVALID_ARGUMENT)  # type: ignore[arg-type]


def test_display_path_is_workspace_relative_with_forward_slashes(workspace: Path) -> None:
    nested = workspace / "resumes" / "2024" / "cv.txt"
    nested.parent.mkdir(parents=True)
    nested.write_text("hello", encoding="utf-8")

    assert _display_path(_resolve_path(str(nested))) == "resumes/2024/cv.txt"
    metadata = assert_ok(fs_tools.read_file(str(nested)))["metadata"]
    assert metadata["path"] == "resumes/2024/cv.txt"


def test_workspace_root_is_read_from_env_on_every_call(
    workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = tmp_path / "other"
    other.mkdir()
    assert _resolve_path("cv.txt") == (workspace / "cv.txt").resolve()

    monkeypatch.setenv("WORKSPACE_ROOT", str(other))
    assert _resolve_path("cv.txt") == (other / "cv.txt").resolve()
    _assert_rejected(str(workspace / "cv.txt"))


def test_default_workspace_root_is_sample_data_in_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("WORKSPACE_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    assert _resolve_path("cv.txt") == (tmp_path / "sample_data" / "cv.txt").resolve()
