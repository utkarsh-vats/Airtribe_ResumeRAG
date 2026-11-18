from collections.abc import Callable
from pathlib import Path

import pytest
from helpers import assert_err, assert_ok

import fs_tools
from fs_tools import read_file, search_in_file

DATA_KEYS = {
    "path",
    "keyword",
    "whole_word",
    "found",
    "total_matches",
    "returned_matches",
    "truncated",
    "matches",
    "warnings",
}
MATCH_KEYS = {"start", "end", "matched_text", "line_number", "page", "context"}
ELLIPSIS = chr(0x2026)

THREE_LINES = "Python developer\nSkills: python, SQL\nLoves PYTHON scripting\n"


def _write(workspace: Path, name: str, text: str) -> str:
    (workspace / name).write_text(text, encoding="utf-8", newline="")
    return name


def _search(*args: object, **kwargs: object) -> dict:
    data = assert_ok(search_in_file(*args, **kwargs))  # type: ignore[arg-type]
    assert set(data) == DATA_KEYS
    for match in data["matches"]:
        assert set(match) == MATCH_KEYS
    return data


def _assert_offsets_match_read_file(filepath: str, data: dict) -> None:
    content = assert_ok(read_file(filepath))["content"]
    for match in data["matches"]:
        assert content[match["start"] : match["end"]] == match["matched_text"]


# --- Matching -----------------------------------------------------------------------------


@pytest.mark.parametrize("keyword", ["PYTHON", "python", "Python", "pYtHoN"])
def test_case_insensitive_and_preserves_original_casing(workspace: Path, keyword: str) -> None:
    _write(workspace, "cv.txt", THREE_LINES)

    data = _search("cv.txt", keyword)

    assert data["found"] is True
    assert data["keyword"] == keyword
    assert [m["matched_text"] for m in data["matches"]] == ["Python", "python", "PYTHON"]


def test_multiple_matches_with_line_numbers_and_offsets(workspace: Path) -> None:
    _write(workspace, "cv.txt", THREE_LINES)

    data = _search("cv.txt", "python")

    assert data["total_matches"] == data["returned_matches"] == 3
    assert data["truncated"] is False
    assert [m["line_number"] for m in data["matches"]] == [1, 2, 3]
    assert all(m["page"] is None for m in data["matches"])
    _assert_offsets_match_read_file("cv.txt", data)


def test_offsets_are_not_shifted_by_case_folding(workspace: Path) -> None:
    # U+0130 lowercases to two code points, so lower()-based offsets would drift.
    _write(workspace, "cv.txt", f"{chr(0x130) * 3} knows python")

    match = _search("cv.txt", "PYTHON")["matches"][0]

    assert match["start"] == 10
    _assert_offsets_match_read_file("cv.txt", _search("cv.txt", "PYTHON"))


def test_keyword_is_stripped(workspace: Path) -> None:
    _write(workspace, "cv.txt", THREE_LINES)

    data = _search("cv.txt", "  python \n")

    assert data["keyword"] == "python"
    assert data["total_matches"] == 3


def test_no_hits_is_success_not_found(workspace: Path) -> None:
    _write(workspace, "cv.txt", THREE_LINES)

    data = _search("cv.txt", "kubernetes")

    assert data["found"] is False
    assert data["total_matches"] == data["returned_matches"] == 0
    assert data["matches"] == []
    assert data["truncated"] is False


# --- Context ------------------------------------------------------------------------------


def test_context_ellipses_at_start_middle_and_end(workspace: Path) -> None:
    text = "Python at start. " + "lorem ipsum " * 20 + "middle Python here " + "dolor sit " * 20
    text += "ends with Python"
    _write(workspace, "cv.txt", text)

    start, middle, end = (m["context"] for m in _search("cv.txt", "python")["matches"])

    assert start.startswith("Python at start") and start.endswith(ELLIPSIS)
    assert middle.startswith(ELLIPSIS) and middle.endswith(ELLIPSIS)
    assert "middle Python here" in middle
    assert end.startswith(ELLIPSIS) and end.endswith("ends with Python")


def test_context_window_is_about_60_chars_each_side(workspace: Path) -> None:
    _write(workspace, "cv.txt", "a" * 100 + "Python" + "b" * 100)

    context = _search("cv.txt", "python")["matches"][0]["context"]

    assert context == f"{ELLIPSIS}{'a' * 60}Python{'b' * 60}{ELLIPSIS}"


def test_context_collapses_whitespace_and_newlines(workspace: Path) -> None:
    _write(workspace, "cv.txt", "alpha\n\n  beta Python\tgamma\r\ndelta")

    context = _search("cv.txt", "python")["matches"][0]["context"]

    assert context == "alpha beta Python gamma delta"


def test_context_replaces_control_characters(workspace: Path) -> None:
    # DEL (0x7F) is sanitised at extraction; the zero-width space is handled by the context builder.
    _write(workspace, "cv.txt", f"Skills\n {chr(0x7F)} Python,{chr(0x200B)}SQL\f next")

    context = _search("cv.txt", "python")["matches"][0]["context"]

    assert context == "Skills Python, SQL next"


# --- Formats ------------------------------------------------------------------------------


def test_pdf_page_numbers_across_pages(make_pdf: Callable[..., Path]) -> None:
    make_pdf(
        "cv.pdf", [["Intro line", "Python on page one"], ["Python on page two", "More python"]]
    )

    data = _search("cv.pdf", "python")

    assert [m["page"] for m in data["matches"]] == [1, 2, 2]
    assert all("\f" not in m["context"] for m in data["matches"])
    _assert_offsets_match_read_file("cv.pdf", data)


def test_docx_match_inside_table_cell(make_docx: Callable[..., Path]) -> None:
    make_docx("cv.docx", before=["Skills"], table=[["Languages", "Python, Go"]], after=["Done"])

    data = _search("cv.docx", "python")

    assert data["total_matches"] == 1
    match = data["matches"][0]
    assert match["page"] is None
    assert "Languages | Python, Go" in match["context"]
    _assert_offsets_match_read_file("cv.docx", data)


def test_markdown_is_searchable(workspace: Path) -> None:
    _write(workspace, "notes.md", "# Shortlist\n- Jane (Python)\n")

    assert _search("notes.md", "python")["matches"][0]["line_number"] == 2


# --- Literal matching & whole word --------------------------------------------------------


@pytest.mark.parametrize(
    ("keyword", "expected"),
    [("C++", 1), ("node.js", 1), ("(a)", 1), ("a.b", 0), ("[x]", 0), ("*", 0)],
)
def test_regex_metacharacters_are_literal(workspace: Path, keyword: str, expected: int) -> None:
    _write(workspace, "cv.txt", "Skills: C++, node.js, (a) and axb, nodexjs")

    assert _search("cv.txt", keyword)["total_matches"] == expected


def test_whole_word(workspace: Path) -> None:
    _write(workspace, "cv.txt", "JavaScript and Java")

    loose = _search("cv.txt", "java")
    strict = _search("cv.txt", "java", whole_word=True)

    assert [m["matched_text"] for m in loose["matches"]] == ["Java", "Java"]
    assert loose["whole_word"] is False
    assert strict["total_matches"] == 1
    assert strict["matches"][0]["start"] == len("JavaScript and ")
    assert strict["whole_word"] is True


@pytest.mark.parametrize(("keyword", "expected"), [("C++", 1), (".NET", 1), ("go", 1)])
def test_whole_word_with_non_word_edges(workspace: Path, keyword: str, expected: int) -> None:
    _write(workspace, "cv.txt", "I know C++ well, some .NET, and go but not golang")

    assert _search("cv.txt", keyword, whole_word=True)["total_matches"] == expected


# --- Cap ----------------------------------------------------------------------------------


def test_match_cap_truncates_but_counts_all(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fs_tools, "MAX_SEARCH_MATCHES", 3)
    _write(workspace, "cv.txt", "python " * 10)

    data = _search("cv.txt", "python")

    assert data["total_matches"] == 10
    assert data["returned_matches"] == len(data["matches"]) == 3
    assert data["truncated"] is True
    assert [m["start"] for m in data["matches"]] == [0, 7, 14]  # the first matches, in order


def test_default_match_cap_is_50() -> None:
    assert fs_tools.MAX_SEARCH_MATCHES == 50


# --- Argument errors ----------------------------------------------------------------------


@pytest.mark.parametrize("keyword", ["", "   ", "\n\t", None, 5, ["python"], "x" * 201])
def test_invalid_keyword(workspace: Path, keyword: object) -> None:
    _write(workspace, "cv.txt", THREE_LINES)

    assert_err(search_in_file("cv.txt", keyword), "INVALID_ARGUMENT")  # type: ignore[arg-type]


def test_keyword_at_length_limit_is_allowed(workspace: Path) -> None:
    _write(workspace, "cv.txt", "x" * 300)

    assert _search("cv.txt", "x" * 200)["total_matches"] == 1


@pytest.mark.parametrize("whole_word", ["yes", 1, None])
def test_invalid_whole_word(workspace: Path, whole_word: object) -> None:
    _write(workspace, "cv.txt", THREE_LINES)

    assert_err(search_in_file("cv.txt", "python", whole_word), "INVALID_ARGUMENT")  # type: ignore[arg-type]


# --- Error propagation & empty documents --------------------------------------------------


@pytest.mark.parametrize(
    ("filepath", "code"),
    [
        ("missing.txt", "FILE_NOT_FOUND"),
        ("../escape.txt", "PATH_OUTSIDE_WORKSPACE"),
        ("sheet.xlsx", "UNSUPPORTED_FORMAT"),
        ("corrupt.pdf", "EXTRACTION_FAILED"),
        ("folder", "NOT_A_FILE"),
    ],
)
def test_read_errors_match_read_file(workspace: Path, filepath: str, code: str) -> None:
    (workspace / "sheet.xlsx").write_bytes(b"PK\x03\x04")
    (workspace / "corrupt.pdf").write_bytes(b"%PDF-1.7\n" + bytes(range(256)))
    (workspace / "folder").mkdir()

    searched = search_in_file(filepath, "python")

    assert_err(searched, code)
    assert searched == read_file(filepath)


def test_pdf_without_text_is_success_with_warning(make_pdf: Callable[..., Path]) -> None:
    make_pdf("scan.pdf", [[]], shapes_only=True)

    data = _search("scan.pdf", "python")

    assert data["found"] is False
    assert data["matches"] == []
    assert data["warnings"] == assert_ok(read_file("scan.pdf"))["metadata"]["warnings"]
    assert "scanned" in data["warnings"][0]


def test_empty_file_is_success_with_warning(workspace: Path) -> None:
    _write(workspace, "empty.txt", "")

    data = _search("empty.txt", "python")

    assert data["found"] is False
    assert data["warnings"] == ["File is empty."]


def test_path_is_workspace_relative(workspace: Path) -> None:
    (workspace / "resumes").mkdir()
    _write(workspace, "resumes/cv.txt", THREE_LINES)

    assert _search(str(workspace / "resumes" / "cv.txt"), "sql")["path"] == "resumes/cv.txt"


def test_unexpected_exception_is_internal_error(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(workspace, "cv.txt", THREE_LINES)

    def boom(*args: object) -> None:
        raise RuntimeError("sensitive internal detail")

    monkeypatch.setattr(fs_tools, "_keyword_pattern", boom)

    message = assert_err(search_in_file("cv.txt", "python"), "INTERNAL_ERROR")
    assert "sensitive internal detail" not in message
