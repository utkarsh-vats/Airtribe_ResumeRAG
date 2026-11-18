from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from helpers import assert_err, assert_ok

import fs_tools
from fs_tools import read_file

# --- Plain text ---------------------------------------------------------------------------


def test_reads_utf8_text_with_full_metadata(workspace: Path) -> None:
    text = "Jane Roe\nSkills: Python, SQL\n"
    (workspace / "cv.txt").write_bytes(text.encode("utf-8"))

    data = assert_ok(read_file("cv.txt"))

    assert data["content"] == text
    meta = data["metadata"]
    assert meta["path"] == "cv.txt"
    assert meta["name"] == "cv.txt"
    assert meta["extension"] == ".txt"
    assert meta["size_bytes"] == len(text.encode("utf-8"))
    assert meta["char_count"] == len(text)
    assert meta["word_count"] == 5
    assert meta["encoding"] == "utf-8"
    assert meta["warnings"] == []
    assert "page_count" not in meta


def test_reads_markdown(workspace: Path) -> None:
    (workspace / "notes.md").write_text("# Shortlist\n- Jane Roe\n", encoding="utf-8")

    data = assert_ok(read_file("notes.md"))

    assert "Shortlist" in data["content"]
    assert data["metadata"]["extension"] == ".md"


def test_strips_utf8_bom(workspace: Path) -> None:
    (workspace / "bom.txt").write_bytes(b"\xef\xbb\xbfJane Roe")

    data = assert_ok(read_file("bom.txt"))

    assert data["content"] == "Jane Roe"
    assert data["metadata"]["encoding"] == "utf-8"


def test_falls_back_to_cp1252(workspace: Path) -> None:
    raw = b"Ren\xe9 Lef\xe8vre \x96 r\xe9sum\xe9\x92s"  # invalid as UTF-8
    (workspace / "cp1252.txt").write_bytes(raw)

    data = assert_ok(read_file("cp1252.txt"))

    assert data["content"] == raw.decode("cp1252")
    assert data["metadata"]["encoding"] == "cp1252"


def test_falls_back_to_latin1_when_cp1252_fails(workspace: Path) -> None:
    raw = b"abc \x81 def \xe9"  # 0x81 is undefined in cp1252 and invalid as UTF-8
    (workspace / "latin1.txt").write_bytes(raw)

    data = assert_ok(read_file("latin1.txt"))

    # latin-1 decodes 0x81 to U+0081, a C1 control character, which is sanitised to a space.
    assert data["content"] == raw.decode("latin-1").replace(chr(0x81), " ")
    assert data["content"].endswith(chr(0xE9))
    assert data["metadata"]["encoding"] == "latin-1"


# --- Sanitising ---------------------------------------------------------------------------


def test_control_characters_become_spaces_one_to_one(workspace: Path) -> None:
    raw = b"a\x7fb\x00c\x1bd\x0be" + chr(0x85).encode("utf-8") + b"f"
    (workspace / "ctrl.txt").write_bytes(raw)

    data = assert_ok(read_file("ctrl.txt"))

    assert data["content"] == "a b c d e f"
    assert data["metadata"]["char_count"] == len("a b c d e f")


def test_line_endings_are_normalised_to_lf(workspace: Path) -> None:
    (workspace / "crlf.txt").write_bytes(b"one\r\ntwo\rthree\nfour\r\n\r\nfive")

    data = assert_ok(read_file("crlf.txt"))

    assert data["content"] == "one\ntwo\nthree\nfour\n\nfive"


def test_tab_newline_and_form_feed_survive(workspace: Path) -> None:
    (workspace / "keep.txt").write_bytes(b"a\tb\nc\fd")

    assert assert_ok(read_file("keep.txt"))["content"] == "a\tb\nc\fd"


def test_pdf_bullet_glyph_is_not_extracted_as_del(make_pdf: Callable[..., Path]) -> None:
    # pypdf extracts reportlab's standard-font bullet as DEL (0x7F).
    make_pdf("bullets.pdf", [[f"{chr(0x2022)} Python", f"{chr(0x2022)} SQL"]])

    content = assert_ok(read_file("bullets.pdf"))["content"]

    assert chr(0x7F) not in content
    assert "Python" in content and "SQL" in content


def test_docx_control_characters_are_sanitised(make_docx: Callable[..., Path]) -> None:
    make_docx("ctrl.docx", before=["Tab\there", "Line\rbreak"])

    content = assert_ok(read_file("ctrl.docx"))["content"]

    assert "\r" not in content
    assert "Tab\there" in content


def test_empty_file_is_success_with_warning(workspace: Path) -> None:
    (workspace / "empty.txt").write_bytes(b"")

    data = assert_ok(read_file("empty.txt"))

    assert data["content"] == ""
    assert data["metadata"]["char_count"] == 0
    assert data["metadata"]["word_count"] == 0
    assert data["metadata"]["warnings"] == ["File is empty."]


# --- PDF ----------------------------------------------------------------------------------


def test_reads_pdf_text(make_pdf: Callable[..., Path]) -> None:
    make_pdf("cv.pdf", [["Jane Roe", "Senior Kubernetes Engineer"]])

    data = assert_ok(read_file("cv.pdf"))

    assert "Jane Roe" in data["content"]
    assert "Senior Kubernetes Engineer" in data["content"]
    assert data["metadata"]["page_count"] == 1
    assert data["metadata"]["warnings"] == []
    assert "encoding" not in data["metadata"]


def test_multi_page_pdf_reports_page_count_and_separates_pages(
    make_pdf: Callable[..., Path],
) -> None:
    make_pdf("long.pdf", [["Page one text"], ["Page two text"], ["Page three text"]])

    data = assert_ok(read_file("long.pdf"))

    assert data["metadata"]["page_count"] == 3
    pages = data["content"].split(fs_tools.PDF_PAGE_SEPARATOR)
    assert [p.strip() for p in pages] == ["Page one text", "Page two text", "Page three text"]


def test_pdf_without_text_layer_is_success_with_warning(make_pdf: Callable[..., Path]) -> None:
    make_pdf("scan.pdf", [[]], shapes_only=True)

    data = assert_ok(read_file("scan.pdf"))

    assert data["content"].strip() == ""
    assert data["metadata"]["page_count"] == 1
    assert len(data["metadata"]["warnings"]) == 1
    assert "scanned" in data["metadata"]["warnings"][0]


def test_corrupt_pdf_is_extraction_failed(workspace: Path) -> None:
    garbage = bytes((i * 37 + 11) % 256 for i in range(512))
    (workspace / "corrupt.pdf").write_bytes(b"%PDF-1.7\n" + garbage)

    message = assert_err(read_file("corrupt.pdf"), "EXTRACTION_FAILED")

    assert str(workspace) not in message


# --- DOCX ---------------------------------------------------------------------------------


def test_reads_docx_paragraphs_and_tables_in_document_order(
    make_docx: Callable[..., Path],
) -> None:
    make_docx(
        "cv.docx",
        before=["Jane Roe", "Skills"],
        table=[["Languages", "Java, Kotlin"], ["Cloud", "Kubernetes"]],
        after=["Experience at Example Corp"],
    )

    data = assert_ok(read_file("cv.docx"))

    content = data["content"]
    assert "Languages | Java, Kotlin" in content
    assert "Cloud | Kubernetes" in content
    order = [content.index(s) for s in ("Skills", "Languages |", "Experience at Example Corp")]
    assert order == sorted(order)
    assert data["metadata"]["extension"] == ".docx"
    assert "page_count" not in data["metadata"]


def test_empty_docx_is_success_with_warning(make_docx: Callable[..., Path]) -> None:
    make_docx("blank.docx")

    data = assert_ok(read_file("blank.docx"))

    assert data["content"].strip() == ""
    assert data["metadata"]["warnings"] == ["Document contains no text."]


def test_corrupt_docx_is_extraction_failed(workspace: Path) -> None:
    (workspace / "corrupt.docx").write_bytes(b"PK\x03\x04 not really a zip")

    message = assert_err(read_file("corrupt.docx"), "EXTRACTION_FAILED")

    assert str(workspace) not in message


# --- Extensions ---------------------------------------------------------------------------


def test_uppercase_extensions_are_supported(workspace: Path, make_pdf: Callable[..., Path]) -> None:
    (workspace / "CV.TXT").write_text("upper text", encoding="utf-8")
    make_pdf("CV.PDF", [["upper pdf"]])

    assert assert_ok(read_file("CV.TXT"))["content"] == "upper text"
    pdf = assert_ok(read_file("CV.PDF"))
    assert "upper pdf" in pdf["content"]
    assert pdf["metadata"]["extension"] == ".pdf"
    assert pdf["metadata"]["name"] == "CV.PDF"


@pytest.mark.parametrize("name", ["sheet.xlsx", "photo.png", "noextension"])
def test_unsupported_extension(workspace: Path, name: str) -> None:
    (workspace / name).write_bytes(b"\x00\x01\x02")

    assert_err(read_file(name), "UNSUPPORTED_FORMAT")


# --- Path / filesystem errors ---------------------------------------------------------------


def test_missing_file(workspace: Path) -> None:
    message = assert_err(read_file("nope.txt"), "FILE_NOT_FOUND")

    assert "nope.txt" in message
    assert str(workspace) not in message


def test_directory_is_not_a_file(workspace: Path) -> None:
    (workspace / "resumes").mkdir()

    assert_err(read_file("resumes"), "NOT_A_FILE")


def test_path_outside_workspace(workspace: Path) -> None:
    assert_err(read_file("../secret.txt"), "PATH_OUTSIDE_WORKSPACE")


def test_invalid_argument(workspace: Path) -> None:
    assert_err(read_file(None), "INVALID_ARGUMENT")  # type: ignore[arg-type]


def test_file_over_size_cap_is_rejected(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fs_tools, "MAX_READ_BYTES", 10)
    (workspace / "at_cap.txt").write_bytes(b"x" * 10)
    (workspace / "over_cap.txt").write_bytes(b"x" * 11)

    assert assert_ok(read_file("at_cap.txt"))["metadata"]["size_bytes"] == 10
    assert_err(read_file("over_cap.txt"), "FILE_TOO_LARGE")


def test_default_size_cap_is_10_mb() -> None:
    assert fs_tools.MAX_READ_BYTES == 10 * 1024 * 1024


@pytest.mark.parametrize(
    ("exc", "code"),
    [(PermissionError("denied"), "PERMISSION_DENIED"), (OSError("disk"), "IO_ERROR")],
)
def test_os_errors_map_to_codes(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, exc: OSError, code: str
) -> None:
    (workspace / "cv.txt").write_text("hello", encoding="utf-8")

    def fail(path: Path) -> tuple[str, dict]:
        raise exc

    monkeypatch.setattr(fs_tools, "_extract_text", fail)

    assert_err(read_file("cv.txt"), code)


def test_unexpected_exception_is_internal_error_with_generic_message(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (workspace / "cv.txt").write_text("hello", encoding="utf-8")

    def boom(path: Path) -> tuple[str, dict]:
        raise RuntimeError("sensitive internal detail")

    monkeypatch.setattr(fs_tools, "_extract_text", boom)

    message = assert_err(read_file("cv.txt"), "INTERNAL_ERROR")
    assert "sensitive internal detail" not in message


# --- Metadata -----------------------------------------------------------------------------


def test_modified_is_iso8601_utc(workspace: Path) -> None:
    (workspace / "cv.txt").write_text("hello", encoding="utf-8")

    modified = assert_ok(read_file("cv.txt"))["metadata"]["modified"]

    parsed = datetime.fromisoformat(modified)
    assert parsed.utcoffset() == timedelta(0)
