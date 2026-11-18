"""Part A: sandboxed file-system tools for resume files (PDF, TXT, DOCX).

Exposes four tools intended to be called by an LLM via function calling:

- ``read_file(filepath) -> dict``
- ``list_files(directory, extension=None) -> list``
- ``write_file(filepath, content) -> dict``
- ``search_in_file(filepath, keyword, whole_word=False) -> dict``

All paths are resolved inside ``WORKSPACE_ROOT``. See DESIGN.md for the
result envelope, error codes and sandbox rules.
"""

import contextlib
import logging
import os
import re
import tempfile
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import docx
from docx.table import Table
from pypdf import PdfReader

logger = logging.getLogger(__name__)

DEFAULT_WORKSPACE_ROOT = "./sample_data"
MAX_READ_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_WRITE_BYTES = 1024 * 1024  # 1 MB of UTF-8 encoded content
MAX_SEARCH_MATCHES = 50  # matches returned; all are still counted
MAX_KEYWORD_LENGTH = 200
SEARCH_CONTEXT_CHARS = 60
ELLIPSIS = "\N{HORIZONTAL ELLIPSIS}"
TEXT_EXTENSIONS = frozenset({".txt", ".md"})
READ_EXTENSIONS = TEXT_EXTENSIONS | {".pdf", ".docx"}
TEXT_ENCODINGS = ("utf-8", "cp1252", "latin-1")  # latin-1 decodes any byte sequence
PDF_PAGE_SEPARATOR = "\n\f\n"  # form feed between pages, as pdftotext does
_PLAIN_EXTENSION = re.compile(r"\.[a-z0-9_-]+")
# Unicode Cc control characters except \t (0x09), \n (0x0A) and \f (0x0C, the PDF page separator).
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0d-\x1f\x7f-\x9f]")


class ErrorCode(StrEnum):
    """Stable error codes returned in error envelopes; DESIGN.md section 2 describes each one.

    A ``StrEnum``, so members compare equal to their plain string values.
    """

    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    PATH_OUTSIDE_WORKSPACE = "PATH_OUTSIDE_WORKSPACE"
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    NOT_FOUND = "NOT_FOUND"
    NOT_A_FILE = "NOT_A_FILE"
    NOT_A_DIRECTORY = "NOT_A_DIRECTORY"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    CONTENT_TOO_LARGE = "CONTENT_TOO_LARGE"
    EXTRACTION_FAILED = "EXTRACTION_FAILED"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    IO_ERROR = "IO_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class FsToolError(Exception):
    """A tool failure with a stable error code and a message that is safe to show the LLM."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        """Store ``code`` and ``message``; ``str(exc)`` is ``"CODE: message"``."""
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


# --- Envelope -----------------------------------------------------------------------------


def _ok(data: Any) -> dict:
    return {"success": True, "data": data, "error": None}


def _err(code: ErrorCode | str, message: str) -> dict:
    return {"success": False, "data": None, "error": {"code": str(code), "message": message}}


# --- Sandbox ------------------------------------------------------------------------------


def _workspace_root() -> Path:
    """Absolute workspace root, read from the environment on every call."""
    return Path(os.environ.get("WORKSPACE_ROOT") or DEFAULT_WORKSPACE_ROOT).resolve()


def _resolve_path(user_path: str) -> Path:
    """Resolve an untrusted path to an absolute path inside the workspace root.

    Symlinks and ``..`` are resolved *before* the containment check, so traversal, absolute
    paths outside the root and symlink escapes are all rejected the same way.
    """
    if not isinstance(user_path, str) or not user_path.strip():
        raise FsToolError(ErrorCode.INVALID_ARGUMENT, "Path must be a non-empty string.")
    if "\x00" in user_path:
        raise FsToolError(ErrorCode.INVALID_ARGUMENT, "Path must not contain NUL bytes.")

    root = _workspace_root()
    candidate = Path(user_path)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise FsToolError(
            ErrorCode.PATH_OUTSIDE_WORKSPACE, f"Path '{user_path}' is outside the workspace."
        )
    return resolved


def _display_path(path: Path) -> str:
    """Workspace-relative path with forward slashes, identical on Windows and Linux."""
    relative = path.relative_to(_workspace_root()).as_posix()
    return relative if relative != "." else ""


def _iso_utc(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=UTC).isoformat()


def _os_error(exc: OSError, display: str) -> FsToolError:
    if isinstance(exc, PermissionError):
        return FsToolError(ErrorCode.PERMISSION_DENIED, f"Permission denied: '{display}'.")
    return FsToolError(ErrorCode.IO_ERROR, f"I/O error accessing '{display}'.")


# --- Text extraction ----------------------------------------------------------------------


def _extract_text_file(path: Path) -> tuple[str, dict]:
    raw = path.read_bytes()
    for encoding in TEXT_ENCODINGS:
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        return text.removeprefix("\ufeff"), {"encoding": encoding}
    raise AssertionError("unreachable: latin-1 decodes any byte sequence")


def _extract_pdf(path: Path) -> tuple[str, dict]:
    # pypdf may only log on open for some corrupt files and raise later, on page access,
    # so construction and extraction share one error mapping.
    try:
        reader = PdfReader(path)
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:
        raise FsToolError(
            ErrorCode.EXTRACTION_FAILED,
            f"Could not extract text from PDF ({type(exc).__name__}); "
            "the file may be corrupt or encrypted.",
        ) from exc
    return PDF_PAGE_SEPARATOR.join(pages), {"page_count": len(pages)}


def _table_lines(table: Table) -> list[str]:
    lines = []
    for row in table.rows:
        cells, seen = [], set()
        for cell in row.cells:  # merged cells repeat; keep each one once
            if id(cell._tc) not in seen:
                seen.add(id(cell._tc))
                cells.append(cell.text.strip())
        lines.append(" | ".join(cells))
    return lines


def _extract_docx(path: Path) -> tuple[str, dict]:
    try:
        document = docx.Document(str(path))
        lines = []
        for block in document.iter_inner_content():  # paragraphs and tables in document order
            if isinstance(block, Table):
                lines.extend(_table_lines(block))
            else:
                lines.append(block.text)
    except Exception as exc:
        raise FsToolError(
            ErrorCode.EXTRACTION_FAILED,
            f"Could not extract text from DOCX ({type(exc).__name__}); the file may be corrupt.",
        ) from exc
    return "\n".join(lines), {}


def _extract_text(path: Path) -> tuple[str, dict]:
    """Return ``(text, format-specific metadata)`` for a supported file."""
    extension = path.suffix.lower()
    if extension in TEXT_EXTENSIONS:
        return _extract_text_file(path)
    if extension == ".pdf":
        return _extract_pdf(path)
    return _extract_docx(path)


def _sanitise_text(text: str) -> str:
    """Normalise line endings to "\\n", then replace control characters 1:1 with spaces.

    Runs before any offsets are computed. Keeps "\\n", "\\t" and "\\f"; removes e.g. the DEL
    pypdf extracts for some bullet glyphs.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return _CONTROL_CHARS.sub(" ", text)


def _empty_text_warning(extension: str) -> str:
    if extension == ".pdf":
        return "No extractable text - possibly a scanned/image PDF."
    if extension == ".docx":
        return "Document contains no text."
    return "File is empty."


def _load_document(filepath: str) -> tuple[str, dict]:
    """Resolve, validate and extract a readable file; return ``(content, metadata)``.

    Shared by ``read_file`` and ``search_in_file`` so both fail identically. Raises
    ``FsToolError``.
    """
    path = _resolve_path(filepath)
    display = _display_path(path)
    extension = path.suffix.lower()
    try:
        if not path.exists():
            raise FsToolError(ErrorCode.FILE_NOT_FOUND, f"File '{display}' does not exist.")
        if not path.is_file():
            raise FsToolError(ErrorCode.NOT_A_FILE, f"'{display}' is not a regular file.")
        if extension not in READ_EXTENSIONS:
            supported = ", ".join(sorted(READ_EXTENSIONS))
            raise FsToolError(
                ErrorCode.UNSUPPORTED_FORMAT,
                f"Unsupported file type '{path.suffix}'. Supported: {supported}.",
            )
        stat = path.stat()
        if stat.st_size > MAX_READ_BYTES:
            raise FsToolError(
                ErrorCode.FILE_TOO_LARGE,
                f"File '{display}' is {stat.st_size} bytes; the limit is {MAX_READ_BYTES} bytes.",
            )
        content, format_metadata = _extract_text(path)
    except OSError as exc:
        raise _os_error(exc, display) from exc

    content = _sanitise_text(content)
    warnings = [] if content.strip() else [_empty_text_warning(extension)]
    metadata = {
        "path": display,
        "name": path.name,
        "extension": extension,
        "size_bytes": stat.st_size,
        "modified": _iso_utc(stat.st_mtime),
        "char_count": len(content),
        "word_count": len(content.split()),
        **format_metadata,
        "warnings": warnings,
    }
    return content, metadata


# --- Tools --------------------------------------------------------------------------------


def read_file(filepath: str) -> dict:
    """Read a resume (.txt, .md, .pdf, .docx) and return its text plus metadata.

    Never raises: every failure is returned as an error envelope.
    """
    try:
        content, metadata = _load_document(filepath)
        return _ok({"content": content, "metadata": metadata})
    except FsToolError as exc:
        return _err(exc.code, exc.message)
    except Exception:
        logger.exception("read_file failed unexpectedly")
        return _err(ErrorCode.INTERNAL_ERROR, "Unexpected internal error while reading the file.")


def _normalise_extension(extension: str | None) -> str | None:
    """Turn "pdf", ".PDF" or "*.pdf" into ".pdf"; None or blank means no filter."""
    if extension is None:
        return None
    if not isinstance(extension, str):
        raise FsToolError(ErrorCode.INVALID_ARGUMENT, "Extension must be a string.")
    value = extension.strip().lower()
    if not value:
        return None
    value = value.removeprefix("*")
    if not value.startswith("."):
        value = f".{value}"
    if not _PLAIN_EXTENSION.fullmatch(value):
        raise FsToolError(
            ErrorCode.INVALID_ARGUMENT,
            f"Invalid extension filter '{extension}'; use a plain extension such as 'pdf'.",
        )
    return value


def list_files(directory: str, extension: str | None = None) -> list:
    """List the files (non-recursive) in a workspace directory, optionally filtered by extension.

    ``""`` and ``"."`` mean the workspace root. Dotfiles and symlinks that resolve outside the
    workspace are skipped. Raises ``FsToolError`` on failure; the tool dispatcher wraps it.
    """
    root = _workspace_root()
    path = root if directory in ("", ".") else _resolve_path(directory)
    wanted = _normalise_extension(extension)
    display = _display_path(path) or "."

    try:
        if not path.exists():
            raise FsToolError(ErrorCode.NOT_FOUND, f"Directory '{display}' does not exist.")
        if not path.is_dir():
            raise FsToolError(ErrorCode.NOT_A_DIRECTORY, f"'{display}' is not a directory.")
        entries = list(path.iterdir())
    except OSError as exc:
        raise _os_error(exc, display) from exc

    files = []
    for entry in entries:
        if entry.name.startswith("."):
            continue
        suffix = entry.suffix.lower()
        if wanted is not None and suffix != wanted:
            continue
        try:
            target = entry.resolve(strict=False)
            if not target.is_relative_to(root) or not target.is_file():
                continue  # symlink escape (never leak its metadata) or not a regular file
            stat = target.stat()
        except OSError:
            logger.debug("skipping unreadable entry %s", entry.name)
            continue
        files.append(
            {
                "name": entry.name,
                "path": _display_path(entry),
                "extension": suffix,
                "size_bytes": stat.st_size,
                "modified": _iso_utc(stat.st_mtime),
                "supported": suffix in READ_EXTENSIONS,
            }
        )
    files.sort(key=lambda f: (f["name"].casefold(), f["name"]))
    return files


def _validate_content(content: str) -> None:
    if not isinstance(content, str):
        raise FsToolError(ErrorCode.INVALID_ARGUMENT, "Content must be a string.")
    try:
        size = len(content.encode("utf-8"))
    except UnicodeEncodeError as exc:  # e.g. lone surrogates
        raise FsToolError(ErrorCode.INVALID_ARGUMENT, "Content is not valid Unicode text.") from exc
    if size > MAX_WRITE_BYTES:
        raise FsToolError(
            ErrorCode.CONTENT_TOO_LARGE,
            f"Content is {size} bytes when UTF-8 encoded; the limit is {MAX_WRITE_BYTES} bytes.",
        )


def _ensure_parent_dirs(path: Path) -> list[str]:
    """Create missing parent directories of ``path`` (inside the root); return those created."""
    root = _workspace_root()
    if not root.is_dir():
        raise FsToolError(ErrorCode.NOT_FOUND, "The workspace root directory does not exist.")
    created = []
    current = root
    for part in path.parent.relative_to(root).parts:
        current = current / part
        if current.is_dir():
            continue
        if current.exists():
            raise FsToolError(
                ErrorCode.NOT_A_DIRECTORY,
                f"'{_display_path(current)}' is a file, so it cannot contain other files.",
            )
        current.mkdir()
        created.append(_display_path(current))
    return created


def _atomic_write(path: Path, content: str) -> None:
    """Write via a hidden temp file in the same directory, then ``os.replace`` onto ``path``.

    On any failure the temp file is removed and an existing target is left untouched.
    """
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o644  # mkstemp creates 0o600
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with open(fd, "w", encoding="utf-8", newline="") as handle:  # no "\r\n" translation
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise


def write_file(filepath: str, content: str) -> dict:
    """Atomically write UTF-8 text to a .txt or .md file, creating parent directories.

    Never raises: every failure is returned as an error envelope.
    """
    try:
        path = _resolve_path(filepath)
        display = _display_path(path)
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            supported = ", ".join(sorted(TEXT_EXTENSIONS))
            raise FsToolError(
                ErrorCode.UNSUPPORTED_FORMAT,
                f"Cannot write '{path.suffix or path.name}' files. Supported: {supported}.",
            )
        _validate_content(content)
        try:
            if path.is_dir():
                raise FsToolError(ErrorCode.NOT_A_FILE, f"'{display}' is a directory.")
            created_dirs = _ensure_parent_dirs(path)
            overwritten = path.exists()
            _atomic_write(path, content)
            stat = path.stat()
        except OSError as exc:
            raise _os_error(exc, display) from exc
        return _ok(
            {
                "path": display,
                "name": path.name,
                "size_bytes": stat.st_size,
                "overwritten": overwritten,
                "created_dirs": created_dirs,
                "modified": _iso_utc(stat.st_mtime),
            }
        )
    except FsToolError as exc:
        return _err(exc.code, exc.message)
    except Exception:
        logger.exception("write_file failed unexpectedly")
        return _err(ErrorCode.INTERNAL_ERROR, "Unexpected internal error while writing the file.")


def _keyword_pattern(keyword: str, whole_word: bool) -> re.Pattern[str]:
    """Compile a literal, case-insensitive pattern for ``keyword`` (already stripped)."""
    pattern = re.escape(keyword)
    if whole_word:
        # Only add \b on a side that is a word character, so "C++" or ".NET" still match.
        if re.match(r"\w", keyword[0]):
            pattern = rf"\b{pattern}"
        if re.match(r"\w", keyword[-1]):
            pattern = rf"{pattern}\b"
    return re.compile(pattern, re.IGNORECASE)


def _validate_keyword(keyword: str, whole_word: bool) -> str:
    if not isinstance(keyword, str) or not keyword.strip():
        raise FsToolError(ErrorCode.INVALID_ARGUMENT, "Keyword must be a non-empty string.")
    keyword = keyword.strip()
    if len(keyword) > MAX_KEYWORD_LENGTH:
        raise FsToolError(
            ErrorCode.INVALID_ARGUMENT,
            f"Keyword is {len(keyword)} characters; the limit is {MAX_KEYWORD_LENGTH}.",
        )
    if not isinstance(whole_word, bool):
        raise FsToolError(ErrorCode.INVALID_ARGUMENT, "whole_word must be a boolean.")
    return keyword


def _match_context(content: str, start: int, end: int) -> str:
    """About SEARCH_CONTEXT_CHARS either side, whitespace collapsed, ellipsis where cut.

    Control characters are already spaces (``_sanitise_text``), but invisible format characters
    such as zero-width spaces and soft hyphens are not, so non-printables become spaces here too.
    """
    lo = max(0, start - SEARCH_CONTEXT_CHARS)
    hi = min(len(content), end + SEARCH_CONTEXT_CHARS)
    printable = "".join(c if c.isprintable() else " " for c in content[lo:hi])
    snippet = " ".join(printable.split())
    prefix = ELLIPSIS if lo > 0 else ""
    suffix = ELLIPSIS if hi < len(content) else ""
    return f"{prefix}{snippet}{suffix}"


def search_in_file(filepath: str, keyword: str, whole_word: bool = False) -> dict:
    """Case-insensitive literal search for ``keyword`` in any file ``read_file`` supports.

    Match offsets index into the same extracted text ``read_file`` returns. Never raises:
    every failure (including read/extraction errors) is returned as an error envelope.
    """
    try:
        keyword = _validate_keyword(keyword, whole_word)
        content, metadata = _load_document(filepath)
        is_pdf = "page_count" in metadata

        matches, total = [], 0
        for match in _keyword_pattern(keyword, whole_word).finditer(content):
            total += 1
            if len(matches) >= MAX_SEARCH_MATCHES:
                continue
            start, end = match.span()
            matches.append(
                {
                    "start": start,
                    "end": end,
                    "matched_text": match.group(),
                    "line_number": content.count("\n", 0, start) + 1,
                    "page": content.count("\f", 0, start) + 1 if is_pdf else None,
                    "context": _match_context(content, start, end),
                }
            )
        return _ok(
            {
                "path": metadata["path"],
                "keyword": keyword,
                "whole_word": whole_word,
                "found": total > 0,
                "total_matches": total,
                "returned_matches": len(matches),
                "truncated": total > len(matches),
                "matches": matches,
                "warnings": metadata["warnings"],
            }
        )
    except FsToolError as exc:
        return _err(exc.code, exc.message)
    except Exception:
        logger.exception("search_in_file failed unexpectedly")
        return _err(ErrorCode.INTERNAL_ERROR, "Unexpected internal error while searching the file.")
