"""Chunkers: turn a parsed resume into chunks for embedding.

- ``SectionChunker`` (default): one chunk per section, and one per role inside experience
  sections, each with an optional contextual header for embedding
  (``Section: Experience | Role: <Title> at <Company>`` or ``Section: <Kind>``). The header
  block (name, contact) is never chunked.
- ``FixedSizeChunker``: the naive baseline. Overlapping word windows over the whole cleaned
  text, header included, with no context.

``Chunk.body_start`` is the line offset in ``text`` where the body begins: a role chunk starts
with its role-header lines (title, company, dates), everything else starts with body. The
keyword index uses it to take skill tokens from role bodies only, as metadata extraction does.

Chunkers do no file IO: they take the resume's path (relative to the workspace root, e.g.
``resumes/john_doe.pdf``) and its ``ParsedResume``. ``chunker_id`` encodes every parameter
that changes the output, so step 8 can derive a collection name from it.
"""

import hashlib
import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Protocol

from rag.sections import ParsedResume, SectionKind, content_start, preamble, split_roles

DEFAULT_MAX_WORDS = 300  # bge-small truncates at 512 tokens; 300 words leaves headroom
DEFAULT_WINDOW = 120
DEFAULT_OVERLAP = 30


@dataclass(frozen=True)
class Chunk:
    """One chunk of one resume."""

    chunk_id: str
    candidate_id: str  # resume file stem, matching the manifest's candidate ids
    resume_path: str  # relative to the workspace root, e.g. "resumes/john_doe.pdf"
    chunker_id: str
    section: str  # a SectionKind value
    role_index: int | None  # position among the resume's roles; None outside roles
    role_title: str | None
    role_company: str | None
    part: int  # > 0 only when an oversize chunk was split
    text: str  # the body, as extracted (lines joined with "\n")
    embed_text: str  # what gets embedded: text, optionally after a contextual header
    start_line: int  # span in ParsedResume.lines (end exclusive)
    end_line: int
    # Line offset in ``text`` where the body starts: the number of role-header lines at the
    # top of a role chunk's first part, else 0. Not part of the chunk id.
    body_start: int = 0


class Chunker(Protocol):
    """Anything that turns a parsed resume into chunks."""

    chunker_id: str

    def chunk(self, resume_path: str, parsed: ParsedResume) -> list[Chunk]:
        """The chunks of one resume, in document order."""
        ...


def chunk_id(
    resume_path: str, chunker_id: str, section: str, role_index: int | None, part: int, text: str
) -> str:
    """Deterministic id: a SHA-256 over the identifying fields, truncated to 16 hex chars."""
    payload = json.dumps([resume_path, chunker_id, section, role_index, part, text])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def candidate_id(resume_path: str) -> str:
    """The candidate id for a resume: its file stem (``resumes/john_doe.pdf`` -> ``john_doe``)."""
    return PurePosixPath(resume_path).stem


def word_count(text: str) -> int:
    """Whitespace-separated words in ``text``."""
    return len(text.split())


# --- Section-aware chunker ----------------------------------------------------------------


@dataclass(frozen=True)
class _Piece:
    section: SectionKind
    role_index: int | None
    role_title: str | None
    role_company: str | None
    start_line: int
    end_line: int
    header_lines: int = 0  # role-header lines at the start of a role piece


class SectionChunker:
    """One chunk per section and per experience role, with an optional contextual header."""

    def __init__(self, max_words: int = DEFAULT_MAX_WORDS, contextual_header: bool = True):
        """``max_words`` caps a chunk body; ``contextual_header`` adds the context line."""
        if max_words < 1:
            raise ValueError("max_words must be positive")
        self.max_words = max_words
        self.contextual_header = contextual_header
        suffix = "" if max_words == DEFAULT_MAX_WORDS else f"-w{max_words}"
        self.chunker_id = f"section-v1{suffix}" + ("" if contextual_header else "-noctx")

    def _pieces(self, parsed: ParsedResume) -> list[_Piece]:
        pieces: list[_Piece] = []
        role_index = 0
        for section in parsed.sections:
            if section.kind is not SectionKind.EXPERIENCE:
                pieces.append(
                    _Piece(section.kind, None, None, None, content_start(section), section.end_line)
                )
                continue
            roles = split_roles(section)
            start, end = preamble(section, roles)
            pieces.append(_Piece(section.kind, None, None, None, start, end))
            for role in roles:
                pieces.append(
                    _Piece(
                        section.kind,
                        role_index,
                        role.title,
                        role.company,
                        role.start_line,
                        role.end_line,
                        role.end_line - role.start_line - len(role.lines),
                    )
                )
                role_index += 1
        return pieces

    def context(self, section: SectionKind, title: str | None, company: str | None) -> str:
        """The contextual header line for a chunk."""
        header = f"Section: {section.value.title()}"
        if title is not None:
            header += f" | Role: {title} at {company}"
        return header

    def _parts(self, lines: list[str], start: int) -> list[tuple[list[str], int, int]]:
        """Split lines into parts of at most max_words, at line boundaries where possible."""
        parts: list[tuple[list[str], int, int]] = []
        current: list[str] = []
        current_start = start
        words = 0
        for offset, line in enumerate(lines):
            n = word_count(line)
            if current and words + n > self.max_words:
                parts.append((current, current_start, start + offset))
                current, current_start, words = [], start + offset, 0
            if n > self.max_words:  # one huge line: split it by words
                tokens = line.split()
                for i in range(0, len(tokens), self.max_words):
                    parts.append(
                        (
                            [" ".join(tokens[i : i + self.max_words])],
                            start + offset,
                            start + offset + 1,
                        )
                    )
                current_start = start + offset + 1
                continue
            current.append(line)
            words += n
        if current:
            parts.append((current, current_start, start + len(lines)))
        return parts

    def chunk(self, resume_path: str, parsed: ParsedResume) -> list[Chunk]:
        """Section and role chunks in document order; the header block is skipped."""
        chunks = []
        for piece in self._pieces(parsed):
            lines = list(parsed.lines[piece.start_line : piece.end_line])
            if not any(line.strip() for line in lines):
                continue
            context = self.context(piece.section, piece.role_title, piece.role_company)
            body_line = piece.start_line + piece.header_lines
            for part, (part_lines, start, end) in enumerate(self._parts(lines, piece.start_line)):
                text = "\n".join(part_lines)
                embed_text = f"{context}\n{text}" if self.contextual_header else text
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id(
                            resume_path,
                            self.chunker_id,
                            piece.section.value,
                            piece.role_index,
                            part,
                            text,
                        ),
                        candidate_id=candidate_id(resume_path),
                        resume_path=resume_path,
                        chunker_id=self.chunker_id,
                        section=piece.section.value,
                        role_index=piece.role_index,
                        role_title=piece.role_title,
                        role_company=piece.role_company,
                        part=part,
                        text=text,
                        embed_text=embed_text,
                        start_line=start,
                        end_line=end,
                        body_start=max(0, min(len(part_lines), body_line - start)),
                    )
                )
        return chunks


# --- Fixed-size baseline ------------------------------------------------------------------


def _line_sections(parsed: ParsedResume) -> list[str]:
    """The section kind of every cleaned line ("header" before the first section)."""
    kinds = [SectionKind.HEADER.value] * len(parsed.lines)
    for section in parsed.sections:
        for i in range(section.start_line, section.end_line):
            kinds[i] = section.kind.value
    return kinds


class FixedSizeChunker:
    """Overlapping word windows over the whole cleaned text; no structure, no context."""

    def __init__(self, size: int = DEFAULT_WINDOW, overlap: int = DEFAULT_OVERLAP):
        """Windows of ``size`` words; consecutive windows share ``overlap`` words."""
        if size < 1 or not 0 <= overlap < size:
            raise ValueError("need size >= 1 and 0 <= overlap < size")
        self.size = size
        self.overlap = overlap
        self.chunker_id = f"fixed-w{size}-o{overlap}"

    def windows(self, n_words: int) -> list[tuple[int, int]]:
        """Word-index windows ``[start, end)``; the last one ends at the last word."""
        if n_words == 0:
            return []
        spans, start, step = [], 0, self.size - self.overlap
        while True:
            end = min(start + self.size, n_words)
            spans.append((start, end))
            if end == n_words:
                return spans
            start += step

    def chunk(self, resume_path: str, parsed: ParsedResume) -> list[Chunk]:
        """One chunk per window, labelled with the section most of its words come from."""
        words = [(i, w) for i, line in enumerate(parsed.lines) for w in line.split()]
        kinds = _line_sections(parsed)
        chunks = []
        for part, (start, end) in enumerate(self.windows(len(words))):
            window = words[start:end]
            lines: dict[int, list[str]] = {}
            for line_index, word in window:
                lines.setdefault(line_index, []).append(word)
            text = "\n".join(" ".join(ws) for ws in lines.values())
            # Majority section by word count; ties go to the earliest in the window.
            counts = Counter(kinds[line_index] for line_index, _ in window)
            section = max(counts, key=lambda k: (counts[k], -list(counts).index(k)))
            chunks.append(
                Chunk(
                    chunk_id=chunk_id(resume_path, self.chunker_id, section, None, part, text),
                    candidate_id=candidate_id(resume_path),
                    resume_path=resume_path,
                    chunker_id=self.chunker_id,
                    section=section,
                    role_index=None,
                    role_title=None,
                    role_company=None,
                    part=part,
                    text=text,
                    embed_text=text,
                    start_line=window[0][0],
                    end_line=window[-1][0] + 1,
                )
            )
        return chunks


# --- Registry -----------------------------------------------------------------------------

_CHUNKERS: dict[str, Callable[..., Chunker]] = {
    "section": SectionChunker,
    "fixed": FixedSizeChunker,
}


def available_chunkers() -> list[str]:
    """Registered chunker names."""
    return sorted(_CHUNKERS)


def get_chunker(name: str, **params) -> Chunker:
    """A chunker by registry name, e.g. ``get_chunker("fixed", size=200, overlap=50)``."""
    try:
        factory = _CHUNKERS[name]
    except KeyError:
        raise ValueError(f"unknown chunker {name!r}; available: {available_chunkers()}") from None
    return factory(**params)
