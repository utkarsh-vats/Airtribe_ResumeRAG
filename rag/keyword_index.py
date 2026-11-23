"""BM25 keyword index over a collection's stored chunks: the keyword side of hybrid retrieval.

The index is built in memory from the chunk documents (``chunk.text``) and metadata that the
vector store holds, so it always matches the collection; nothing is persisted. Scoring is
``rank_bm25.BM25Okapi`` with configurable ``k1`` and ``b``.

Tokenizer modes (``TokenizerMode``):

- ``words``: lowercased word tokens over the whole chunk text. Symbols that belong to a name
  stay attached (``c++``, ``c#``, ``.net``, ``node.js``); a small English stopword list is
  dropped; no stemming.
- ``skills``: only canonical skill tokens (``skill:kubernetes``) from ``rag.skill_matcher``, so
  aliases collide (``k8s`` and ``Kubernetes``) and the ambiguous-name case rules apply. They
  follow the extraction contract: only chunks from skill-source sections (skills, experience,
  projects) get them, and only from the text after ``body_start`` (role-header lines are not
  a source). Implications are not added: a token means the text mentions the skill.
- ``words+skills`` (default): both, in one bag per chunk.

A query is tokenised with the same mode; for its skill tokens the whole query text is a source.
``search_skills`` builds the query from canonical skill names directly.
"""

import re
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from rank_bm25 import BM25Okapi

from rag.sections import SectionKind
from rag.skill_matcher import find_mentions
from rag.skills import canonicalize

DEFAULT_K1 = 1.5
DEFAULT_B = 0.75
SKILL_PREFIX = "skill:"
SKILL_SOURCE_SECTIONS = frozenset(
    {SectionKind.SKILLS.value, SectionKind.EXPERIENCE.value, SectionKind.PROJECTS.value}
)

# Function words only. Single letters that are skills (c, r) and short skill names (go) are
# deliberately absent.
_STOPWORD_TEXT = """
    a an and are as at be been but by for from had has have he her his i if in into is it its
    me my no not of on or our she so than that the their them then there these they this to
    us was we were what when where which who will with you your
"""
STOPWORDS = frozenset(_STOPWORD_TEXT.split())

# A word token: letters/digits, optionally joined by dots (node.js, asp.net), optionally with
# a leading dot when nothing word-like precedes it (.net) and trailing + or # after a letter
# (c++, c#). Runs on case-folded text.
_WORD = re.compile(r"(?:(?<![^\W_])\.)?[^\W_]+(?:\.[^\W_]+)*(?:(?<=[^\W\d_])[+#]+)?")


class TokenizerMode(StrEnum):
    """Which tokens the index and its queries use."""

    WORDS = "words"
    SKILLS = "skills"
    WORDS_SKILLS = "words+skills"

    @property
    def words(self) -> bool:
        """Whether word tokens are used."""
        return self is not TokenizerMode.SKILLS

    @property
    def skills(self) -> bool:
        """Whether skill tokens are used."""
        return self is not TokenizerMode.WORDS


DEFAULT_MODE = TokenizerMode.WORDS_SKILLS


class KeywordIndexError(RuntimeError):
    """The stored chunks can't be indexed (e.g. written by an older pipeline)."""


def word_tokens(text: str) -> list[str]:
    """Lowercased, symbol-aware word tokens without stopwords."""
    return [t for t in _WORD.findall(text.casefold()) if t not in STOPWORDS]


def skill_token(canonical: str) -> str:
    """``skill:<name>``: the canonical name case-folded, spaces as ``_`` (``skill:rest_apis``)."""
    return SKILL_PREFIX + "_".join(canonical.casefold().split())


def skill_tokens(text: str) -> list[str]:
    """One skill token per skill mention in ``text`` (mention contract, no implications)."""
    return [skill_token(m.canonical) for m in find_mentions(text)]


def document_tokens(text: str, section: str, body_start: int, mode: TokenizerMode) -> list[str]:
    """A chunk's tokens: words over all of it, skills from a skill-source section's body."""
    tokens = word_tokens(text) if mode.words else []
    if mode.skills and section in SKILL_SOURCE_SECTIONS:
        tokens += skill_tokens("\n".join(text.split("\n")[body_start:]))
    return tokens


def query_tokens(text: str, mode: TokenizerMode) -> list[str]:
    """A query's tokens; the whole query text is a skill source."""
    tokens = word_tokens(text) if mode.words else []
    if mode.skills:
        tokens += skill_tokens(text)
    return tokens


@dataclass(frozen=True)
class KeywordDoc:
    """One stored chunk as the keyword index sees it."""

    chunk_id: str
    text: str
    candidate_id: str
    section: str
    body_start: int = 0
    role_index: int | None = None


class KeywordIndex:
    """BM25 over one collection's chunks, built in memory."""

    def __init__(
        self,
        docs: Iterable[KeywordDoc],
        mode: TokenizerMode | str = DEFAULT_MODE,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ):
        """Tokenise ``docs`` (ordered by chunk id) and build the BM25 model."""
        start = time.perf_counter()
        self.mode = TokenizerMode(mode)
        self.k1 = k1
        self.b = b
        self.docs = tuple(sorted(docs, key=lambda d: d.chunk_id))
        ids = [d.chunk_id for d in self.docs]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate chunk ids")
        self._by_id = {d.chunk_id: i for i, d in enumerate(self.docs)}
        self._tokens = [
            document_tokens(d.text, d.section, d.body_start, self.mode) for d in self.docs
        ]
        # BM25Okapi divides by the average length, so an index with no tokens has no model.
        self.bm25 = BM25Okapi(self._tokens, k1=k1, b=b) if any(self._tokens) else None
        self.build_seconds = time.perf_counter() - start
        self.query_seconds: list[float] = []  # one entry per search call

    @classmethod
    def from_store(
        cls,
        store: Any,
        mode: TokenizerMode | str = DEFAULT_MODE,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> KeywordIndex:
        """Build from a ``VectorStore``'s stored documents and chunk metadata."""
        found = store.get(include=["documents", "metadatas"])
        docs = []
        for chunk_id, text, meta in zip(
            found["ids"], found["documents"], found["metadatas"], strict=True
        ):
            if "body_start" not in meta:
                raise KeywordIndexError(
                    f"chunk {chunk_id} has no body_start: the collection was written by an "
                    "older pipeline; re-index it"
                )
            docs.append(
                KeywordDoc(
                    chunk_id,
                    text or "",
                    meta["candidate_id"],
                    meta["section"],
                    meta["body_start"],
                    meta.get("role_index"),
                )
            )
        return cls(docs, mode=mode, k1=k1, b=b)

    def __len__(self) -> int:
        """Number of indexed chunks."""
        return len(self.docs)

    def doc(self, chunk_id: str) -> KeywordDoc:
        """One indexed chunk; raises ``KeyError`` if unknown."""
        return self.docs[self._by_id[chunk_id]]

    def tokens(self, chunk_id: str) -> list[str]:
        """The tokens indexed for one chunk; raises ``KeyError`` if unknown."""
        return list(self._tokens[self._by_id[chunk_id]])

    @property
    def last_query_seconds(self) -> float | None:
        """Latency of the most recent search, or None before the first."""
        return self.query_seconds[-1] if self.query_seconds else None

    def search(
        self, text: str, n: int, where: Mapping[str, Iterable[str]] | None = None
    ) -> list[tuple[str, float]]:
        """Top ``n`` ``(chunk_id, score)`` for a free-text query."""
        return self._search(query_tokens(text, self.mode), n, where)

    def search_skills(
        self, skills: Sequence[str], n: int, where: Mapping[str, Iterable[str]] | None = None
    ) -> list[tuple[str, float]]:
        """Top ``n`` for a query made of canonical skill names (or aliases), one term each.

        In ``words`` mode the names are word-tokenised; in ``skills`` mode each becomes its
        skill token; ``words+skills`` uses both. Raises ``ValueError`` for an unknown skill.
        """
        names: list[str] = []
        for skill in skills:
            canonical = canonicalize(skill)
            if canonical is None:
                raise ValueError(f"unknown skill {skill!r}")
            if canonical not in names:
                names.append(canonical)
        tokens: list[str] = []
        for name in names:
            if self.mode.words:
                tokens += word_tokens(name)
            if self.mode.skills:
                tokens.append(skill_token(name))
        return self._search(tokens, n, where)

    def _allowed(self, where: Mapping[str, Iterable[str]] | None) -> list[bool]:
        if not where:
            return [True] * len(self.docs)
        unknown = set(where) - {"candidate_ids", "sections"}
        if unknown:
            raise ValueError(f"unknown filter keys {sorted(unknown)}; use candidate_ids, sections")
        candidates = set(where["candidate_ids"]) if "candidate_ids" in where else None
        sections = set(where["sections"]) if "sections" in where else None
        return [
            (candidates is None or d.candidate_id in candidates)
            and (sections is None or d.section in sections)
            for d in self.docs
        ]

    def _search(
        self, tokens: list[str], n: int, where: Mapping[str, Iterable[str]] | None
    ) -> list[tuple[str, float]]:
        start = time.perf_counter()
        allowed = self._allowed(where)
        results: list[tuple[str, float]] = []
        if tokens and n > 0 and self.bm25 is not None:
            scores = self.bm25.get_scores(tokens)
            results = sorted(
                (
                    (doc.chunk_id, float(score))
                    for doc, score, ok in zip(self.docs, scores, allowed, strict=True)
                    if ok and score > 0
                ),
                key=lambda item: (-item[1], item[0]),
            )[:n]
        self.query_seconds.append(time.perf_counter() - start)
        return results
