"""Relevant excerpts: short, verbatim resume sentences that back a match.

From a ``ScoredMatch``'s top chunks (at most three, best first):

1. Only the body of each chunk is used: the lines after ``body_start``, so role-header lines
   (title, company, dates) never appear. Lines are joined with spaces into units, except
   after a **title line**: a line with no terminal punctuation and at most five words,
   followed by a line starting with an uppercase letter (a project name before its
   description). A title line is a unit of its own and is never an excerpt. Long wrapped
   lines still join even when the continuation starts with a capital ("...REST APIs with"
   + "Django REST Framework."). Units are split into sentences
   (``rag.jd_parser.split_sentences``: not at ``e.g.``, ``i.e.`` or decimals).
2. Each sentence scores 2 per distinct JD must-have skill and 1 per distinct nice-to-have it
   mentions (``rag.skill_matcher``, the same mention rules as everywhere else). Ties go to
   chunk order, then sentence order.
3. Experience and projects chunks are preferred. Summary, education, certifications and
   other sections are used only when no preferred sentence mentions a JD skill. Skills,
   hobbies and coursework chunks never supply excerpts (the skills section already feeds
   ``matched_skills``).
4. If no sentence mentions a JD skill, the excerpt is the first body sentence of the chunk
   with the best semantic score.
5. Duplicates are dropped, and each excerpt is cut at a word boundary with an ellipsis when
   longer than ``max_chars``.

The result is deterministic: it depends only on the match and the JD.
"""

from rag.jd_parser import JobRequirements, split_sentences
from rag.retrieval import TopChunk
from rag.scoring import ScoredMatch
from rag.skill_matcher import find_mentions

DEFAULT_MAX_EXCERPTS = 3
DEFAULT_MAX_CHARS = 220
ELLIPSIS = "\u2026"
PREFERRED_SECTIONS = frozenset({"experience", "projects"})
EXCLUDED_SECTIONS = frozenset({"skills", "hobbies", "coursework"})
TITLE_MAX_WORDS = 5
TERMINAL_PUNCTUATION = (".", "!", "?")
MUST_HAVE_WEIGHT = 2
NICE_TO_HAVE_WEIGHT = 1


def is_title_line(line: str, next_line: str) -> bool:
    """A short unpunctuated line followed by a capitalised one: a name, not a wrapped line."""
    return (
        not line.endswith(TERMINAL_PUNCTUATION)
        and len(line.split()) <= TITLE_MAX_WORDS
        and next_line[:1].isupper()
    )


def body_units(chunk: TopChunk) -> list[str]:
    """The chunk body's lines joined into units; title lines are dropped (see above)."""
    lines = [ln.strip() for ln in chunk.text.split("\n")[chunk.body_start :] if ln.strip()]
    units: list[str] = []
    current: list[str] = []
    for i, line in enumerate(lines):
        following = lines[i + 1] if i + 1 < len(lines) else ""
        title_like = bool(following) and is_title_line(line, following)
        if title_like and (not current or current[-1].endswith(TERMINAL_PUNCTUATION)):
            # A name on its own line (a project title): it ends what came before, and is
            # neither glued to its description nor used as an excerpt.
            if current:
                units.append(" ".join(current))
                current = []
            continue
        current.append(line)
        if title_like:  # a short unpunctuated tail before a capitalised line ends the unit
            units.append(" ".join(current))
            current = []
    if current:
        units.append(" ".join(current))
    return units


def body_sentences(chunk: TopChunk) -> list[str]:
    """The sentences of the chunk's body (role-header and title lines excluded)."""
    return [s for unit in body_units(chunk) for s in split_sentences(unit)]


def truncate(text: str, max_chars: int) -> str:
    """``text`` cut at a word boundary to at most ``max_chars``, ending in an ellipsis."""
    if len(text) <= max_chars:
        return text
    cut = text[: max_chars - 1]
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    return cut.rstrip(" ,;:-") + ELLIPSIS


def sentence_score(sentence: str, must: frozenset[str], nice: frozenset[str]) -> int:
    """2 per distinct must-have skill mentioned, 1 per distinct nice-to-have."""
    skills = {m.canonical for m in find_mentions(sentence)}
    return MUST_HAVE_WEIGHT * len(skills & must) + NICE_TO_HAVE_WEIGHT * len(skills & nice)


def _dedupe_and_cut(sentences: list[str], max_excerpts: int, max_chars: int) -> list[str]:
    seen: set[str] = set()
    out = []
    for sentence in sentences:
        key = " ".join(sentence.split()).casefold()
        if key not in seen:
            seen.add(key)
            out.append(truncate(sentence, max_chars))
        if len(out) == max_excerpts:
            break
    return out


def select_excerpts(
    match: ScoredMatch,
    job: JobRequirements,
    max_excerpts: int = DEFAULT_MAX_EXCERPTS,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> list[str]:
    """Up to ``max_excerpts`` sentences from the match's top chunks; see the module docstring."""
    if max_excerpts < 1 or max_chars < 2:
        raise ValueError("need max_excerpts >= 1 and max_chars >= 2")
    must = frozenset(job.critical_skills)
    nice = frozenset(job.nice_to_have) - must
    chunks = [c for c in match.top_chunks if c.section not in EXCLUDED_SECTIONS]
    # (score, chunk order, sentence order, sentence) for every sentence that names a JD skill
    scored: dict[bool, list[tuple[int, int, int, str]]] = {True: [], False: []}
    for chunk_order, chunk in enumerate(chunks):
        preferred = chunk.section in PREFERRED_SECTIONS
        for sentence_order, sentence in enumerate(body_sentences(chunk)):
            score = sentence_score(sentence, must, nice)
            if score > 0:
                scored[preferred].append((score, chunk_order, sentence_order, sentence))
    candidates = scored[True] or scored[False]
    if candidates:
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        return _dedupe_and_cut([item[3] for item in candidates], max_excerpts, max_chars)
    semantic = [c for c in chunks if c.semantic_weighted is not None]
    best = sorted(semantic, key=lambda c: -(c.semantic_weighted or 0.0))  # stable: chunk order
    for chunk in best[:1]:
        sentences = body_sentences(chunk)
        if sentences:
            return _dedupe_and_cut(sentences[:1], max_excerpts, max_chars)
    return []
