"""Relevant excerpts: short, verbatim resume sentences that back a match.

From a ``ScoredMatch``'s top chunks (at most three, best first):

1. Only the body of each chunk is used: the lines after ``body_start``, so role-header lines
   (title, company, dates) never appear. Lines are joined with spaces and split into
   sentences (``rag.jd_parser.split_sentences``: not at ``e.g.``, ``i.e.`` or decimals).
2. Each sentence scores 2 per distinct JD must-have skill and 1 per distinct nice-to-have it
   mentions (``rag.skill_matcher``, the same mention rules as everywhere else). Ties go to
   chunk order, then sentence order.
3. Experience, projects and skills chunks are preferred. Summary, education, certifications
   and other sections are used only when no preferred sentence mentions a JD skill. Hobbies
   and coursework never are.
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
PREFERRED_SECTIONS = frozenset({"experience", "projects", "skills"})
EXCLUDED_SECTIONS = frozenset({"hobbies", "coursework"})
MUST_HAVE_WEIGHT = 2
NICE_TO_HAVE_WEIGHT = 1


def body_sentences(chunk: TopChunk) -> list[str]:
    """The sentences of the chunk's body (role-header lines excluded)."""
    body = " ".join(line.strip() for line in chunk.text.split("\n")[chunk.body_start :])
    return split_sentences(body)


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
