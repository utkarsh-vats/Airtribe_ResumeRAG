"""Skill mentions in text: the extractor side of the mention contract.

Implements the rules in ``data/ground_truth/README.md`` ("Skill-mention rules"):

- Non-ambiguous skills match by canonical name or alias, case-insensitively.
- Ambiguous skills match their safe aliases case-insensitively, but their canonical name only
  in its exact case (``Spark`` yes, ``spark`` no).
- A match must not have a word character (letter, digit, ``_``) directly before or after it,
  which works for names with symbols (``C++``, ``C#``, ``.NET``, ``Node.js``) and keeps
  ``Java`` out of ``JavaScript``. A space inside a name matches any run of whitespace, so a
  name wrapped across lines still counts.
- Overlaps: the longest match wins; between equally long matches the earlier one wins. Kept
  matches never overlap (``Apache Spark`` is one Spark mention, ``AWS Lambda`` is not also
  ``AWS``). Implications are applied by the caller afterwards.

Patterns are compiled once at import. Which text to scan (skills sections, role bodies,
projects) is the caller's decision; see ``rag.metadata``.
"""

import re
from typing import NamedTuple

from rag.skills import SKILLS


class Mention(NamedTuple):
    """One skill mention: canonical name, the text as written, and its span."""

    canonical: str
    matched_text: str
    span: tuple[int, int]


def _pattern(form: str, flags: int) -> re.Pattern[str]:
    body = r"\s+".join(re.escape(part) for part in form.split())
    return re.compile(rf"(?<!\w){body}(?!\w)", flags)


def _compile() -> tuple[tuple[str, re.Pattern[str]], ...]:
    patterns = []
    for skill in SKILLS:
        patterns.append((skill.name, _pattern(skill.name, 0 if skill.ambiguous else re.IGNORECASE)))
        patterns.extend((skill.name, _pattern(alias, re.IGNORECASE)) for alias in skill.aliases)
    return tuple(patterns)


_PATTERNS = _compile()


def find_mentions(text: str) -> list[Mention]:
    """Non-overlapping skill mentions in ``text``, in text order."""
    candidates = [
        Mention(name, match.group(), match.span())
        for name, pattern in _PATTERNS
        for match in pattern.finditer(text)
    ]
    # Longest first, then earliest; keep each candidate that overlaps nothing kept so far.
    candidates.sort(key=lambda m: (m.span[0] - m.span[1], m.span[0]))
    kept: list[Mention] = []
    for mention in candidates:
        start, end = mention.span
        if all(end <= k.span[0] or start >= k.span[1] for k in kept):
            kept.append(mention)
    return sorted(kept, key=lambda m: m.span)


def mentioned_skills(text: str) -> set[str]:
    """Canonical names mentioned in ``text`` (before implications)."""
    return {m.canonical for m in find_mentions(text)}
