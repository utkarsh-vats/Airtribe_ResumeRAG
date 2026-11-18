"""Reference skill-mention matcher (test helper only).

Implements the contract in ``data/ground_truth/README.md`` so the tests can check that persona
text and ground truth agree. The runtime extractor in ``rag/`` (step 6) implements the same
rules independently. Don't import this module from ``rag/``.

- Non-ambiguous skills: canonical name or alias, case-insensitive.
- Ambiguous skills: aliases case-insensitive; the canonical name only in its exact case.
- Matches are bounded by non-word characters (so ``C++``, ``C#``, ``.NET`` and ``Node.js``
  work, and ``Java`` never matches inside ``JavaScript``). When matches overlap, the longest
  wins (``React Native`` beats ``React``, ``AWS Lambda`` beats ``AWS``).

The caller decides which text to scan: skills sections, bullets and project descriptions.
Hobbies, coursework, titles, company and institution names are never passed in.
"""

import re
from dataclasses import dataclass
from functools import cache

from rag.skills import SKILLS


@dataclass(frozen=True)
class Mention:
    skill: str
    text: str
    start: int
    end: int
    via_alias: bool


def _compile(form: str, flags: int) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(form)}(?!\w)", flags)


@cache
def _patterns() -> tuple[tuple[str, bool, re.Pattern[str]], ...]:
    patterns = []
    for skill in SKILLS:
        flags = 0 if skill.ambiguous else re.IGNORECASE
        patterns.append((skill.name, False, _compile(skill.name, flags)))
        patterns.extend((skill.name, True, _compile(a, re.IGNORECASE)) for a in skill.aliases)
    return tuple(patterns)


def find_mentions(text: str) -> list[Mention]:
    """Non-overlapping skill mentions in ``text``, longest match first, in text order."""
    candidates = [
        Mention(skill, m.group(), m.start(), m.end(), via_alias)
        for skill, via_alias, pattern in _patterns()
        for m in pattern.finditer(text)
    ]
    candidates.sort(key=lambda m: (m.start - m.end, m.start))
    kept: list[Mention] = []
    for mention in candidates:
        if all(mention.end <= k.start or mention.start >= k.end for k in kept):
            kept.append(mention)
    return sorted(kept, key=lambda m: m.start)


def mentioned_skills(text: str) -> set[str]:
    """Canonical names of the skills mentioned in ``text``."""
    return {m.skill for m in find_mentions(text)}
