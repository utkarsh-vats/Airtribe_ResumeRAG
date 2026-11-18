"""Reference JD section splitter (test helper only).

Implements the heading contract in ``data/job_descriptions/README.md`` so the tests can check
the JD files. The runtime JD parser in ``rag/`` (step 10) implements it independently; don't
import this module from ``rag/``.

A heading is a line that, after stripping whitespace and one trailing ``:``, equals one of the
known heading strings (case-insensitive). Lines before the first heading form the ``header``
(title and company line).
"""

from dataclasses import dataclass

REQUIREMENTS = "requirements"
NICE_TO_HAVE = "nice_to_have"
OTHER = "other"
HEADER = "header"

HEADINGS: dict[str, tuple[str, ...]] = {
    REQUIREMENTS: (
        "Requirements",
        "What you'll need",
        "Must have",
        "You have",
        "What we're looking for",
    ),
    NICE_TO_HAVE: ("Nice to have", "Bonus points", "Preferred", "Preferred qualifications"),
    OTHER: (
        "About us",
        "Who we are",
        "About the role",
        "The role",
        "The team",
        "What you'll do",
        "Responsibilities",
        "Your day-to-day",
        "Location and benefits",
        "Where you'll work",
        "Benefits",
        "Perks",
    ),
}

_KIND_BY_HEADING = {h.casefold(): kind for kind, hs in HEADINGS.items() for h in hs}


@dataclass(frozen=True)
class Section:
    kind: str
    heading: str | None
    text: str


def heading_kind(line: str) -> str | None:
    """The section kind if ``line`` is a heading, else None."""
    return _KIND_BY_HEADING.get(line.strip().removesuffix(":").strip().casefold())


def split_jd(text: str) -> list[Section]:
    """Split a JD into its header and headed sections, in order."""
    sections: list[Section] = []
    kind, heading, body = HEADER, None, []
    for line in text.splitlines():
        found = heading_kind(line)
        if found:
            sections.append(Section(kind, heading, "\n".join(body).strip()))
            kind, heading, body = found, line.strip(), []
        else:
            body.append(line)
    sections.append(Section(kind, heading, "\n".join(body).strip()))
    return sections


def text_of(sections: list[Section], *kinds: str) -> str:
    return "\n".join(s.text for s in sections if s.kind in kinds)
