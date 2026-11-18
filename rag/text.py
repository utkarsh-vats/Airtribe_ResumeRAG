"""Normalise text extracted by ``fs_tools.read_file`` into clean, ordered lines.

Removes rendering artefacts so section detection sees the same lines for every format: PDF
footers ("<Name> \u00b7 Page N of M") and form-feed page separators, TXT underline rows, bullet
glyphs and indentation, tabs (a DOCX right tab stop becomes " | ", so the modern role line
matches the inline form) and runs of spaces. Blank lines are dropped; order is kept.
"""

import re

FOOTER = re.compile(r"^.+ \u00b7 Page \d+ of \d+$")
UNDERLINE = re.compile(r"^(?:=+|-+)$")
# A bullet glyph at the start of a line: "\u2022" with or without a following space, "-" or "*"
# only when followed by whitespace (so "-5% churn" or "*nix" keep their first character).
BULLET = re.compile(r"^(?:\u2022\s*|[-*]\s+)")
SPACES = re.compile(r"[ \u00a0]{2,}")


def clean_line(line: str) -> str:
    """One line with tabs, bullets, indentation and repeated spaces normalised."""
    line = line.replace("\f", " ").replace("\t", " | ").strip()
    line = BULLET.sub("", line).strip()
    return SPACES.sub(" ", line)


def clean_lines(text: str) -> list[str]:
    """The non-empty content lines of ``text``, without footers or underline rows."""
    lines = []
    for raw in text.splitlines():
        line = clean_line(raw)
        if line and not FOOTER.match(line) and not UNDERLINE.match(line):
            lines.append(line)
    return lines
