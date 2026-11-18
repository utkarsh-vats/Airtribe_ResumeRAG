"""Format-agnostic resume document model (dataset tooling).

``scripts/layouts.py`` turns a persona into a list of these blocks, and each renderer
(``render_txt``, ``render_docx``; PDF in step 3b) draws them. All text is final: layouts
apply casing, colons, date formats and skill labels, so renderers only decide presentation.
"""

from dataclasses import dataclass
from enum import StrEnum


class RoleStyle(StrEnum):
    STACKED = "stacked"  # title on one line; company and dates on the next
    INLINE = "inline"  # title | company | dates on one line


@dataclass(frozen=True)
class Heading:
    """A heading. Level 0 is the candidate's name; level 1 is a section heading."""

    text: str
    level: int = 1
    underline: str | None = None  # TXT: a row of this character under the heading
    styled: bool = True  # DOCX: a real Heading/Title style; otherwise a bold paragraph


@dataclass(frozen=True)
class Paragraph:
    text: str
    bold: bool = False


@dataclass(frozen=True)
class BulletList:
    items: tuple[str, ...]
    marker: str = "-"  # TXT bullet character


@dataclass(frozen=True)
class RoleHeader:
    title: str
    company: str
    date_range: str
    location: str | None = None
    style: RoleStyle = RoleStyle.STACKED


@dataclass(frozen=True)
class Table:
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class Spacer:
    pass


@dataclass(frozen=True)
class Panel:
    """A shaded / boxed group (sidebar layout). Content order stays single-column."""

    blocks: tuple[Block, ...]


type Block = Heading | Paragraph | BulletList | RoleHeader | Table | Spacer | Panel


def block_texts(blocks: list[Block] | tuple[Block, ...]) -> list[str]:
    """Every visible text string in ``blocks``, in order (for tests and checks)."""
    texts: list[str] = []
    for block in blocks:
        match block:
            case Heading(text=text) | Paragraph(text=text):
                texts.append(text)
            case BulletList(items=items):
                texts.extend(items)
            case RoleHeader():
                texts.extend(
                    t for t in (block.title, block.company, block.location, block.date_range) if t
                )
            case Table(rows=rows):
                texts.extend(cell for row in rows for cell in row)
            case Panel(blocks=inner):
                texts.extend(block_texts(inner))
    return texts
