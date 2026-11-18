"""Render resume blocks as DOCX with python-docx. Dataset tooling.

Styled headings use Word's Title / Heading 1 styles, bullets are "List Bullet" paragraphs,
``Table`` blocks are real Word tables and a ``Panel`` is a run of shaded paragraphs. A
``SPLIT`` role header right-aligns its dates with a right tab stop.

Output is byte-identical on every platform: core properties are fixed, and the package
python-docx saves is rewritten (``_normalise_package``) with the same entry order, a fixed
1980-01-01 timestamp, fixed host-system and permission fields, and ``ZIP_STORED``. Storing
instead of deflating means the bytes never depend on the zlib build (CPython's Windows
build ships zlib-ng); the CRC-32 checksums are fully specified, so every zlib agrees on them.
"""

import io
import zipfile
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from docx import Document
from docx.document import Document as DocxDocument
from docx.enum.text import WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt
from docx.text.paragraph import Paragraph as DocxParagraph

from scripts.resume_doc import (
    Block,
    BulletList,
    Heading,
    PageBreak,
    Panel,
    Paragraph,
    RoleHeader,
    RoleStyle,
    Spacer,
    Table,
)

FIXED_TIMESTAMP = datetime(2026, 11, 1, tzinfo=UTC)
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)  # the earliest time a zip entry can carry
PANEL_FILL = "E8EEF4"


def _shade(paragraph: DocxParagraph) -> None:
    shading = OxmlElement("w:shd")
    shading.set(qn("w:val"), "clear")
    shading.set(qn("w:color"), "auto")
    shading.set(qn("w:fill"), PANEL_FILL)
    paragraph._p.get_or_add_pPr().append(shading)


def _text_width(document: DocxDocument):
    section = document.sections[0]
    return section.page_width - section.left_margin - section.right_margin


def _role(document: DocxDocument, block: RoleHeader) -> list[DocxParagraph]:
    company = f"{block.company}, {block.location}" if block.location else block.company
    title = document.add_paragraph()
    title.add_run(block.title).bold = True
    if block.style is RoleStyle.INLINE:
        title.add_run(f" | {company} | {block.date_range}")
        return [title]
    if block.style is RoleStyle.SPLIT:
        title.paragraph_format.tab_stops.add_tab_stop(_text_width(document), WD_TAB_ALIGNMENT.RIGHT)
        title.add_run(f" | {company}")
        title.add_run(f"\t{block.date_range}").italic = True
        return [title]
    details = document.add_paragraph()
    details.add_run(f"{company} | {block.date_range}").italic = True
    return [title, details]


def _add(document: DocxDocument, block: Block, shaded: bool) -> None:
    added: list[DocxParagraph] = []
    match block:
        case Heading(styled=True):
            added.append(document.add_heading(block.text, level=block.level))
        case Heading():
            paragraph = document.add_paragraph()
            run = paragraph.add_run(block.text)
            run.bold = True
            run.font.size = Pt(16 if block.level == 0 else 12)
            added.append(paragraph)
        case Paragraph():
            paragraph = document.add_paragraph()
            paragraph.add_run(block.text).bold = block.bold
            added.append(paragraph)
        case BulletList():
            added += [document.add_paragraph(item, style="List Bullet") for item in block.items]
        case RoleHeader():
            added += _role(document, block)
        case Table():
            columns = max(len(row) for row in block.rows)
            table = document.add_table(rows=len(block.rows), cols=columns)
            table.style = "Table Grid"
            for row, values in zip(table.rows, block.rows, strict=True):
                for cell, value in zip(row.cells, values, strict=False):
                    cell.text = value
        case Spacer():
            added.append(document.add_paragraph())
        case PageBreak():
            document.add_page_break()
        case Panel():
            for inner in block.blocks:
                _add(document, inner, shaded=True)
    if shaded:
        for paragraph in added:
            _shade(paragraph)


def _normalise_package(package: bytes) -> bytes:
    """Rewrite a zip so its bytes depend only on the entries' names and contents."""
    out = io.BytesIO()
    with (
        zipfile.ZipFile(io.BytesIO(package)) as source,
        zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as target,
    ):
        for entry in source.infolist():
            info = zipfile.ZipInfo(entry.filename, date_time=ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 0  # ZipInfo defaults to 0 on Windows and 3 elsewhere
            info.create_version = info.extract_version = 20
            info.external_attr = 0o600 << 16  # what writestr would force anyway
            target.writestr(info, source.read(entry.filename))
    return out.getvalue()


def render_docx(blocks: Iterable[Block], *, author: str, title: str) -> bytes:
    document = Document()
    for block in blocks:
        _add(document, block, shaded=False)
    props = document.core_properties
    props.author = author
    props.last_modified_by = author
    props.title = title
    props.subject = ""
    props.keywords = ""
    props.comments = ""
    props.category = ""
    props.revision = 1
    props.created = FIXED_TIMESTAMP
    props.modified = FIXED_TIMESTAMP
    props.last_printed = FIXED_TIMESTAMP
    buffer = io.BytesIO()
    document.save(buffer)
    return _normalise_package(buffer.getvalue())


def write_docx(blocks: Iterable[Block], path: Path, *, author: str, title: str) -> None:
    path.write_bytes(render_docx(blocks, author=author, title=title))
