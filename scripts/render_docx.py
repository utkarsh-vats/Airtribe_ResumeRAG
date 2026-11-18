"""Render resume blocks as DOCX with python-docx. Dataset tooling.

Styled headings use Word's Title / Heading 1 styles, bullets are "List Bullet" paragraphs,
``Table`` blocks are real Word tables and a ``Panel`` is a run of shaded paragraphs. Core
properties are fixed so document metadata is deterministic. (The zip entry times python-docx
writes still vary; byte-identical DOCX output is step 3b's job.)
"""

from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from docx import Document
from docx.document import Document as DocxDocument
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt
from docx.text.paragraph import Paragraph as DocxParagraph

from scripts.resume_doc import (
    Block,
    BulletList,
    Heading,
    Panel,
    Paragraph,
    RoleHeader,
    RoleStyle,
    Spacer,
    Table,
)

FIXED_TIMESTAMP = datetime(2026, 11, 1, tzinfo=UTC)
PANEL_FILL = "E8EEF4"


def _shade(paragraph: DocxParagraph) -> None:
    shading = OxmlElement("w:shd")
    shading.set(qn("w:val"), "clear")
    shading.set(qn("w:color"), "auto")
    shading.set(qn("w:fill"), PANEL_FILL)
    paragraph._p.get_or_add_pPr().append(shading)


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
            company = f"{block.company}, {block.location}" if block.location else block.company
            title = document.add_paragraph()
            title.add_run(block.title).bold = True
            if block.style is RoleStyle.INLINE:
                title.add_run(f" | {company} | {block.date_range}")
                added.append(title)
            else:
                details = document.add_paragraph()
                details.add_run(f"{company} | {block.date_range}").italic = True
                added += [title, details]
        case Table():
            columns = max(len(row) for row in block.rows)
            table = document.add_table(rows=len(block.rows), cols=columns)
            table.style = "Table Grid"
            for row, values in zip(table.rows, block.rows, strict=True):
                for cell, value in zip(row.cells, values, strict=False):
                    cell.text = value
        case Spacer():
            added.append(document.add_paragraph())
        case Panel():
            for inner in block.blocks:
                _add(document, inner, shaded=True)
    if shaded:
        for paragraph in added:
            _shade(paragraph)


def write_docx(blocks: Iterable[Block], path: Path, *, author: str, title: str) -> None:
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
    document.save(str(path))
