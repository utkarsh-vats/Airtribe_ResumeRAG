"""Render resume blocks as PDF with reportlab platypus. Dataset tooling.

Only the standard Helvetica family is used (no embedded fonts; persona text is guaranteed
cp1252). Hyphenation is off, bullets use the U+2022 glyph, and each role (its header plus its
bullets) is kept together so a page break never splits it. ``SPLIT`` role headers are a
two-cell borderless table: title | company left, dates right-aligned. A ``Panel`` is a single
shaded cell: a visual block in the one column, not a second column.

Documents longer than one page get a "<Name> · Page N of M" footer on every page; that needs
the page count, so those documents are built twice.

Output is byte-identical on every platform: reportlab's invariant mode fixes the creation
date and document ID, page compression is off (so no zlib build is involved), metadata is
fixed and the PDF is written to memory, so no file path can leak in.
"""

import io
from collections.abc import Iterable, Sequence
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    KeepTogether,
    PageTemplate,
    TableStyle,
)
from reportlab.platypus import PageBreak as PdfPageBreak
from reportlab.platypus import Paragraph as PdfParagraph
from reportlab.platypus import Spacer as PdfSpacer
from reportlab.platypus import Table as PdfTable

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

PAGE_SIZE = A4
MARGIN = 18 * mm
FRAME_WIDTH = PAGE_SIZE[0] - 2 * MARGIN
BULLET = "\N{BULLET}"
FOOTER_SEPARATOR = " \N{MIDDLE DOT} "
PANEL_FILL = colors.HexColor("#E8EEF4")
CREATOR = "Airtribe ResumeRAG resume generator"


def _style(name: str, **overrides) -> ParagraphStyle:
    base = {
        "fontName": "Helvetica",
        "fontSize": 10,
        "leading": 13,
        "hyphenationLang": "",
        "embeddedHyphenation": 0,
        "splitLongWords": 0,
    }
    return ParagraphStyle(name, **(base | overrides))


STYLES = {
    "name": _style("name", fontName="Helvetica-Bold", fontSize=18, leading=22, spaceAfter=2),
    "heading": _style(
        "heading", fontName="Helvetica-Bold", fontSize=12, leading=15, spaceBefore=10, spaceAfter=3
    ),
    "heading_plain": _style(
        "heading_plain", fontName="Helvetica-Bold", fontSize=10.5, leading=13, spaceBefore=8
    ),
    "body": _style("body", spaceAfter=2),
    "bold": _style("bold", fontName="Helvetica-Bold", spaceAfter=1),
    "details": _style("details", fontName="Helvetica-Oblique", spaceAfter=2),
    "dates": _style("dates", fontName="Helvetica-Oblique", alignment=TA_RIGHT),
    "bullet": _style("bullet", leftIndent=12, bulletIndent=2, spaceAfter=1),
}


def _p(text: str, style: str, **kwargs) -> PdfParagraph:
    return PdfParagraph(escape(text), STYLES[style], **kwargs)


def _role(block: RoleHeader) -> list[Flowable]:
    company = f"{block.company}, {block.location}" if block.location else block.company
    title = f"<b>{escape(block.title)}</b>"
    match block.style:
        case RoleStyle.INLINE:
            line = f"{title} | {escape(company)} | {escape(block.date_range)}"
            return [PdfParagraph(line, STYLES["body"])]
        case RoleStyle.SPLIT:
            left = PdfParagraph(f"{title} | {escape(company)}", STYLES["body"])
            table = PdfTable(
                [[left, _p(block.date_range, "dates")]],
                colWidths=[FRAME_WIDTH * 0.7, FRAME_WIDTH * 0.3],
            )
            table.setStyle(
                TableStyle(
                    [
                        ("LEFTPADDING", (0, 0), (-1, -1), 0),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                        ("TOPPADDING", (0, 0), (-1, -1), 0),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ]
                )
            )
            return [table]
        case _:
            return [
                PdfParagraph(title, STYLES["bold"]),
                _p(f"{company} | {block.date_range}", "details"),
            ]


def _flowables(block: Block) -> list[Flowable]:
    match block:
        case Heading(level=0):
            return [_p(block.text, "name")]
        case Heading():
            return [_p(block.text, "heading" if block.styled else "heading_plain")]
        case Paragraph():
            return [_p(block.text, "bold" if block.bold else "body")]
        case BulletList():
            return [_p(item, "bullet", bulletText=BULLET) for item in block.items]
        case RoleHeader():
            return _role(block)
        case Table():
            table = PdfTable([[_p(cell, "body") for cell in row] for row in block.rows])
            table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey)]))
            return [table]
        case Spacer():
            return [PdfSpacer(1, 6)]
        case PageBreak():
            return [PdfPageBreak()]
        case Panel():
            inner = [f for b in block.blocks for f in _flowables(b)]
            panel = PdfTable([[inner]], colWidths=[FRAME_WIDTH])
            panel.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), PANEL_FILL),
                        ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
                        ("LEFTPADDING", (0, 0), (-1, -1), 8),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ]
                )
            )
            return [panel, PdfSpacer(1, 6)]
    raise TypeError(f"unknown block {block!r}")


def _story(blocks: Sequence[Block]) -> list[Flowable]:
    """Flowables for ``blocks``; each role header is kept together with its bullets."""
    story: list[Flowable] = []
    index = 0
    while index < len(blocks):
        block = blocks[index]
        if isinstance(block, RoleHeader) and isinstance(
            following := blocks[index + 1] if index + 1 < len(blocks) else None, BulletList
        ):
            story.append(KeepTogether([*_flowables(block), *_flowables(following)]))
            index += 2
            continue
        story += _flowables(block)
        index += 1
    return story


def footer_text(author: str, page: int, total: int) -> str:
    return f"{author}{FOOTER_SEPARATOR}Page {page} of {total}"


def _build(
    blocks: Sequence[Block], author: str, title: str, total: int | None
) -> tuple[bytes, int]:
    buffer = io.BytesIO()
    document = BaseDocTemplate(
        buffer,
        pagesize=PAGE_SIZE,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN,
        title=title,
        author=author,
        subject="",
        keywords="",
        creator=CREATOR,
        invariant=1,
        pageCompression=0,
    )

    def footer(canvas: Canvas, doc: BaseDocTemplate) -> None:
        # Drawn when the page ends, so it follows the page's content in extraction order.
        if total:
            canvas.saveState()
            canvas.setFont("Helvetica", 8)
            canvas.drawCentredString(
                PAGE_SIZE[0] / 2, MARGIN / 2, footer_text(author, doc.page, total)
            )
            canvas.restoreState()

    frame = Frame(MARGIN, MARGIN, FRAME_WIDTH, PAGE_SIZE[1] - 2 * MARGIN, id="body")
    document.addPageTemplates([PageTemplate("page", frames=[frame], onPageEnd=footer)])
    document.build(_story(blocks))
    return buffer.getvalue(), document.page


def render_pdf(blocks: Iterable[Block], *, author: str, title: str) -> bytes:
    blocks = list(blocks)
    data, pages = _build(blocks, author, title, total=None)
    if pages > 1:
        data, again = _build(blocks, author, title, total=pages)
        assert again == pages, "the footer changed the page count"
    return data


def write_pdf(blocks: Iterable[Block], path: Path, *, author: str, title: str) -> None:
    path.write_bytes(render_pdf(blocks, author=author, title=title))
