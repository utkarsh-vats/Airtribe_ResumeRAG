"""Shared fixtures: an isolated workspace under tmp_path and builders for PDF/DOCX files."""

from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from docx import Document
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Workspace root at ``tmp_path/workspace``; ``tmp_path`` itself is outside the sandbox."""
    root = tmp_path / "workspace"
    root.mkdir()
    monkeypatch.setenv("WORKSPACE_ROOT", str(root))
    return root


@pytest.fixture
def make_pdf(workspace: Path) -> Callable[..., Path]:
    """Build a PDF in the workspace; each item of ``pages`` is the text lines of one page."""

    def build(relpath: str, pages: Sequence[Sequence[str]], shapes_only: bool = False) -> Path:
        path = workspace / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        pdf = canvas.Canvas(str(path), pagesize=A4, invariant=1)
        for lines in pages:
            if shapes_only:
                pdf.rect(72, 600, 200, 100, fill=1)
            for offset, line in enumerate(lines):
                pdf.drawString(72, 780 - offset * 16, line)
            pdf.showPage()
        pdf.save()
        return path

    return build


@pytest.fixture
def make_docx(workspace: Path) -> Callable[..., Path]:
    """Build a DOCX: ``before`` paragraphs, an optional table, then ``after`` paragraphs."""

    def build(
        relpath: str,
        before: Sequence[str] = (),
        table: Sequence[Sequence[str]] = (),
        after: Sequence[str] = (),
    ) -> Path:
        path = workspace / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        document = Document()
        for text in before:
            document.add_paragraph(text)
        if table:
            grid = document.add_table(rows=0, cols=len(table[0]))
            for row in table:
                for cell, text in zip(grid.add_row().cells, row, strict=True):
                    cell.text = text
        for text in after:
            document.add_paragraph(text)
        document.save(str(path))
        return path

    return build
