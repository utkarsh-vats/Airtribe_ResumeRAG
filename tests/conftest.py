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


@pytest.fixture(scope="session")
def fake_corpus(tmp_path_factory):
    """All 35 resumes indexed once with FakeEmbedder (section chunker), plus a skills index.

    Shared by several modules; tests must not write to its store or sidecar.
    """
    from datetime import date
    from types import SimpleNamespace

    from rag.chunking import SectionChunker
    from rag.embedders.fake import FakeEmbedder
    from rag.indexer import index_directory
    from rag.keyword_index import KeywordIndex
    from rag.store import VectorStore
    from scripts.persona_schema import ROOT

    chunker, embedder = SectionChunker(), FakeEmbedder()
    store = VectorStore(
        tmp_path_factory.mktemp("fake_corpus") / "chroma",
        chunker.chunker_id,
        embedder.embedder_id,
        embedder.dim,
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("WORKSPACE_ROOT", str(ROOT / "data"))
        report = index_directory("resumes", chunker, embedder, store, date(2026, 11, 1))
    assert len(report.added) == 35 and not report.failed
    return SimpleNamespace(
        store=store,
        sidecar=store.load_sidecar(),
        embedder=embedder,
        keyword_index=KeywordIndex.from_store(store, mode="skills"),
    )
