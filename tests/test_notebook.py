"""notebooks/experiments.ipynb as committed: read only, never executed here."""

import re

import nbformat
import pytest

from scripts.persona_schema import ROOT

NOTEBOOK = ROOT / "notebooks" / "experiments.ipynb"
ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]|/home/|/Users/")
FIGURE = re.compile(r"notebooks/figures/[\w.-]+\.png")
SECTIONS = (
    "## 1. Dataset overview",
    "## 2. Extraction accuracy",
    "## 3. Chunking",
    "## 4. Retrieval modes",
    "## 5. Keyword tokenizer modes",
    "## 6. Query source",
)


@pytest.fixture(scope="module")
def notebook():
    return nbformat.read(str(NOTEBOOK), as_version=4)


def _texts(output) -> list[str]:
    """Every textual payload of an output (images excluded)."""
    if output.output_type == "stream":
        return [output.text]
    if output.output_type == "error":
        return ["\n".join(output.traceback)]
    data = output.get("data", {})
    return [str(v) for k, v in data.items() if not k.startswith("image/")]


def _code(notebook):
    return [c for c in notebook.cells if c.cell_type == "code"]


def test_valid_nbformat(notebook):
    nbformat.validate(notebook)


def test_every_code_cell_ran_without_errors(notebook):
    for i, cell in enumerate(_code(notebook)):
        assert cell.outputs, f"code cell {i} has no outputs"
        assert all(o.output_type != "error" for o in cell.outputs), f"code cell {i} errored"


def test_execution_counts_are_sequential(notebook):
    assert [c.execution_count for c in _code(notebook)] == list(range(1, len(_code(notebook)) + 1))


def test_outputs_have_no_ansi_and_no_absolute_paths(notebook):
    for i, cell in enumerate(_code(notebook)):
        for output in cell.outputs:
            for text in _texts(output):
                assert "\x1b" not in text, f"ANSI escape in code cell {i}"
                found = ABSOLUTE_PATH.search(text)
                assert found is None, f"absolute path {found.group()!r} in code cell {i}"


def test_outputs_have_no_stderr_noise(notebook):
    for i, cell in enumerate(_code(notebook)):
        streams = [o for o in cell.outputs if o.output_type == "stream" and o.name == "stderr"]
        assert not streams, f"stderr in code cell {i}: {streams[0].text[:200]!r}"


def test_every_section_heading_is_present(notebook):
    markdown = "\n".join(c.source for c in notebook.cells if c.cell_type == "markdown")
    for heading in SECTIONS:
        assert heading in markdown


def test_findings_are_written(notebook):
    findings = [c.source for c in notebook.cells if c.source.startswith("### Findings")]
    assert len(findings) == len(SECTIONS)
    assert all("FINDINGS_" not in f and len(f) > 200 for f in findings)


def test_referenced_figures_exist(notebook):
    referenced = set()
    for cell in notebook.cells:
        referenced |= set(FIGURE.findall(cell.source))
        for output in cell.get("outputs", []):
            for text in _texts(output):
                referenced |= set(FIGURE.findall(text))
    assert referenced, "the notebook saves no figures"
    for figure in sorted(referenced):
        assert (ROOT / figure).is_file(), figure
