"""Render persona resumes into ``data/resumes/`` (``python -m scripts.generate_resumes``).

Each persona is rendered in its own layout and format to ``data/resumes/<id>.<ext>``, the
``resume_path`` in ``manifest.json`` (relative to ``data/``). Output is byte-identical across
runs and platforms, so ``--check`` regenerates into a temporary directory and fails if any
committed file differs, is missing or is extra.
"""

import argparse
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path

from scripts.layouts import LAYOUT_FUNCTIONS
from scripts.persona_schema import ROOT, Persona, ResumeFormat, load_personas
from scripts.render_docx import write_docx
from scripts.render_pdf import write_pdf
from scripts.render_txt import write_txt

DATA_DIR = ROOT / "data"
RESUME_DIR = DATA_DIR / "resumes"


def resume_filename(persona: Persona) -> str:
    return f"{persona.id}.{persona.render.format.value}"


def render(persona: Persona, out_dir: Path) -> Path:
    """Write ``persona``'s resume into ``out_dir`` and return its path."""
    blocks = LAYOUT_FUNCTIONS[persona.render.layout](persona)
    path = out_dir / resume_filename(persona)
    meta = {"author": persona.name, "title": f"{persona.name} - Resume"}
    match persona.render.format:
        case ResumeFormat.TXT:
            write_txt(blocks, path)
        case ResumeFormat.DOCX:
            write_docx(blocks, path, **meta)
        case ResumeFormat.PDF:
            write_pdf(blocks, path, **meta)
    return path


def generate(personas: Iterable[Persona], out_dir: Path) -> list[Path]:
    """Render every persona into ``out_dir``; return the written paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    return [render(persona, out_dir) for persona in personas]


def compare(expected_dir: Path, actual_dir: Path) -> list[str]:
    """Problems with ``actual_dir`` against freshly generated ``expected_dir``, sorted."""
    expected = {p.name: p for p in expected_dir.iterdir() if p.is_file()}
    actual = {p.name: p for p in actual_dir.iterdir() if p.is_file()} if actual_dir.exists() else {}
    problems = [f"missing: {name}" for name in sorted(expected.keys() - actual.keys())]
    problems += [f"extra: {name}" for name in sorted(actual.keys() - expected.keys())]
    problems += [
        f"differs: {name}"
        for name in sorted(expected.keys() & actual.keys())
        if expected[name].read_bytes() != actual[name].read_bytes()
    ]
    return sorted(problems, key=lambda p: p.split(": ", 1)[1])


def check(personas: Iterable[Persona], resume_dir: Path) -> list[str]:
    """Regenerate into a temporary directory and compare it with ``resume_dir``."""
    with tempfile.TemporaryDirectory() as tmp:
        fresh = Path(tmp)
        generate(personas, fresh)
        return compare(fresh, resume_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", action="append", metavar="ID", help="render only this persona")
    parser.add_argument("--out-dir", type=Path, default=RESUME_DIR)
    parser.add_argument(
        "--check", action="store_true", help="exit 1 if --out-dir differs from a fresh generation"
    )
    args = parser.parse_args(argv)
    personas = load_personas()
    if args.only:
        if args.check:
            parser.error("--check compares the whole directory; it can't be combined with --only")
        known = {p.id for p in personas}
        unknown = sorted(set(args.only) - known)
        if unknown:
            parser.error(f"unknown persona id(s): {', '.join(unknown)}")
        personas = [p for p in personas if p.id in args.only]
    if args.check:
        problems = check(personas, args.out_dir)
        for problem in problems:
            print(problem, file=sys.stderr)
        if problems:
            print("Resumes are stale; run: python -m scripts.generate_resumes", file=sys.stderr)
            return 1
        print(f"{args.out_dir} is up to date")
        return 0
    for path in generate(personas, args.out_dir):
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
