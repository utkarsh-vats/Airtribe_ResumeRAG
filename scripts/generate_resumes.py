"""Render persona resumes into ``data/resumes/`` (``python -m scripts.generate_resumes``).

Each persona is rendered in its own layout and format to ``data/resumes/<id>.<ext>``, the
``resume_path`` in ``manifest.json`` (relative to ``data/``). PDF rendering arrives in step 3b;
until then PDF personas are skipped with a message.
"""

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path

from scripts.layouts import LAYOUT_FUNCTIONS
from scripts.persona_schema import ROOT, Persona, ResumeFormat, load_personas
from scripts.render_docx import write_docx
from scripts.render_txt import write_txt

DATA_DIR = ROOT / "data"
RESUME_DIR = DATA_DIR / "resumes"
SUPPORTED = frozenset({ResumeFormat.TXT, ResumeFormat.DOCX})


def resume_filename(persona: Persona) -> str:
    return f"{persona.id}.{persona.render.format.value}"


def render(persona: Persona, out_dir: Path) -> Path:
    """Write ``persona``'s resume into ``out_dir`` and return its path."""
    fmt = persona.render.format
    if fmt not in SUPPORTED:
        raise ValueError(f"{persona.id}: {fmt} rendering is not implemented yet (step 3b)")
    blocks = LAYOUT_FUNCTIONS[persona.render.layout](persona)
    path = out_dir / resume_filename(persona)
    if fmt is ResumeFormat.TXT:
        write_txt(blocks, path)
    else:
        write_docx(blocks, path, author=persona.name, title=f"{persona.name} - Resume")
    return path


def generate(personas: Iterable[Persona], out_dir: Path) -> tuple[list[Path], list[str]]:
    """Render every supported persona; return ``(written paths, skipped ids)``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written, skipped = [], []
    for persona in personas:
        if persona.render.format in SUPPORTED:
            written.append(render(persona, out_dir))
        else:
            skipped.append(persona.id)
    return written, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", action="append", metavar="ID", help="render only this persona")
    parser.add_argument("--out-dir", type=Path, default=RESUME_DIR)
    args = parser.parse_args(argv)
    personas = load_personas()
    if args.only:
        known = {p.id for p in personas}
        unknown = sorted(set(args.only) - known)
        if unknown:
            parser.error(f"unknown persona id(s): {', '.join(unknown)}")
        personas = [p for p in personas if p.id in args.only]
    written, skipped = generate(personas, args.out_dir)
    for path in written:
        print(f"Wrote {path}")
    if skipped:
        print(
            f"Skipped {len(skipped)} PDF persona(s); PDF rendering arrives in step 3b: "
            + ", ".join(skipped),
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
