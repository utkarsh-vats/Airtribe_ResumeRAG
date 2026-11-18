"""Render resume blocks as plain text (UTF-8, LF line endings). Dataset tooling."""

from collections.abc import Iterable
from pathlib import Path

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


def _role_lines(role: RoleHeader) -> list[str]:
    company = f"{role.company}, {role.location}" if role.location else role.company
    if role.style is RoleStyle.INLINE:
        return [f"{role.title} | {company} | {role.date_range}"]
    return [role.title, f"{company} | {role.date_range}"]


def _lines(blocks: Iterable[Block]) -> list[str]:
    lines: list[str] = []
    for block in blocks:
        match block:
            case Heading(level=0):
                lines.append(block.text)
            case Heading():
                lines += ["", block.text]
                if block.underline:
                    lines.append(block.underline * len(block.text))
            case Paragraph():
                lines.append(block.text)
            case BulletList():
                lines += [f"{block.marker} {item}" for item in block.items]
            case RoleHeader():
                lines += _role_lines(block)
            case Table():
                lines += [" | ".join(row) for row in block.rows]
            case Spacer():
                lines.append("")
            case PageBreak():
                pass
            case Panel():
                inner = _lines(block.blocks)
                rule = "+" + "-" * (max(map(len, inner), default=0) + 2)
                lines += [rule, *(f"| {line}".rstrip() for line in inner), rule]
    return lines


def render_txt(blocks: Iterable[Block]) -> str:
    """The resume text: one LF-terminated line per line, no repeated blank lines."""
    out: list[str] = []
    for line in _lines(blocks):
        if line or (out and out[-1]):
            out.append(line)
    while out and not out[-1]:
        out.pop()
    return "\n".join(out) + "\n"


def write_txt(blocks: Iterable[Block], path: Path) -> None:
    path.write_bytes(render_txt(blocks).encode("utf-8"))
