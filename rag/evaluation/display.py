"""Plain rendering for notebooks: pandas DataFrames or plain-text tables, never rich.

Nothing here emits ANSI escape codes, so notebook outputs stay clean in any viewer and in
version control. pandas is imported lazily (it is a dev dependency, not a runtime one).
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def to_frame(
    rows: Iterable[Mapping[str, Any]], columns: Sequence[str] | None = None, digits: int = 3
) -> Any:
    """A DataFrame of ``rows`` (optionally only ``columns``), floats rounded to ``digits``."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - pandas is a dev dependency
        raise ImportError("to_frame() needs pandas (pip install pandas)") from exc
    frame = pd.DataFrame([dict(r) for r in rows])
    if columns is not None:
        frame = frame[list(columns)]
    return frame.round(digits)


def _cell(value: Any, digits: int) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if isinstance(value, list | tuple):
        return ", ".join(str(v) for v in value)
    return str(value)


def text_table(
    rows: Iterable[Mapping[str, Any]], columns: Sequence[str] | None = None, digits: int = 3
) -> str:
    """A plain-text table: header, dashes, then one line per row; numbers right-aligned."""
    rows = [dict(r) for r in rows]
    if not rows:
        return "(no rows)"
    columns = list(columns) if columns is not None else list(rows[0])
    cells = [[_cell(row.get(c), digits) for c in columns] for row in rows]
    numeric = [
        all(isinstance(row.get(c), int | float) or row.get(c) is None for row in rows)
        for c in columns
    ]
    widths = [max(len(c), *(len(line[i]) for line in cells)) for i, c in enumerate(columns)]

    def fmt(values: Sequence[str]) -> str:
        return "  ".join(
            v.rjust(w) if is_num else v.ljust(w)
            for v, w, is_num in zip(values, widths, numeric, strict=True)
        ).rstrip()

    lines = [fmt(columns), "  ".join("-" * w for w in widths)]
    lines += [fmt(line) for line in cells]
    return "\n".join(lines)


def pivot_text(
    rows: Iterable[Mapping[str, Any]], index: str, column: str, value: str, digits: int = 3
) -> str:
    """A plain-text pivot: one line per ``index`` value, one column per ``column`` value."""
    rows = list(rows)
    columns = list(dict.fromkeys(r[column] for r in rows))
    table: dict[Any, dict[str, Any]] = {}
    for r in rows:
        table.setdefault(r[index], {index: r[index]})[str(r[column])] = r[value]
    return text_table(table.values(), [index, *(str(c) for c in columns)], digits)
