"""Plain rendering for notebooks: pandas DataFrames or plain-text tables, never rich.

Nothing here emits ANSI escape codes, so notebook outputs stay clean in any viewer and in
version control. pandas is imported lazily (it is a dev dependency, not a runtime one).
"""

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


def to_frame(
    rows: Iterable[Mapping[str, Any]], columns: Sequence[str] | None = None, digits: int = 3
) -> Any:
    """A DataFrame of ``rows`` (optionally only ``columns``), floats rounded to ``digits``."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - pandas is a dev dependency
        raise ImportError("to_frame() needs pandas (pip install pandas)") from exc
    rows = [dict(r) for r in rows]
    frame = pd.DataFrame(rows)
    if columns is not None:
        frame = frame[list(columns)]
    for column in frame.columns:  # ints with gaps stay ints (<NA>), not floats with NaN
        values = [r.get(column) for r in rows]
        present = [v for v in values if v is not None]
        if present and all(isinstance(v, int) and not isinstance(v, bool) for v in present):
            frame[column] = pd.array(values, dtype="Int64")
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


# --- Notebook setup and figures -----------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = REPO_ROOT / "notebooks" / "figures"
MODE_COLORS = {"semantic": "#4C72B0", "keyword": "#DD8452", "hybrid": "#55A868"}
COLLECTION_COLORS = {
    "section": "#4C72B0",
    "fixed": "#C44E52",
    "section-noctx": "#8172B3",
    "section-minilm": "#937860",
    "section-bge-base": "#DA8BC3",
    "section-gemini": "#64B5CD",
}
FALLBACK_COLORS = ("#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3")
QUIET_LOGGERS = (
    "sentence_transformers",
    "transformers",
    "huggingface_hub",
    "chromadb",
    "httpx",
    "urllib3",
    "rag",
)


def notebook_setup() -> None:
    """Quiet libraries, set pandas options and the one matplotlib style used throughout.

    Call before anything loads a model: it sets the Hugging Face environment variables that
    silence the Windows symlink warning, the progress bars and the Hub's log messages, and
    (unless already set) ``HF_HUB_OFFLINE=1``, since the notebook only uses models that
    ``scripts.build_experiment_collections`` has already downloaded.
    """
    import logging
    import os
    import warnings

    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["HF_HUB_VERBOSITY"] = "error"  # huggingface_hub resets its log level on import
    # The notebook only uses models the build script already downloaded: no network checks.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
    warnings.filterwarnings(
        "ignore", message=".*iscoroutinefunction.*", category=DeprecationWarning
    )
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
    for name in QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)

    import matplotlib
    import pandas as pd

    pd.set_option("display.max_columns", 30)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_colwidth", 120)
    pd.set_option("display.precision", 3)
    matplotlib.rcParams.update(
        {
            "figure.figsize": (9, 4.5),
            "figure.dpi": 100,
            "savefig.dpi": 110,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "legend.frameon": False,
            "font.size": 10,
        }
    )


def color_for(key: str, palette: Mapping[str, str], index: int = 0) -> str:
    """The fixed colour for ``key``, else a fallback by position."""
    return palette.get(key, FALLBACK_COLORS[index % len(FALLBACK_COLORS)])


def save_figure(fig: Any, section: str, name: str, directory: Path | None = None) -> str:
    """Save ``fig`` as ``<section>_<name>.png``; returns the repo-relative path (no drive)."""
    directory = directory or FIGURES_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{section}_{name}.png"
    fig.savefig(path)
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def _panels(rows: list[dict], panel: str | None) -> list[Any]:
    return list(dict.fromkeys(r[panel] for r in rows)) if panel else [None]


def bar_chart(
    rows: Iterable[Mapping[str, Any]],
    category: str,
    series: str,
    value: str,
    *,
    title: str,
    ylabel: str,
    palette: Mapping[str, str] | None = None,
    panel: str | None = None,
    ylim: tuple[float, float] | None = (0.0, 1.0),
    digits: int = 3,
) -> Any:
    """Grouped bars: ``category`` on x, one bar per ``series``, optional panels; annotated."""
    import matplotlib.pyplot as plt

    rows = [dict(r) for r in rows]
    panels = _panels(rows, panel)
    fig, axes = plt.subplots(1, len(panels), squeeze=False, sharey=True)
    fig.set_size_inches(max(5.0, 5.0 * len(panels)), 4.5)
    for ax, panel_value in zip(axes[0], panels, strict=True):
        subset = [r for r in rows if panel is None or r[panel] == panel_value]
        categories = list(dict.fromkeys(r[category] for r in subset))
        names = list(dict.fromkeys(r[series] for r in subset))
        width = 0.8 / max(1, len(names))
        for i, name in enumerate(names):
            lookup = {r[category]: r[value] for r in subset if r[series] == name}
            xs = [c + (i - (len(names) - 1) / 2) * width for c in range(len(categories))]
            ys = [lookup.get(c) or 0.0 for c in categories]
            bars = ax.bar(
                xs, ys, width, label=str(name), color=color_for(str(name), palette or {}, i)
            )
            ax.bar_label(bars, labels=[f"{y:.{digits}f}" for y in ys], fontsize=8, padding=2)
        ax.set_xticks(range(len(categories)), [str(c) for c in categories])
        ax.set_xlabel(category.replace("_", " "))
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1] * 1.08)  # headroom for the labels
        ax.set_title(f"{title} ({panel_value})" if panel else title)
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


def line_chart(
    rows: Iterable[Mapping[str, Any]],
    x: str,
    y: str,
    series: str,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    palette: Mapping[str, str] | None = None,
    panel: str | None = None,
    ylim: tuple[float, float] | None = (0.0, 1.0),
) -> Any:
    """One line per ``series``, optional side-by-side panels."""
    import matplotlib.pyplot as plt

    rows = [dict(r) for r in rows]
    panels = _panels(rows, panel)
    fig, axes = plt.subplots(1, len(panels), squeeze=False, sharey=True)
    fig.set_size_inches(max(5.0, 5.0 * len(panels)), 4.5)
    for ax, panel_value in zip(axes[0], panels, strict=True):
        subset = [r for r in rows if panel is None or r[panel] == panel_value]
        for i, name in enumerate(dict.fromkeys(r[series] for r in subset)):
            points = sorted((r[x], r[y]) for r in subset if r[series] == name)
            ax.plot(
                [p[0] for p in points],
                [p[1] for p in points],
                marker="o",
                markersize=3,
                label=str(name),
                color=color_for(str(name), palette or {}, i),
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_title(f"{title} ({panel_value})" if panel else title)
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


def histogram(
    values: Sequence[float], *, bins: int, title: str, xlabel: str, ylabel: str = "resumes"
) -> Any:
    """A single histogram."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.hist(list(values), bins=bins, color=FALLBACK_COLORS[0], edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig


def melt(
    rows: Iterable[Mapping[str, Any]],
    keys: Sequence[str],
    metrics: Sequence[str],
    metric_key: str = "metric",
    value_key: str = "value",
) -> list[dict]:
    """Wide metric rows to long ones: ``keys`` kept, one row per metric (for grouped bars)."""
    return [
        {k: row[k] for k in keys} | {metric_key: metric, value_key: row[metric]}
        for row in rows
        for metric in metrics
    ]
