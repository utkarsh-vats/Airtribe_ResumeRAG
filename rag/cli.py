"""Shared command-line plumbing for ``resume_rag.py`` and ``job_matcher.py``.

Both CLIs follow the same conventions: ``.env`` is loaded without overriding real environment
variables; settings come from flags, then the environment, then defaults (``WORKSPACE_ROOT``
./data, ``CHROMA_DIR`` ./.chroma, ``EMBEDDER`` huggingface, ``AS_OF_DATE`` blank = today);
usage and configuration errors raise ``UsageError`` and become one line on stderr with exit
code 2. Rendering lives in ``rag.console``.
"""

import argparse
import os
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

import fs_tools
from rag.chunking import Chunker, available_chunkers, get_chunker
from rag.embedders import available_embedders, from_env, get_embedder
from rag.embedders.base import Embedder
from rag.indexer import DEFAULT_WORKSPACE_ROOT
from rag.jd_parser import JD_EXTENSIONS, JobDescriptionError, load_job_description
from rag.store import Sidecar, collection_name, read_sidecar

DEFAULT_CHROMA_DIR = "./.chroma"
DEFAULT_JD_DIRECTORY = "job_descriptions"
CHUNKER_DEFAULT = "section"


class UsageError(Exception):
    """A usage or configuration problem: reported in one line, exit code 2."""


class ArgumentParser(argparse.ArgumentParser):
    """An argparse parser whose errors raise ``UsageError`` instead of exiting."""

    def error(self, message: str) -> None:
        """Raise instead of printing usage and exiting."""
        raise UsageError(message)


def add_configuration(sub: argparse.ArgumentParser) -> None:
    """``--chunker``, ``--chunker-opt``, ``--embedder`` and ``--model``."""
    sub.add_argument("--chunker", choices=available_chunkers(), default=CHUNKER_DEFAULT)
    sub.add_argument(
        "--chunker-opt",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="chunker parameter, e.g. max_words=200 or size=200 (repeatable)",
    )
    sub.add_argument("--embedder", choices=available_embedders(), help="default: $EMBEDDER")
    sub.add_argument("--model", help="embedding model id (huggingface or gemini)")


def prepare_environment(root: Path, load_env: bool = True) -> None:
    """Load ``root/.env`` (real variables win) and default ``WORKSPACE_ROOT`` to ./data."""
    if load_env:
        load_dotenv(root / ".env", override=False)
    if not setting("WORKSPACE_ROOT"):
        os.environ["WORKSPACE_ROOT"] = DEFAULT_WORKSPACE_ROOT


def setting(name: str, default: str = "") -> str:
    """An environment variable, stripped; blank counts as unset."""
    return (os.environ.get(name) or "").strip() or default


def resolve_as_of(flag: str | None, today: Callable[[], date]) -> date:
    """``--as-of``, else ``AS_OF_DATE``, else ``today()``."""
    text = (flag or "").strip() or setting("AS_OF_DATE")
    if not text:
        return today()
    try:
        return date.fromisoformat(text)
    except ValueError:
        raise UsageError(f"invalid as-of date {text!r}; expected YYYY-MM-DD") from None


def _option_value(text: str) -> int | bool | str:
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    try:
        return int(text)
    except ValueError:
        return text


def make_chunker(name: str, options: Sequence[str]) -> Chunker:
    """A chunker from ``--chunker`` and its ``--chunker-opt KEY=VALUE`` options."""
    params = {}
    for option in options:
        key, sep, value = option.partition("=")
        if not sep or not key.strip():
            raise UsageError(f"--chunker-opt expects KEY=VALUE, got {option!r}")
        params[key.strip().replace("-", "_")] = _option_value(value.strip())
    try:
        return get_chunker(name, **params)
    except TypeError as exc:
        raise UsageError(f"bad option for the {name} chunker: {exc}") from None
    except ValueError as exc:
        raise UsageError(f"{name} chunker: {exc}") from None


def make_embedder(name: str | None, model: str | None) -> Embedder:
    """``--embedder`` / ``--model`` over ``EMBEDDER`` and the backend's env settings."""
    env = dict(os.environ)
    if name:
        env["EMBEDDER"] = name
    chosen = (env.get("EMBEDDER") or "").strip() or "huggingface"
    if chosen not in available_embedders():
        raise UsageError(f"unknown embedder {chosen!r}; available: {available_embedders()}")
    if model:
        env["HF_EMBEDDING_MODEL" if chosen == "huggingface" else "GEMINI_EMBEDDING_MODEL"] = model
    if chosen == "fake":
        return get_embedder("fake")
    try:
        return from_env(env)
    except ValueError as exc:  # includes GeminiConfigError
        raise UsageError(str(exc)) from None


def chroma_dir() -> Path:
    """``CHROMA_DIR`` or ``./.chroma``."""
    return Path(setting("CHROMA_DIR", DEFAULT_CHROMA_DIR))


def require_index(chunker: Chunker, embedder: Embedder) -> tuple[str, Sidecar]:
    """The collection name and sidecar for this configuration; ``UsageError`` if not indexed."""
    name = collection_name(chunker.chunker_id, embedder.embedder_id)
    sidecar = read_sidecar(chroma_dir(), name)
    if sidecar is None:
        raise UsageError(f"no index for collection {name!r} in {chroma_dir()}; run index first")
    return name, sidecar


def job_descriptions_in(directory: str) -> dict[str, str]:
    """``{workspace path: text}`` for every .txt/.pdf/.docx JD in a workspace folder."""
    jobs = {}
    for entry in sorted(fs_tools.list_files(directory), key=lambda e: e["path"]):
        if entry["extension"] in JD_EXTENSIONS:
            try:
                text, origin = load_job_description(entry["path"])
            except JobDescriptionError as exc:
                raise UsageError(f"{entry['path']}: {exc}") from None
            jobs[origin] = text
    if not jobs:
        raise UsageError(f"no .txt, .pdf or .docx job descriptions in {directory!r}")
    return jobs
