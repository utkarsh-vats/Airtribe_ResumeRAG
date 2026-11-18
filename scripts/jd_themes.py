"""The six JD themes as structured requirements (dataset tooling, not runtime code).

``data/ground_truth/jd_themes.json`` records what each job description in step 4 *intends* to
require. The persona tests use it to keep the relevance labels in ``persona_plan.md``
consistent with the manifest. The runtime JD parser (step 10) never reads this file; it has
to recover the same requirements from the JD text.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag.skills import canonicalize
from scripts.persona_schema import GROUND_TRUTH_DIR

JD_THEMES_PATH = GROUND_TRUTH_DIR / "jd_themes.json"


class JDThemeError(ValueError):
    """``jd_themes.json`` does not match the expected structure."""


@dataclass(frozen=True)
class MustHave:
    any_of: tuple[str, ...]
    min_years: int | None

    def describe(self) -> str:
        skills = " or ".join(self.any_of)
        return skills if self.min_years is None else f"{self.min_years}+ yrs {skills}"


@dataclass(frozen=True)
class JDTheme:
    id: str
    title: str
    min_total_years: int | None
    must_have: tuple[MustHave, ...]
    nice_to_have: tuple[str, ...]


def _skill(value: Any, where: str) -> str:
    if not isinstance(value, str) or canonicalize(value) != value:
        raise JDThemeError(f"{where}: {value!r} is not a canonical skill name")
    return value


def _years(value: Any, where: str) -> int | None:
    if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
        raise JDThemeError(f"{where}: expected a non-negative integer or null")
    return value


def _keys(obj: Any, keys: set[str], where: str) -> dict:
    if not isinstance(obj, dict) or set(obj) != keys:
        raise JDThemeError(f"{where}: expected an object with keys {sorted(keys)}")
    return obj


def _theme(raw: Any, where: str) -> JDTheme:
    keys = {"id", "title", "min_total_years", "must_have", "nice_to_have"}
    obj = _keys(raw, keys, where)
    must_have = []
    for i, req in enumerate(obj["must_have"]):
        req = _keys(req, {"any_of", "min_years"}, f"{where}.must_have[{i}]")
        if not req["any_of"]:
            raise JDThemeError(f"{where}.must_have[{i}].any_of: must not be empty")
        any_of = tuple(_skill(s, f"{where}.must_have[{i}].any_of") for s in req["any_of"])
        must_have.append(MustHave(any_of, _years(req["min_years"], f"{where}.must_have[{i}]")))
    if not must_have:
        raise JDThemeError(f"{where}.must_have: must not be empty")
    return JDTheme(
        id=obj["id"],
        title=obj["title"],
        min_total_years=_years(obj["min_total_years"], f"{where}.min_total_years"),
        must_have=tuple(must_have),
        nice_to_have=tuple(_skill(s, f"{where}.nice_to_have") for s in obj["nice_to_have"]),
    )


def load_jd_themes(path: Path = JD_THEMES_PATH) -> list[JDTheme]:
    """Load and validate the themes, in file order."""
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, list):
        raise JDThemeError(f"{path.name}: expected a list of themes")
    themes = [_theme(item, f"{path.name}[{i}]") for i, item in enumerate(raw)]
    ids = [t.id for t in themes]
    if len(set(ids)) != len(ids):
        raise JDThemeError(f"{path.name}: duplicate theme ids")
    return themes


def unmet_must_haves(theme: JDTheme, candidate: dict) -> list[str]:
    """Must-haves (and the total-years minimum) a manifest candidate fails.

    A must-have is met when one skill in ``any_of`` is among the candidate's skills and, if
    ``min_years`` is set, that same skill has at least that many evidenced years.
    """
    unmet = []
    if theme.min_total_years is not None and candidate["total_years"] < theme.min_total_years:
        unmet.append(f"{theme.min_total_years}+ total yrs")
    for req in theme.must_have:
        if not any(
            skill in candidate["skills"]
            and (req.min_years is None or candidate["skill_years"][skill] >= req.min_years)
            for skill in req.any_of
        ):
            unmet.append(req.describe())
    return unmet


def unmet_details(theme: JDTheme, candidate: dict) -> list[str]:
    """``unmet_must_haves`` with the candidate's evidence, for humans reading ``qrels.json``.

    Same rule and order, e.g. ``"Python 5+ yrs (has 4.3)"``, ``"Django or FastAPI (missing)"``,
    ``"4+ total yrs (has 3.9)"``.
    """
    details = []
    if theme.min_total_years is not None and candidate["total_years"] < theme.min_total_years:
        details.append(f"{theme.min_total_years}+ total yrs (has {candidate['total_years']})")
    for req in theme.must_have:
        present = [s for s in req.any_of if s in candidate["skills"]]
        best = max((candidate["skill_years"][s] for s in present), default=None)
        skills = " or ".join(req.any_of)
        if req.min_years is None:
            if not present:
                details.append(f"{skills} (missing)")
        elif best is None:
            details.append(f"{skills} {req.min_years}+ yrs (missing)")
        elif best < req.min_years:
            details.append(f"{skills} {req.min_years}+ yrs (has {best})")
    return details
