"""Loading and validating the ground truth: ``qrels.json`` and ``manifest.json``.

Both are read from paths (default ``data/ground_truth/``) and validated before use, so a
malformed file fails with ``GroundTruthError`` naming the problem rather than producing
silently wrong metrics.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

GROUND_TRUTH_DIR = Path(__file__).resolve().parents[2] / "data" / "ground_truth"
QRELS_PATH = GROUND_TRUTH_DIR / "qrels.json"
MANIFEST_PATH = GROUND_TRUTH_DIR / "manifest.json"
LABEL_GRADES = {"S": 2, "P": 1, "D": 0, "none": 0}


class GroundTruthError(ValueError):
    """A ground-truth file is missing or malformed."""


def _require(obj: Any, keys: set[str], where: str) -> dict:
    if not isinstance(obj, dict):
        raise GroundTruthError(f"{where}: expected an object")
    missing = keys - set(obj)
    if missing:
        raise GroundTruthError(f"{where}: missing {sorted(missing)}")
    return obj


def _read(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise GroundTruthError(f"{path}: not found") from None
    except json.JSONDecodeError as exc:
        raise GroundTruthError(f"{path}: invalid JSON ({exc})") from None


# --- qrels --------------------------------------------------------------------------------


@dataclass(frozen=True)
class JDJudgements:
    """Every candidate's judgement for one JD."""

    jd_id: str
    title: str
    jd_path: str  # workspace-relative, e.g. "job_descriptions/python_backend.txt"
    grades: dict[str, int]
    labels: dict[str, str]  # "S", "P", "D" or "none"
    eligible: dict[str, bool]
    unmet: dict[str, tuple[str, ...]]

    def with_label(self, label: str) -> set[str]:
        """Candidate ids with this label."""
        return {cid for cid, value in self.labels.items() if value == label}


@dataclass(frozen=True)
class Qrels:
    """All JDs' judgements."""

    as_of: str
    jds: dict[str, JDJudgements]
    candidate_ids: tuple[str, ...] = field(default=())


def parse_qrels(data: Any, where: str = "qrels") -> Qrels:
    """Validate and wrap a qrels dict."""
    root = _require(data, {"as_of", "jds"}, where)
    if not isinstance(root["jds"], list) or not root["jds"]:
        raise GroundTruthError(f"{where}.jds: expected a non-empty list")
    jds: dict[str, JDJudgements] = {}
    candidates: set[str] | None = None
    for i, raw in enumerate(root["jds"]):
        at = f"{where}.jds[{i}]"
        jd = _require(raw, {"jd_id", "title", "jd_path", "judgements"}, at)
        if jd["jd_id"] in jds:
            raise GroundTruthError(f"{at}: duplicate jd_id {jd['jd_id']!r}")
        grades, labels, eligible, unmet = {}, {}, {}, {}
        for j, item in enumerate(jd["judgements"]):
            jat = f"{at}.judgements[{j}]"
            entry = _require(item, {"candidate_id", "label", "grade", "eligible", "unmet"}, jat)
            cid, label = entry["candidate_id"], entry["label"]
            if cid in labels:
                raise GroundTruthError(f"{jat}: duplicate candidate {cid!r}")
            if label not in LABEL_GRADES:
                raise GroundTruthError(f"{jat}: unknown label {label!r}")
            if entry["grade"] != LABEL_GRADES[label]:
                raise GroundTruthError(
                    f"{jat}: grade {entry['grade']!r} does not match label {label!r}"
                )
            if not isinstance(entry["eligible"], bool):
                raise GroundTruthError(f"{jat}: eligible must be true or false")
            if not isinstance(entry["unmet"], list) or entry["eligible"] == bool(entry["unmet"]):
                raise GroundTruthError(f"{jat}: unmet must be a list, empty exactly when eligible")
            grades[cid], labels[cid] = entry["grade"], label
            eligible[cid], unmet[cid] = entry["eligible"], tuple(entry["unmet"])
        if candidates is None:
            candidates = set(labels)
        elif set(labels) != candidates:
            raise GroundTruthError(f"{at}: judges a different set of candidates")
        jds[jd["jd_id"]] = JDJudgements(
            jd["jd_id"], jd["title"], jd["jd_path"], grades, labels, eligible, unmet
        )
    return Qrels(str(root["as_of"]), jds, tuple(sorted(candidates or ())))


def load_qrels(path: str | Path = QRELS_PATH) -> Qrels:
    """Read and validate ``qrels.json``."""
    return parse_qrels(_read(Path(path)), str(path))


# --- manifest -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ManifestCandidate:
    """What a resume says, as generated (the extraction target)."""

    candidate_id: str
    name: str
    resume_path: str
    skills: tuple[str, ...]
    skill_years: dict[str, float]
    total_years: float
    highest_education: dict | None  # degree, field, institution, end_year


@dataclass(frozen=True)
class Manifest:
    """All candidates in the manifest."""

    as_of: str
    candidates: dict[str, ManifestCandidate]


def parse_manifest(data: Any, where: str = "manifest") -> Manifest:
    """Validate and wrap a manifest dict."""
    root = _require(data, {"as_of", "candidates"}, where)
    if not isinstance(root["candidates"], list) or not root["candidates"]:
        raise GroundTruthError(f"{where}.candidates: expected a non-empty list")
    keys = {"candidate_id", "name", "resume_path", "skills", "skill_years", "total_years"}
    candidates = {}
    for i, raw in enumerate(root["candidates"]):
        at = f"{where}.candidates[{i}]"
        item = _require(raw, keys | {"highest_education"}, at)
        cid = item["candidate_id"]
        if cid in candidates:
            raise GroundTruthError(f"{at}: duplicate candidate {cid!r}")
        if set(item["skill_years"]) != set(item["skills"]):
            raise GroundTruthError(f"{at}: skill_years keys differ from skills")
        if not isinstance(item["total_years"], int | float):
            raise GroundTruthError(f"{at}: total_years must be a number")
        education = item["highest_education"]
        if education is not None:
            _require(education, {"degree", "field", "institution", "end_year"}, at)
        candidates[cid] = ManifestCandidate(
            cid,
            item["name"],
            item["resume_path"],
            tuple(item["skills"]),
            {k: float(v) for k, v in item["skill_years"].items()},
            float(item["total_years"]),
            education,
        )
    return Manifest(str(root["as_of"]), candidates)


def load_manifest(path: str | Path = MANIFEST_PATH) -> Manifest:
    """Read and validate ``manifest.json``."""
    return parse_manifest(_read(Path(path)), str(path))
