"""Ranking metrics over a ranked list of candidate ids and graded relevance.

Grades follow ``qrels.json``: 2 strong (S), 1 partial (P), 0 distractor (D) or none.
Binary metrics take a relevance **threshold**: ``STRICT`` (grade 2, S only) or ``LENIENT``
(grade >= 1, S and P).

Edge cases, the same for every function:

- **No relevant candidates** for a JD (at the threshold, or an all-zero ideal DCG): the metric
  is ``None``, meaning undefined. ``macro_average`` skips ``None``, so such a JD is never
  counted as 0 or 1.
- **k beyond the list length**: positions past the end of the ranking count as non-relevant.
  Precision and intrusion still divide by k, so a list of 3 cannot have P@10 above 0.3.
- **Duplicate ids** in a ranking raise ``ValueError``, and so does ``k < 1``.
- Ids in the ranking without a grade count as grade 0.
"""

import math
from collections.abc import Iterable, Mapping, Sequence

STRICT = 2
LENIENT = 1
DISTRACTOR = "D"


def _check(ranking: Sequence[str], k: int | None = None) -> None:
    if len(set(ranking)) != len(ranking):
        seen: set[str] = set()
        dupes = sorted({c for c in ranking if c in seen or seen.add(c)})
        raise ValueError(f"duplicate candidate ids in ranking: {dupes}")
    if k is not None and k < 1:
        raise ValueError(f"k must be >= 1, not {k}")


def relevant(grades: Mapping[str, int], threshold: int) -> set[str]:
    """Candidates with ``grade >= threshold``."""
    return {cid for cid, grade in grades.items() if grade >= threshold}


def precision_at_k(
    ranking: Sequence[str], grades: Mapping[str, int], k: int, threshold: int = STRICT
) -> float | None:
    """Relevant candidates in the top k, divided by k."""
    _check(ranking, k)
    rel = relevant(grades, threshold)
    if not rel:
        return None
    return sum(1 for cid in ranking[:k] if cid in rel) / k


def recall_at_k(
    ranking: Sequence[str], grades: Mapping[str, int], k: int, threshold: int = STRICT
) -> float | None:
    """Relevant candidates in the top k, divided by all relevant candidates."""
    _check(ranking, k)
    rel = relevant(grades, threshold)
    if not rel:
        return None
    return sum(1 for cid in ranking[:k] if cid in rel) / len(rel)


def r_precision(
    ranking: Sequence[str], grades: Mapping[str, int], threshold: int = STRICT
) -> float | None:
    """Precision at R, where R is the number of relevant candidates."""
    _check(ranking)
    rel = relevant(grades, threshold)
    if not rel:
        return None
    return precision_at_k(ranking, grades, len(rel), threshold)


def mrr(ranking: Sequence[str], grades: Mapping[str, int], threshold: int = STRICT) -> float | None:
    """1 / rank of the first relevant candidate (0.0 if none is ranked)."""
    _check(ranking)
    rel = relevant(grades, threshold)
    if not rel:
        return None
    for rank, cid in enumerate(ranking, start=1):
        if cid in rel:
            return 1.0 / rank
    return 0.0


def _dcg(gains: Iterable[float]) -> float:
    return sum(gain / math.log2(position + 1) for position, gain in enumerate(gains, start=1))


def ndcg_at_k(ranking: Sequence[str], grades: Mapping[str, int], k: int) -> float | None:
    """Graded nDCG@k: gain ``2**grade - 1``, discount ``log2(rank + 1)``, ideal from the qrels."""
    _check(ranking, k)
    ideal = _dcg(sorted((2**g - 1 for g in grades.values()), reverse=True)[:k])
    if ideal == 0:
        return None
    return _dcg(2 ** grades.get(cid, 0) - 1 for cid in ranking[:k]) / ideal


def distractor_intrusion_at_k(ranking: Sequence[str], labels: Mapping[str, str], k: int) -> float:
    """The share of the top k labelled D (divided by k; defined even with no relevant ones)."""
    _check(ranking, k)
    return sum(1 for cid in ranking[:k] if labels.get(cid) == DISTRACTOR) / k


def macro_average(values: Iterable[float | None]) -> float | None:
    """Mean of the defined values; None if none is defined."""
    defined = [v for v in values if v is not None]
    return sum(defined) / len(defined) if defined else None
