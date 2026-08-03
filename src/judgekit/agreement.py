"""Chance-corrected agreement statistics for judge panels.

Every function here answers one question: *how much of the agreement you see
would you have gotten from judges who were not reading the items at all?*
Raw percent agreement cannot answer it, which is why raw percent agreement is
the most over-reported and least informative number in evaluation write-ups.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Sequence
from itertools import combinations

import numpy as np

__all__ = [
    "cohen_kappa",
    "fleiss_kappa",
    "interpret_kappa",
    "krippendorff_alpha",
    "pairwise_kappas",
    "percent_agreement",
]


def _as_matrix(ratings: Sequence[Sequence[Hashable]]) -> list[list[Hashable]]:
    rows = [list(r) for r in ratings]
    if not rows:
        raise ValueError("ratings is empty")
    width = len(rows[0])
    if any(len(r) != width for r in rows):
        raise ValueError("all items must have the same number of ratings")
    return rows


def percent_agreement(ratings: Sequence[Sequence[Hashable]]) -> float:
    """Fraction of judge PAIRS that agree, averaged over items.

    Reported only as a foil: on an imbalanced task two judges who both always
    say "exclude" will score near 1.0 while carrying no information at all.
    """
    rows = _as_matrix(ratings)
    scores = []
    for row in rows:
        pairs = list(combinations(row, 2))
        if not pairs:
            continue
        scores.append(sum(a == b for a, b in pairs) / len(pairs))
    return float(np.mean(scores)) if scores else float("nan")


def cohen_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    """Cohen's kappa for exactly two judges rating the same items."""
    if len(a) != len(b):
        raise ValueError("both judges must rate the same number of items")
    n = len(a)
    if n == 0:
        raise ValueError("no items")
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum((ca[k] / n) * (cb[k] / n) for k in set(ca) | set(cb))
    if np.isclose(expected, 1.0):
        return 1.0
    return float((observed - expected) / (1 - expected))


def fleiss_kappa(ratings: Sequence[Sequence[Hashable]]) -> float:
    """Fleiss' kappa for a fixed number of judges rating every item.

    ``ratings[i][j]`` is judge j's label for item i.
    """
    rows = _as_matrix(ratings)
    n_items = len(rows)
    n_judges = len(rows[0])
    if n_judges < 2:
        raise ValueError("Fleiss' kappa needs at least two judges")

    categories = sorted({c for row in rows for c in row}, key=repr)
    index = {c: k for k, c in enumerate(categories)}
    counts = np.zeros((n_items, len(categories)), dtype=float)
    for i, row in enumerate(rows):
        for label in row:
            counts[i, index[label]] += 1

    p_item = (counts * (counts - 1)).sum(axis=1) / (n_judges * (n_judges - 1))
    p_bar = float(p_item.mean())
    p_cat = counts.sum(axis=0) / (n_items * n_judges)
    p_expected = float((p_cat ** 2).sum())
    if np.isclose(p_expected, 1.0):
        return 1.0
    return float((p_bar - p_expected) / (1 - p_expected))


def krippendorff_alpha(
    ratings: Sequence[Sequence[Hashable]], level: str = "nominal"
) -> float:
    """Krippendorff's alpha. Handles missing values encoded as ``None``.

    Use this instead of Fleiss when judges did not all rate every item, which
    is the normal case once you start sampling.
    """
    rows = [list(r) for r in ratings]
    values = sorted({v for row in rows for v in row if v is not None}, key=repr)
    if not values:
        raise ValueError("no observed ratings")
    idx = {v: k for k, v in enumerate(values)}

    if level == "nominal":
        def delta(x: int, y: int) -> float:
            return float(x != y)
    elif level == "interval":
        nums = [float(v) for v in values]  # type: ignore[arg-type]
        def delta(x: int, y: int) -> float:
            return (nums[x] - nums[y]) ** 2
    else:
        raise ValueError("level must be 'nominal' or 'interval'")

    observed_num = 0.0
    n_pairable = 0
    coincidence = Counter()
    for row in rows:
        present = [idx[v] for v in row if v is not None]
        m = len(present)
        if m < 2:
            continue
        n_pairable += m
        for x, y in combinations(present, 2):
            observed_num += 2 * delta(x, y) / (m - 1)
        for v in present:
            coincidence[v] += 1

    if n_pairable < 2:
        return float("nan")

    total = sum(coincidence.values())
    expected_num = 0.0
    for x in coincidence:
        for y in coincidence:
            if x == y:
                expected_num += coincidence[x] * (coincidence[y] - 1) * delta(x, y)
            else:
                expected_num += coincidence[x] * coincidence[y] * delta(x, y)
    expected = expected_num / (total - 1) if total > 1 else 0.0
    if np.isclose(expected, 0.0):
        return 1.0
    return float(1 - (observed_num / expected))


def pairwise_kappas(
    ratings: Sequence[Sequence[Hashable]], judge_names: Sequence[str] | None = None
) -> dict[tuple[str, str], float]:
    """Cohen's kappa for every judge pair.

    A panel mean hides the shape of the disagreement. Two judges agreeing at
    0.75 while both disagree with a third at 0.05 is a completely different
    finding from three judges all sitting at 0.28, and the panel mean is the
    same either way.
    """
    rows = _as_matrix(ratings)
    n_judges = len(rows[0])
    names = list(judge_names) if judge_names else [f"judge_{i}" for i in range(n_judges)]
    if len(names) != n_judges:
        raise ValueError("judge_names length does not match the rating matrix")
    out: dict[tuple[str, str], float] = {}
    for i, j in combinations(range(n_judges), 2):
        col_i = [row[i] for row in rows]
        col_j = [row[j] for row in rows]
        out[(names[i], names[j])] = cohen_kappa(col_i, col_j)
    return out


def interpret_kappa(kappa: float) -> str:
    """Landis & Koch bands, stated as what the number LICENSES you to claim."""
    if np.isnan(kappa):
        return "undefined"
    if kappa < 0.00:
        return "worse than chance - the panel is anti-correlated; check label polarity"
    if kappa < 0.20:
        return "slight - the panel is not measuring a shared construct; do not aggregate"
    if kappa < 0.40:
        return "fair - a shared construct may exist but the rubric is not carrying it"
    if kappa < 0.60:
        return "moderate - usable with a documented disagreement-resolution procedure"
    if kappa < 0.80:
        return "substantial - safe to aggregate; report the resolution rule anyway"
    return "almost perfect - check for a leaked cue or a degenerate label distribution"
