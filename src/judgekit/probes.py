"""Bias probes for LLM judges.

Each probe is a controlled perturbation: change exactly one thing that should
not matter, re-score, and measure how much the verdict moved. A judge whose
score moves when only the *presentation* changed is measuring presentation.

The probes are deliberately model-agnostic. You pass in a ``score_fn`` that
takes a prompt-shaped record and returns a numeric score or label; the probe
handles the perturbation, the pairing, and the statistics.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "ProbeResult",
    "position_bias",
    "scale_direction_bias",
    "self_enhancement_bias",
    "verbosity_bias",
]


@dataclass(frozen=True)
class ProbeResult:
    name: str
    effect: float
    n: int
    detail: str

    @property
    def flagged(self) -> bool:
        return abs(self.effect) >= self.threshold

    @property
    def threshold(self) -> float:
        return {
            "position": 0.05,
            "verbosity": 0.10,
            "self_enhancement": 0.05,
            "scale_direction": 0.10,
        }.get(self.name, 0.05)

    def __str__(self) -> str:
        mark = "FLAG" if self.flagged else "ok  "
        return f"[{mark}] {self.name:<17} effect={self.effect:+.3f} (n={self.n})  {self.detail}"


def position_bias(
    pairs: Sequence[tuple[str, str]],
    compare_fn: Callable[[str, str], int],
) -> ProbeResult:
    """Does the judge prefer whichever answer it sees FIRST?

    Each pair is scored twice, in both orders. ``compare_fn(a, b)`` returns
    ``1`` if a wins, ``-1`` if b wins, ``0`` for a tie. A consistent judge
    should flip its answer when the order flips.

    ``effect`` is the rate of order-induced position preference minus the rate
    expected by chance: 0.0 is clean, positive means first-slot preference.
    """
    if not pairs:
        raise ValueError("no pairs supplied")
    first_slot_wins = 0
    decisive = 0
    for a, b in pairs:
        forward = compare_fn(a, b)
        reverse = compare_fn(b, a)
        if forward == 0 or reverse == 0:
            continue
        decisive += 1
        # forward=1 means A (first) won; reverse=1 means B (now first) won.
        if forward == 1:
            first_slot_wins += 1
        if reverse == 1:
            first_slot_wins += 1
    if decisive == 0:
        return ProbeResult("position", float("nan"), 0, "all comparisons were ties")
    rate = first_slot_wins / (2 * decisive)
    return ProbeResult(
        "position",
        rate - 0.5,
        decisive,
        f"first-slot win rate {rate:.1%} against a 50% null",
    )


def verbosity_bias(
    records: Sequence[dict],
    score_fn: Callable[[dict], float],
    padding: str = (
        "\n\nTo restate the above in different words, and to be thorough about "
        "the reasoning: the same conclusion follows for the same reasons."
    ),
    repeats: int = 3,
) -> ProbeResult:
    """Does the score rise when the answer is padded with content-free text?

    The padding adds length and adds no information. Any positive effect is
    the judge rewarding length.

    ``effect`` is the mean score change, normalised by the observed score range.
    """
    if not records:
        raise ValueError("no records supplied")
    deltas = []
    for record in records:
        base = score_fn(record)
        padded = dict(record)
        text_key = "answer" if "answer" in record else "text"
        padded[text_key] = str(record[text_key]) + padding * repeats
        deltas.append(score_fn(padded) - base)
    baseline = [score_fn(r) for r in records]
    spread = float(np.ptp(baseline)) or 1.0
    effect = float(np.mean(deltas)) / spread
    return ProbeResult(
        "verbosity",
        effect,
        len(records),
        f"mean score change {np.mean(deltas):+.3f} over a {spread:.3f} score range",
    )


def self_enhancement_bias(
    records: Sequence[dict],
    score_fn: Callable[[dict], float],
    author_key: str = "author_model",
    judge_identity: str | None = None,
) -> ProbeResult:
    """Does the judge score answers written by ITSELF higher?

    Requires records tagged with the model that produced them. ``effect`` is
    the mean score for own-authored answers minus the mean for others.
    """
    if judge_identity is None:
        raise ValueError("judge_identity is required to detect self-enhancement")
    own = [score_fn(r) for r in records if r.get(author_key) == judge_identity]
    other = [score_fn(r) for r in records if r.get(author_key) != judge_identity]
    if not own or not other:
        return ProbeResult(
            "self_enhancement",
            float("nan"),
            len(records),
            "need answers both from and not from the judge model",
        )
    effect = float(np.mean(own) - np.mean(other))
    return ProbeResult(
        "self_enhancement",
        effect,
        len(own) + len(other),
        f"own={np.mean(own):.3f} vs other={np.mean(other):.3f} (n_own={len(own)})",
    )


def scale_direction_bias(
    scores: Mapping[str, tuple[float, float]],
    scale_min: float,
    scale_max: float,
) -> ProbeResult:
    """Does the score follow the rubric's meaning, or the size of the number?

    Works on recorded scores and calls no model. ``scores[item_id]`` is
    ``(normal, reversed)``: the judge's score under the rubric as written, and
    its score under the same rubric with the scale reversed (the best anchor
    now carries ``scale_min`` instead of ``scale_max``). The reversed score is
    mapped back with ``scale_max + scale_min - reversed``. A judge that follows
    the rubric's meaning gives the same mapped score; a judge that favours the
    top (or bottom) number whatever it means does not.

    ``effect`` is the mean absolute gap between the normal and mapped-back
    scores, divided by the scale width ``scale_max - scale_min``. It is flagged
    at 0.10 or more, a tenth of the scale (0.4 points on a 1 to 5 scale).

    After the scale-reversal probe in Kim et al., "Rubric Semantics or Numerical
    Priors? Auditing Scale-Direction Bias in LLM Judges", arXiv:2611.25588. The
    paper's floor normalisation against each judge's own run-to-run noise is not
    applied here.
    """
    if not scores:
        raise ValueError("no scores supplied")
    if not scale_max > scale_min:
        raise ValueError("scale_max must be greater than scale_min")
    width = float(scale_max - scale_min)
    gaps = []
    for item_id, pair in scores.items():
        if len(pair) != 2:
            raise ValueError(f"item {item_id!r}: need (normal, reversed) scores")
        normal, reversed_score = (float(v) for v in pair)
        for value in (normal, reversed_score):
            if not np.isfinite(value) or not scale_min <= value <= scale_max:
                raise ValueError(
                    f"item {item_id!r}: score {value} is outside {scale_min} to {scale_max}"
                )
        mapped_back = scale_max + scale_min - reversed_score
        gaps.append(abs(normal - mapped_back))
    mean_gap = float(np.mean(gaps))
    return ProbeResult(
        "scale_direction",
        mean_gap / width,
        len(gaps),
        f"mean gap {mean_gap:.3f} points after mapping the reversed scale back, "
        f"on a {scale_min:g} to {scale_max:g} scale",
    )
