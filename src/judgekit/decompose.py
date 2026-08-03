"""Split panel disagreement into *item ambiguity* and *rubric underspecification*.

This is the question a bare kappa cannot answer.

A low inter-judge kappa has two completely different causes that demand
opposite fixes:

* **Item ambiguity** - the item genuinely does not have one right label. The
  rubric is fine. The fix is to change the *task*: add an abstain option,
  route to a human, or drop the item from the scored set. Rewriting the rubric
  will not help and will produce false confidence.
* **Rubric underspecification** - each judge is internally consistent but has
  settled on a different, stable reading of the criterion. The items are fine.
  The fix is to change the *rubric*: pin down the disputed clause, add worked
  examples at the boundary. Adding judges or averaging harder will not help.

You cannot tell these apart from a single pass. The design change that makes
them separable is **replication**: have each judge score each item more than
once, independently (fresh context, and ideally shuffled option order).

Then:

* Low **intra**-judge agreement  -> the item moves the same judge around -> ambiguity.
* High intra-judge agreement but low **inter**-judge agreement -> each judge is
  stable and they are stably different -> underspecification.

``decompose`` reports both, plus a per-item verdict so you can act on the
specific rows rather than on a panel-level average.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from .agreement import fleiss_kappa

__all__ = ["AMBIGUOUS", "CLEAN", "UNDERSPECIFIED", "UNSTABLE", "Decomposition", "ItemVerdict", "decompose"]

AMBIGUOUS = "item_ambiguous"
UNDERSPECIFIED = "rubric_underspecified"
CLEAN = "clean"
UNSTABLE = "judge_unstable"

# Replications = ratings[item][judge] -> list of repeated labels for that judge.
Replicated = Mapping[str, Mapping[str, Sequence[Hashable]]]


@dataclass(frozen=True)
class ItemVerdict:
    item_id: str
    self_consistency: float
    cross_judge_consensus: float
    modal_labels: dict[str, Hashable]
    verdict: str

    @property
    def actionable(self) -> str:
        return {
            AMBIGUOUS: "Change the TASK: allow abstain, route to human, or exclude from scoring.",
            UNDERSPECIFIED: "Change the RUBRIC: pin the disputed clause, add a boundary example.",
            UNSTABLE: "Change the JUDGE: lower temperature, fix the prompt, or replace the model.",
            CLEAN: "No action - judges are stable and they agree.",
        }[self.verdict]


@dataclass
class Decomposition:
    intra_judge_agreement: dict[str, float]
    mean_self_consistency: float
    single_pass_kappa: float
    modal_kappa: float
    items: list[ItemVerdict] = field(default_factory=list)
    self_consistency_threshold: float = 0.8
    consensus_threshold: float = 0.6

    @property
    def inter_judge_kappa(self) -> float:
        """The headline number: what you would have measured WITHOUT replication.

        This is deliberately the single-pass kappa, because it is the number you
        already have and the one that sent you here.
        """
        return self.single_pass_kappa

    @property
    def majority_vote_inflation(self) -> float:
        """How much majority-voting over replicates flatters the panel.

        A large positive gap is a warning, not a win: it means your reported
        agreement is being manufactured by aggregation over a panel whose
        individual judgements are not reproducible. Report both numbers.
        """
        return self.modal_kappa - self.single_pass_kappa

    @property
    def counts(self) -> Counter:
        return Counter(item.verdict for item in self.items)

    @property
    def dominant_cause(self) -> str:
        contested = {k: v for k, v in self.counts.items() if k != CLEAN}
        if not contested:
            return CLEAN
        return max(contested, key=lambda k: contested[k])

    def summary(self) -> str:
        c = self.counts
        n = len(self.items) or 1
        lines = [
            f"inter-judge Fleiss kappa (single pass): {self.single_pass_kappa:.3f}",
            (
                f"  same panel after majority vote:       {self.modal_kappa:.3f}"
                f"   (inflation {self.majority_vote_inflation:+.3f})"
            ),
            f"mean intra-judge self-consistency:      {self.mean_self_consistency:.3f}",
            "",
            "disagreement attributed to:",
            f"  item ambiguity          {c[AMBIGUOUS]:>4}  ({c[AMBIGUOUS] / n:.0%})",
            f"  rubric underspecified   {c[UNDERSPECIFIED]:>4}  ({c[UNDERSPECIFIED] / n:.0%})",
            f"  judge instability       {c[UNSTABLE]:>4}  ({c[UNSTABLE] / n:.0%})",
            f"  clean                   {c[CLEAN]:>4}  ({c[CLEAN] / n:.0%})",
            "",
            f"dominant cause: {self.dominant_cause}",
        ]
        return "\n".join(lines)


def _modal(labels: Sequence[Hashable]) -> Hashable:
    return Counter(labels).most_common(1)[0][0]


def _self_consistency(labels: Sequence[Hashable]) -> float:
    """Fraction of within-judge replicate PAIRS that match."""
    pairs = list(combinations(labels, 2))
    if not pairs:
        return float("nan")
    return sum(a == b for a, b in pairs) / len(pairs)


def _consensus(labels: Sequence[Hashable]) -> float:
    """Fraction of cross-judge pairs that match on their modal labels."""
    pairs = list(combinations(labels, 2))
    if not pairs:
        return float("nan")
    return sum(a == b for a, b in pairs) / len(pairs)


def decompose(
    ratings: Replicated,
    self_consistency_threshold: float = 0.8,
    consensus_threshold: float = 0.6,
) -> Decomposition:
    """Attribute each item's disagreement to its cause.

    Parameters
    ----------
    ratings
        ``ratings[item_id][judge_id]`` -> sequence of repeated labels from that
        judge for that item. At least two judges, at least two replicates.
    self_consistency_threshold
        Below this, a judge is treated as not stable on that item.
    consensus_threshold
        Below this, the judges are treated as not agreeing on that item.

    Returns
    -------
    Decomposition
    """
    if not ratings:
        raise ValueError("no ratings supplied")

    judges = sorted({j for per_item in ratings.values() for j in per_item})
    if len(judges) < 2:
        raise ValueError("need at least two judges to attribute disagreement")

    replicate_counts = {
        len(labels) for per_item in ratings.values() for labels in per_item.values()
    }
    if max(replicate_counts) < 2:
        raise ValueError(
            "need at least two replicates per judge per item - that is the whole "
            "point: without replication, ambiguity and underspecification are "
            "mathematically indistinguishable"
        )

    per_judge_scores: dict[str, list[float]] = {j: [] for j in judges}
    verdicts: list[ItemVerdict] = []
    modal_matrix: list[list[Hashable]] = []
    first_pass_matrix: list[list[Hashable]] = []

    for item_id in sorted(ratings):
        per_item = ratings[item_id]
        modal = {j: _modal(per_item[j]) for j in judges if j in per_item}
        if len(modal) < 2:
            continue

        selfs = []
        for j, labels in per_item.items():
            sc = _self_consistency(labels)
            if not np.isnan(sc):
                per_judge_scores[j].append(sc)
                selfs.append(sc)
        item_self = float(np.mean(selfs)) if selfs else float("nan")
        item_consensus = _consensus(list(modal.values()))

        # ORDER MATTERS. Self-consistency is checked FIRST.
        #
        # If judges cannot reproduce themselves, their modal labels can still
        # coincide by chance - three judges each flipping a coin four times will
        # often share a modal label. Checking consensus first would score that
        # item "clean" and hide the instability completely. Agreement built on
        # coin flips is not agreement.
        if item_self < self_consistency_threshold:
            verdict = AMBIGUOUS
        elif item_consensus >= consensus_threshold:
            verdict = CLEAN
        else:
            # Every judge is stable, and they are stably different.
            verdict = UNDERSPECIFIED

        verdicts.append(
            ItemVerdict(
                item_id=item_id,
                self_consistency=item_self,
                cross_judge_consensus=item_consensus,
                modal_labels=dict(modal),
                verdict=verdict,
            )
        )
        modal_matrix.append([modal[j] for j in judges if j in modal])
        first_pass_matrix.append([per_item[j][0] for j in judges if j in per_item])

    intra = {
        j: float(np.mean(v)) if v else float("nan") for j, v in per_judge_scores.items()
    }
    # A judge who is unstable EVERYWHERE is a broken judge, not an ambiguous corpus.
    for j, score in intra.items():
        if not np.isnan(score) and score < self_consistency_threshold / 2:
            for k, item in enumerate(verdicts):
                if item.verdict == AMBIGUOUS:
                    verdicts[k] = ItemVerdict(
                        item.item_id,
                        item.self_consistency,
                        item.cross_judge_consensus,
                        item.modal_labels,
                        UNSTABLE,
                    )
            break

    usable_modal = [row for row in modal_matrix if len(row) == len(judges)]
    usable_first = [row for row in first_pass_matrix if len(row) == len(judges)]
    modal_kappa = fleiss_kappa(usable_modal) if len(usable_modal) >= 2 else float("nan")
    single_kappa = fleiss_kappa(usable_first) if len(usable_first) >= 2 else float("nan")
    mean_self = float(np.nanmean(list(intra.values()))) if intra else float("nan")

    return Decomposition(
        intra_judge_agreement=intra,
        mean_self_consistency=mean_self,
        single_pass_kappa=single_kappa,
        modal_kappa=modal_kappa,
        items=verdicts,
        self_consistency_threshold=self_consistency_threshold,
        consensus_threshold=consensus_threshold,
    )
