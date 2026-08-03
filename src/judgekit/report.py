"""Render a reliability report a reviewer can act on."""

from __future__ import annotations

from collections.abc import Sequence

from .agreement import interpret_kappa
from .decompose import AMBIGUOUS, CLEAN, UNDERSPECIFIED, UNSTABLE, Decomposition
from .probes import ProbeResult

__all__ = ["markdown_report"]

_NEXT_STEP = {
    AMBIGUOUS: (
        "Most of your disagreement is **item ambiguity**. The rubric is not the "
        "problem. Add an explicit abstain/uncertain label, route those items to a "
        "human, and report them as a named subset rather than scoring them."
    ),
    UNDERSPECIFIED: (
        "Most of your disagreement is **rubric underspecification**. The items are "
        "fine; your judges each hold a different stable reading of the criterion. "
        "Find the disputed clause, rewrite it with a boundary example on each side, "
        "and re-run. Adding more judges will not help."
    ),
    UNSTABLE: (
        "At least one judge cannot reproduce its own verdict. Fix the judge before "
        "interpreting anything else: lower temperature, remove non-determinism from "
        "the prompt, or replace the model."
    ),
    CLEAN: "Judges are stable and they agree. Aggregation is defensible.",
}


def markdown_report(
    decomposition: Decomposition,
    probes: Sequence[ProbeResult] = (),
    title: str = "Judge panel reliability report",
    max_items: int = 20,
) -> str:
    d = decomposition
    counts = d.counts
    n = len(d.items) or 1

    out: list[str] = [f"# {title}", ""]
    out += [
        "## Headline",
        "",
        f"- **Inter-judge Fleiss kappa:** {d.inter_judge_kappa:.3f} - {interpret_kappa(d.inter_judge_kappa)}",
        f"- **Mean intra-judge self-consistency:** {d.mean_self_consistency:.3f}",
        f"- **Dominant cause of disagreement:** `{d.dominant_cause}`",
        "",
        _NEXT_STEP[d.dominant_cause],
        "",
    ]

    out += [
        "## Where the disagreement comes from",
        "",
        "| cause | items | share | what it means |",
        "| --- | ---: | ---: | --- |",
        f"| item ambiguity | {counts[AMBIGUOUS]} | {counts[AMBIGUOUS] / n:.0%} | judges cannot reproduce themselves |",
        f"| rubric underspecified | {counts[UNDERSPECIFIED]} | {counts[UNDERSPECIFIED] / n:.0%} | judges are each stable, and stably different |",
        f"| judge instability | {counts[UNSTABLE]} | {counts[UNSTABLE] / n:.0%} | a judge is not deterministic enough to score with |",
        f"| clean | {counts[CLEAN]} | {counts[CLEAN] / n:.0%} | stable and agreeing |",
        "",
    ]

    out += ["## Per-judge self-consistency", "", "| judge | self-consistency |", "| --- | ---: |"]
    for judge, score in sorted(d.intra_judge_agreement.items()):
        out.append(f"| `{judge}` | {score:.3f} |")
    out.append("")

    contested = [i for i in d.items if i.verdict != CLEAN]
    if contested:
        out += [
            f"## Contested items ({len(contested)} total, showing up to {max_items})",
            "",
            "| item | verdict | self-consistency | consensus | action |",
            "| --- | --- | ---: | ---: | --- |",
        ]
        for item in contested[:max_items]:
            out.append(
                f"| `{item.item_id}` | {item.verdict} | {item.self_consistency:.2f} "
                f"| {item.cross_judge_consensus:.2f} | {item.actionable} |"
            )
        out.append("")

    if probes:
        out += ["## Bias probes", "", "| probe | effect | flagged | detail |", "| --- | ---: | :---: | --- |"]
        for p in probes:
            out.append(f"| {p.name} | {p.effect:+.3f} | {'YES' if p.flagged else 'no'} | {p.detail} |")
        out.append("")

    out += [
        "## Method note",
        "",
        (
            "Ambiguity and underspecification are separated by **replication**: each "
            "judge scores each item more than once, independently. Without replicates "
            "the two causes are mathematically indistinguishable, which is why a single "
            "kappa cannot tell you which fix to apply."
        ),
        "",
    ]
    return "\n".join(out)
