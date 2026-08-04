"""Worked example: a three-judge labeling panel sitting at a low kappa.

Run:  python examples/worked_example.py

The panel below is a realistic caricature of a production LLM labeling
panel: three judges deciding whether an item meets a criterion. The
headline kappa is bad. The point of the example is that the headline kappa
does not tell you what to DO about it - and the decomposition does.
"""

import json
from pathlib import Path

from judgekit import decompose, interpret_kappa, markdown_report

HERE = Path(__file__).parent


def main() -> None:
    ratings = json.loads((HERE / "judge_panel_ratings.json").read_text(encoding="utf-8"))

    result = decompose(ratings)

    print("=" * 70)
    print("WHAT A BARE KAPPA TELLS YOU")
    print("=" * 70)
    print(f"Fleiss kappa = {result.inter_judge_kappa:.3f}  ({interpret_kappa(result.inter_judge_kappa)})")
    print("Actionable next step: none. You know it is bad. You do not know why.")
    print()

    print("=" * 70)
    print("WHAT REPLICATION TELLS YOU")
    print("=" * 70)
    print(result.summary())
    print()

    print("Per-item, the panel splits into two populations that need OPPOSITE fixes:")
    print()
    seen = set()
    for item in result.items:
        if item.verdict in seen or item.verdict == "clean":
            continue
        seen.add(item.verdict)
        print(f"  {item.item_id}  [{item.verdict}]")
        print(f"    self-consistency {item.self_consistency:.2f} | consensus {item.cross_judge_consensus:.2f}")
        print(f"    -> {item.actionable}")
        print()

    out = HERE / "report.md"
    out.write_text(markdown_report(result, title="Screening panel - worked example"), encoding="utf-8")
    print(f"Full markdown report written to {out}")


if __name__ == "__main__":
    main()
