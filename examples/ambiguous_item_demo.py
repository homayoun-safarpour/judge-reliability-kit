"""Demo: ambiguous item wording → item_ambiguous verdict (not rubric rewrite).

Run:  python examples/ambiguous_item_demo.py
"""

from __future__ import annotations

import json
from pathlib import Path

from judgekit import decompose

HERE = Path(__file__).parent


def main() -> None:
    fixture = json.loads((HERE / "ambiguous_item_fixture.json").read_text(encoding="utf-8"))
    print("ITEM TEXT (intentionally ambiguous)")
    print(fixture["item_text"])
    print()
    print("WHY THIS IS AMBIGUOUS")
    print(fixture["why_ambiguous"])
    print()

    result = decompose(fixture["ratings"])
    focus = fixture["focus_item_id"]
    item = next(i for i in result.items if i.item_id == focus)
    print("DECOMPOSITION")
    print(f"  {item.item_id}  [{item.verdict}]")
    print(f"  self-consistency {item.self_consistency:.2f} | consensus {item.cross_judge_consensus:.2f}")
    print(f"  -> {item.actionable}")
    print()
    print(f"Panel modal Fleiss kappa = {result.modal_kappa:.3f}")
    out = HERE / "ambiguous_item_output.txt"
    lines = [
        f"item_id={item.item_id}",
        f"verdict={item.verdict}",
        f"self_consistency={item.self_consistency:.2f}",
        f"cross_judge_consensus={item.cross_judge_consensus:.2f}",
        f"actionable={item.actionable}",
        f"modal_kappa={result.modal_kappa:.3f}",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
