"""Command line entry point: judgekit report ratings.json -o report.md"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .decompose import decompose
from .report import markdown_report


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise SystemExit("input must be a JSON object: {item: {judge: [labels...]}}")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="judgekit",
        description=(
            "Decide whether an LLM judge panel can be trusted, and if not, "
            "whether to fix the rubric or the task."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    rep = sub.add_parser("report", help="produce a reliability report")
    rep.add_argument("ratings", type=Path, help="JSON: {item_id: {judge_id: [label, label, ...]}}")
    rep.add_argument("-o", "--out", type=Path, default=None, help="write markdown here")
    rep.add_argument("--self-consistency-threshold", type=float, default=0.8)
    rep.add_argument("--consensus-threshold", type=float, default=0.6)
    rep.add_argument("--title", default="Judge panel reliability report")

    args = parser.parse_args(argv)

    if args.command == "report":
        ratings = _load(args.ratings)
        result = decompose(
            ratings,
            self_consistency_threshold=args.self_consistency_threshold,
            consensus_threshold=args.consensus_threshold,
        )
        md = markdown_report(result, title=args.title)
        if args.out:
            args.out.write_text(md, encoding="utf-8")
            print(f"wrote {args.out}")
        else:
            print(md)
        # Non-zero exit when the panel is not trustworthy, so CI can gate on it.
        return 0 if result.inter_judge_kappa >= 0.6 else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
