"""Keep the README's displayed results tied to the shipped example data."""

import json
import re
from pathlib import Path

import pytest

from judgekit import decompose, markdown_report

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def example_result():
    ratings = json.loads(
        (ROOT / "examples" / "judge_panel_ratings.json").read_text(encoding="utf-8")
    )
    return decompose(ratings)


def test_readme_first_screen_matches_example_report(example_result):
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    first_screen = readme.split("\n---", 1)[0]
    block = re.search(r"Real output:\s*```\s*\n(.*?)\n```", first_screen, re.DOTALL)
    assert block is not None, "README first screen must include its real-output block"
    headline = markdown_report(example_result).split("## Headline\n", 1)[1]
    expected = [line for line in headline.splitlines() if line.startswith("- **")]
    assert len(expected) == 3
    assert block.group(1).strip().splitlines() == expected


def test_readme_worked_example_matches_computed_summary(example_result):
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    section = readme.split("## Worked example\n", 1)[1].split("### Majority-vote inflation", 1)[0]
    block = re.search(r"Real output:\s*```\s*\n(.*?)\n```", section, re.DOTALL)
    assert block is not None, "README worked example must include its real-output block"
    # Ignore hand-aligned spaces, but require every statistic and all four cause
    # counts (including zero) to match the summary printed by worked_example.py.
    displayed = [" ".join(line.split()) for line in block.group(1).splitlines()]
    for line in example_result.summary().splitlines():
        if line.strip():
            assert " ".join(line.split()) in displayed, f"README is missing current output: {line}"
