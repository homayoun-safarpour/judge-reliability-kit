from pathlib import Path

from judgekit.cli import main

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8-sig")
RATINGS = ROOT / "examples" / "judge_panel_ratings.json"


def test_readme_spoken_h1_is_judgekit():
    assert README.lstrip().startswith("# judgekit\n")


def test_readme_first_screen_matches_top100_craft():
    pip_at = README.find("pip install")
    problem_at = README.find("## The problem")
    assert 0 <= pip_at < problem_at
    head = "\n".join(README.splitlines()[:24])
    assert "# judgekit" in head
    assert "git clone https://github.com/homayoun-safarpour/judge-reliability-kit" in head
    assert "pip install -e" in head
    assert "judgekit report examples/judge_panel_ratings.json" in head
    assert "0.242" in head
    assert "rubric_underspecified" in head
    assert "Interview pack" not in head
    assert "\u2014" not in head


def test_readme_stranger_report_is_untrusted():
    code = main(["report", str(RATINGS)])
    assert code == 1
