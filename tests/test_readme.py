import json
from pathlib import Path

from judgekit.cli import main

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8-sig")
RATINGS = ROOT / "examples" / "judge_panel_ratings.json"


def test_readme_h1_matches_repo_name():
    assert README.lstrip().startswith("# judge-reliability-kit\n")


def test_readme_first_screen_matches_top100_craft():
    pip_at = README.find("pip install")
    problem_at = README.find("## The problem")
    assert 0 <= pip_at < problem_at
    head = "\n".join(README.splitlines()[:26])
    assert "# judge-reliability-kit" in head
    assert "Homayoun Safarpour" in head
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


def test_readme_first_screen_output_is_real(capsys):
    shown = README.split("```text\n", 1)[1].split("```", 1)[0].splitlines()
    main(["report", str(RATINGS)])
    printed = capsys.readouterr().out.splitlines()
    assert len(shown) >= 7
    assert printed[: len(shown)] == shown


def test_readme_clean_panel_exits_zero(tmp_path):
    clean = {
        f"item{i}": {j: [label, label] for j in ("a", "b", "c")}
        for i, label in enumerate(["include", "exclude", "include", "exclude"])
    }
    path = tmp_path / "clean.json"
    path.write_text(json.dumps(clean), encoding="utf-8")
    assert main(["report", str(path)]) == 0
    assert "Fleiss kappa 0.6 or above | 0 |" in README
