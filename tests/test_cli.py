import json
from pathlib import Path

import pytest

from judgekit.cli import main

ROOT = Path(__file__).resolve().parents[1]


def _run(tmp_path, payload, capsys, raw=None):
    path = tmp_path / "ratings.json"
    path.write_text(raw if raw is not None else json.dumps(payload), encoding="utf-8")
    code = main(["report", str(path)])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_string_labels_are_refused_not_split_into_characters(tmp_path, capsys):
    payload = {"i1": {"a": "pass", "b": "partial"}, "i2": {"a": "fail", "b": "fair"}}
    code, out, err = _run(tmp_path, payload, capsys)
    assert code == 1
    assert "kappa" not in out
    assert err.startswith("error: ")
    assert "list of labels" in err


def test_judge_missing_from_an_item_is_refused(tmp_path, capsys):
    payload = {f"d{i}": {"a": ["yes", "yes"], "b": ["no", "no"]} for i in range(8)}
    payload["g1"] = {j: ["yes", "yes"] for j in ("a", "b", "c")}
    payload["g2"] = {j: ["no", "no"] for j in ("a", "b", "c")}
    code, out, err = _run(tmp_path, payload, capsys)
    assert code == 1
    assert "kappa" not in out
    assert "every judge must rate every item" in err


def test_panel_with_one_label_everywhere_does_not_pass(tmp_path, capsys):
    payload = {f"i{i}": {"a": ["ok", "ok"], "b": ["ok", "ok"]} for i in range(5)}
    code, _out, err = _run(tmp_path, payload, capsys)
    assert code == 1
    assert "only one label" in err
    assert "kappa undefined" in err


def test_one_extra_replicate_does_not_bypass_the_single_pass_refusal(tmp_path, capsys):
    payload = {
        "i1": {"a": ["y"], "b": ["n"]},
        "i2": {"a": ["y", "n"], "b": ["n"]},
        "i3": {"a": ["n"], "b": ["y"]},
    }
    code, out, err = _run(tmp_path, payload, capsys)
    assert code == 1
    assert "Dominant cause" not in out
    assert "replicates" in err


def test_one_item_panel_is_refused_with_a_reason(tmp_path, capsys):
    code, out, err = _run(tmp_path, {"i1": {"a": ["y", "y"], "b": ["y", "y"]}}, capsys)
    assert code == 1
    assert "Dominant cause" not in out
    assert "at least two items" in err


@pytest.mark.parametrize(
    "raw",
    [
        "{}",
        '{"i1":{"a":[],"b":["y","y"]}}',
        '{"i1":["y","n"]}',
        '{"i1":{"a":[["y"],["y"]],"b":[["n"],["n"]]}}',
        "{nope",
        "[1, 2]",
    ],
)
def test_bad_input_stops_with_an_error_message_not_a_traceback(tmp_path, capsys, raw):
    code, out, err = _run(tmp_path, None, capsys, raw=raw)
    assert code == 1
    assert out == ""
    assert err.startswith("error: ")
    assert len(err.strip().splitlines()) == 1
    assert "Traceback" not in err


def test_missing_file_stops_with_an_error_message(tmp_path, capsys):
    code = main(["report", str(tmp_path / "absent.json")])
    err = capsys.readouterr().err
    assert code == 1
    assert err.startswith("error: ")


def test_bundled_fixture_with_top_level_ratings_key_is_read(capsys):
    code = main(["report", str(ROOT / "examples" / "ambiguous_item_fixture.json")])
    out = capsys.readouterr().out
    assert "AMB_DEMO_001" in out
    assert "Inter-judge Fleiss kappa" in out
    assert code == 1
