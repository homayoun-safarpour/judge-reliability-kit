import pytest

from judgekit import (
    decompose,
    markdown_report,
    position_bias,
    scale_direction_bias,
    self_enhancement_bias,
    verbosity_bias,
)


def test_position_bias_detects_first_slot_preference():
    # A judge that always picks whatever it sees first.
    def compare(a, b):
        return 1

    pairs = [("answer A", "answer B")] * 20
    result = position_bias(pairs, compare)
    assert result.effect == pytest.approx(0.5)
    assert result.flagged


def test_position_bias_clean_judge_flips_with_order():
    # A judge with a real preference for the string "good" flips when order flips.
    def compare(a, b):
        return 1 if "good" in a else -1

    pairs = [("good answer", "bad answer")] * 20
    result = position_bias(pairs, compare)
    assert result.effect == pytest.approx(0.0)
    assert not result.flagged


def test_verbosity_bias_detects_length_reward():
    def score(record):
        return len(record["answer"]) / 1000.0

    records = [{"answer": "short " * (i + 1)} for i in range(1, 6)]
    result = verbosity_bias(records, score)
    assert result.effect > 0
    assert result.flagged


def test_verbosity_bias_clean_judge_ignores_padding():
    def score(record):
        return 1.0 if "correct" in record["answer"] else 0.0

    records = [{"answer": "this is correct"}, {"answer": "this is wrong"}]
    result = verbosity_bias(records, score)
    assert result.effect == pytest.approx(0.0)
    assert not result.flagged


def test_self_enhancement_detects_own_model_preference():
    def score(record):
        return 0.9 if record["author_model"] == "gpt" else 0.4

    records = [{"answer": "x", "author_model": m} for m in ("gpt", "gpt", "claude", "llama")]
    result = self_enhancement_bias(records, score, judge_identity="gpt")
    assert result.effect == pytest.approx(0.5)
    assert result.flagged


def test_self_enhancement_requires_identity():
    with pytest.raises(ValueError):
        self_enhancement_bias([{"answer": "x", "author_model": "gpt"}], lambda r: 1.0)


def test_probe_result_str_is_readable():
    def compare(a, b):
        return 1

    text = str(position_bias([("a", "b")] * 5, compare))
    assert "position" in text
    assert "FLAG" in text


def test_scale_direction_flags_numeric_prior():
    # A judge that always gives the top number, whichever end means "best".
    # Normal 5 and reversed 5 map back to 5 and 1: a 4-point gap on a 1 to 5 scale.
    scores = {f"item{i}": (5, 5) for i in range(10)}
    result = scale_direction_bias(scores, scale_min=1, scale_max=5)
    assert result.name == "scale_direction"
    assert result.effect == pytest.approx(1.0)
    assert result.n == 10
    assert result.flagged


def test_scale_direction_passes_faithful_judge():
    # A judge that follows the rubric's meaning: the reversed score maps back exactly.
    normal = [1, 2, 3, 4, 5, 4, 2, 3]
    scores = {f"item{i}": (s, 6 - s) for i, s in enumerate(normal)}
    result = scale_direction_bias(scores, scale_min=1, scale_max=5)
    assert result.effect == pytest.approx(0.0)
    assert not result.flagged


def test_scale_direction_rejects_bad_input():
    with pytest.raises(ValueError):
        scale_direction_bias({}, scale_min=1, scale_max=5)
    with pytest.raises(ValueError):
        scale_direction_bias({"a": (6, 1)}, scale_min=1, scale_max=5)
    with pytest.raises(ValueError):
        scale_direction_bias({"a": (3, 3)}, scale_min=5, scale_max=5)


def test_report_lists_scale_direction_probe():
    ratings = {f"i{n}": {j: ["yes", "yes"] for j in ("a", "b")} for n in range(3)}
    probe = scale_direction_bias({"x": (5, 5)}, scale_min=1, scale_max=5)
    md = markdown_report(decompose(ratings), probes=[probe])
    assert "| scale_direction | +1.000 | YES |" in md
    assert "arXiv:2611.25588" in md
