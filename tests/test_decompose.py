import random

import pytest

from judgekit import AMBIGUOUS, CLEAN, UNDERSPECIFIED, UNSTABLE, decompose


def _replicate(label, times=4):
    return [label] * times


def _coinflip(rng, times=4):
    return [rng.choice(["include", "exclude"]) for _ in range(times)]


def test_clean_panel_is_clean():
    ratings = {
        f"item{i}": {j: _replicate("include") for j in ("gpt", "claude", "llama")}
        for i in range(10)
    }
    result = decompose(ratings)
    assert result.dominant_cause == CLEAN
    assert result.mean_self_consistency == pytest.approx(1.0)


def test_unstable_judges_read_as_item_ambiguity():
    # Every judge flip-flops independently: nobody can reproduce themselves,
    # so the ITEM is the problem, not the rubric.
    rng = random.Random(7)
    ratings = {
        f"item{i}": {j: _coinflip(rng) for j in ("gpt", "claude", "llama")}
        for i in range(30)
    }
    result = decompose(ratings)
    assert result.dominant_cause == AMBIGUOUS
    assert result.mean_self_consistency < 0.7


def test_modal_labels_cannot_rescue_an_unstable_panel():
    """Regression: coin-flipping judges often SHARE a modal label by chance.

    Checking cross-judge consensus before self-consistency scored those items
    "clean" and hid the instability entirely. Self-consistency must be checked
    first - agreement built on coin flips is not agreement.
    """
    ratings = {
        f"item{i}": {
            j: ["include", "exclude", "include", "exclude"]
            for j in ("gpt", "claude", "llama")
        }
        for i in range(10)
    }
    result = decompose(ratings)
    # Every judge has the same modal label, so naive consensus is a perfect 1.0...
    assert all(item.cross_judge_consensus == pytest.approx(1.0) for item in result.items)
    # ...and not one item may be scored "clean".
    assert result.counts[CLEAN] == 0
    # Here every judge is a pure coin flipper on every item, so the correct
    # verdict is the judges are broken - which is a stronger finding than
    # ambiguity, and still not "clean".
    assert result.dominant_cause in (AMBIGUOUS, UNSTABLE)
    assert result.dominant_cause == UNSTABLE


def test_majority_vote_inflation_is_reported():
    """Aggregating over replicates can manufacture agreement. Surface it."""
    ratings = {
        f"item{i}": {
            j: ["include", "exclude", "include", "exclude"]
            for j in ("gpt", "claude", "llama")
        }
        for i in range(10)
    }
    result = decompose(ratings)
    assert result.modal_kappa > result.single_pass_kappa or result.modal_kappa == pytest.approx(
        result.single_pass_kappa
    )
    assert result.majority_vote_inflation >= 0.0


def test_stable_but_divergent_judges_read_as_rubric_underspecification():
    # Each judge is perfectly self-consistent and each has settled on a
    # different reading. This is the case a bare kappa cannot distinguish
    # from the one above - both produce a low single-pass kappa.
    ratings = {
        f"item{i}": {
            "gpt": _replicate("include"),
            "claude": _replicate("exclude"),
            "llama": _replicate("borderline"),
        }
        for i in range(10)
    }
    result = decompose(ratings)
    assert result.dominant_cause == UNDERSPECIFIED
    assert result.mean_self_consistency == pytest.approx(1.0)
    assert result.single_pass_kappa < 0.2


def test_the_two_causes_produce_the_same_kappa_but_different_verdicts():
    """The core claim of this package, asserted as a test."""
    rng = random.Random(3)
    ambiguous = {
        f"item{i}": {j: _coinflip(rng) for j in ("a", "b", "c")} for i in range(40)
    }
    underspecified = {
        f"item{i}": {
            "a": _replicate("include"),
            "b": _replicate("exclude"),
            "c": _replicate("include") if i % 2 else _replicate("exclude"),
        }
        for i in range(40)
    }
    r1 = decompose(ambiguous)
    r2 = decompose(underspecified)
    # Both panels look equally broken by the headline number...
    assert r1.single_pass_kappa < 0.25
    assert r2.single_pass_kappa < 0.25
    # ...and they need opposite fixes.
    assert r1.dominant_cause == AMBIGUOUS
    assert r2.dominant_cause == UNDERSPECIFIED


def test_per_item_verdicts_carry_actions():
    ratings = {
        "clean": {j: _replicate("include") for j in ("a", "b", "c")},
        "contested": {
            "a": _replicate("include"),
            "b": _replicate("exclude"),
            "c": _replicate("borderline"),
        },
    }
    result = decompose(ratings)
    by_id = {item.item_id: item for item in result.items}
    assert by_id["clean"].verdict == CLEAN
    assert by_id["contested"].verdict == UNDERSPECIFIED
    assert "RUBRIC" in by_id["contested"].actionable


def test_requires_replication():
    ratings = {"item0": {"a": ["include"], "b": ["exclude"]}}
    with pytest.raises(ValueError, match="replicates"):
        decompose(ratings)


def test_requires_replication_from_every_judge_on_every_item():
    ratings = {
        "item0": {"a": ["include", "exclude"], "b": ["exclude"]},
        "item1": {"a": ["include"], "b": ["exclude"]},
    }
    with pytest.raises(ValueError, match="replicates"):
        decompose(ratings)


def test_string_label_is_not_iterated_as_characters():
    ratings = {"i1": {"a": "pass", "b": "partial"}, "i2": {"a": "fail", "b": "fair"}}
    with pytest.raises(ValueError, match="list of labels"):
        decompose(ratings)


def test_requires_every_judge_on_every_item():
    ratings = {
        "item0": {"a": _replicate("x"), "b": _replicate("y")},
        "item1": {"a": _replicate("x"), "b": _replicate("y"), "c": _replicate("x")},
    }
    with pytest.raises(ValueError, match="every judge must rate every item"):
        decompose(ratings)


def test_requires_two_items():
    ratings = {"item0": {"a": _replicate("x"), "b": _replicate("x")}}
    with pytest.raises(ValueError, match="at least two items"):
        decompose(ratings)


def test_unstable_cutoff_is_half_the_self_consistency_threshold():
    # Both judges repeat themselves on 2 of 6 replicate pairs (0.333) on every item.
    # At the default threshold 0.8 the cut-off is 0.4, so the judges are unstable;
    # at 0.6 the cut-off is 0.3, so the same items stay item_ambiguous.
    ratings = {
        f"item{i}": {"a": ["x", "x", "y", "y"], "b": ["y", "y", "x", "x"]} for i in range(4)
    }
    assert decompose(ratings).counts[UNSTABLE] == 4
    relaxed = decompose(ratings, self_consistency_threshold=0.6)
    assert relaxed.counts[UNSTABLE] == 0
    assert relaxed.counts[AMBIGUOUS] == 4


def test_requires_two_judges():
    ratings = {"item0": {"a": ["include", "include"]}}
    with pytest.raises(ValueError, match="two judges"):
        decompose(ratings)


def test_summary_renders():
    ratings = {
        f"item{i}": {j: _replicate("include") for j in ("a", "b")} for i in range(4)
    }
    text = decompose(ratings).summary()
    assert "single pass" in text
    assert "dominant cause" in text
