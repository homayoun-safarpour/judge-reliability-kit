import math

import pytest

from judgekit import (
    cohen_kappa,
    fleiss_kappa,
    interpret_kappa,
    krippendorff_alpha,
    pairwise_kappas,
    percent_agreement,
)


def test_perfect_agreement_is_one():
    ratings = [["a", "a"], ["b", "b"], ["a", "a"], ["b", "b"]]
    assert fleiss_kappa(ratings) == pytest.approx(1.0)
    assert cohen_kappa([r[0] for r in ratings], [r[1] for r in ratings]) == pytest.approx(1.0)


def test_chance_level_agreement_is_near_zero():
    # Two judges whose labels are independent should land near zero, not near
    # the (high) raw percent agreement.
    a = ["y", "y", "n", "n"] * 10
    b = ["y", "n", "y", "n"] * 10
    assert cohen_kappa(a, b) == pytest.approx(0.0, abs=1e-9)


def test_imbalance_trap_high_percent_low_kappa():
    # The reason percent agreement is reported here only as a foil: both judges
    # say "exclude" almost always, agree 96% of the time, and share nothing.
    a = ["exclude"] * 48 + ["include", "exclude"]
    b = ["exclude"] * 48 + ["exclude", "include"]
    assert percent_agreement(list(zip(a, b))) > 0.9
    assert cohen_kappa(a, b) < 0.0


def test_cohen_matches_known_value():
    # Textbook 2x2: a=20, b=5, c=10, d=15 -> kappa = 0.4
    a = ["y"] * 20 + ["y"] * 5 + ["n"] * 10 + ["n"] * 15
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert cohen_kappa(a, b) == pytest.approx(0.4, abs=1e-9)


def test_fleiss_matches_cohen_for_two_judges_balanced():
    ratings = [["y", "y"], ["y", "n"], ["n", "y"], ["n", "n"]] * 5
    a = [r[0] for r in ratings]
    b = [r[1] for r in ratings]
    assert fleiss_kappa(ratings) == pytest.approx(cohen_kappa(a, b), abs=1e-9)


def test_krippendorff_handles_missing_values():
    ratings = [["a", "a", None], ["b", "b", "b"], ["a", None, "a"], ["b", "b", "b"]]
    assert krippendorff_alpha(ratings) > 0.9


def test_krippendorff_perfect_is_one():
    ratings = [["a", "a", "a"], ["b", "b", "b"], ["c", "c", "c"]]
    assert krippendorff_alpha(ratings) == pytest.approx(1.0)


def test_pairwise_kappas_expose_shape_the_mean_hides():
    # Two judges lockstep, a third orthogonal. The panel mean would read
    # "moderate"; the pairwise view shows one aligned pair and two dead pairs.
    n = 40
    j1 = ["y" if i % 2 else "n" for i in range(n)]
    j2 = list(j1)
    j3 = ["y" if i % 3 else "n" for i in range(n)]
    kappas = pairwise_kappas([list(t) for t in zip(j1, j2, j3)], ["j1", "j2", "j3"])
    assert kappas[("j1", "j2")] == pytest.approx(1.0)
    assert kappas[("j1", "j3")] < 0.4
    assert kappas[("j2", "j3")] < 0.4


def test_interpret_kappa_bands():
    assert "slight" in interpret_kappa(0.20 - 1e-9)
    assert "moderate" in interpret_kappa(0.5)
    assert "substantial" in interpret_kappa(0.7)
    assert "worse than chance" in interpret_kappa(-0.1)
    assert interpret_kappa(math.nan) == "undefined"


def test_rejects_ragged_input():
    with pytest.raises(ValueError):
        fleiss_kappa([["a", "b"], ["a"]])


def test_rejects_single_judge():
    with pytest.raises(ValueError):
        fleiss_kappa([["a"], ["b"]])
