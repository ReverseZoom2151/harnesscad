"""Tests for chatcadplus_prob2text -- graded numeric-to-text verbalisation."""

from __future__ import annotations

import pytest

from chatcadplus_prob2text import (
    Band,
    BandScheme,
    SCHEMES,
    verbalise,
    verbalise_scores,
)


def test_illustrative_four_bands():
    assert verbalise("crack", 0.05) == "No sign of crack"
    assert verbalise("crack", 0.35) == "Small possibility of crack"
    assert verbalise("crack", 0.7) == "Likely to have crack"
    assert verbalise("crack", 0.95) == "Definitely has crack"


def test_band_boundaries_are_half_open_low_inclusive():
    # 0.2 belongs to the second band, 0.5 to the third, 0.9 to the fourth.
    assert verbalise("x", 0.2) == "Small possibility of x"
    assert verbalise("x", 0.5) == "Likely to have x"
    assert verbalise("x", 0.9) == "Definitely has x"


def test_endpoints_covered():
    assert verbalise("x", 0.0) == "No sign of x"
    assert verbalise("x", 1.0) == "Definitely has x"  # last band closed on right


def test_simplistic_two_bands():
    assert verbalise("y", 0.49, scheme="simplistic") == "No y"
    assert verbalise("y", 0.5, scheme="simplistic") == "The prediction is y"


def test_direct_echoes_number():
    assert verbalise("z", 0.873, scheme="direct") == "z score: 0.87"


def test_out_of_range_raises():
    with pytest.raises(ValueError):
        verbalise("x", 1.5)
    with pytest.raises(ValueError):
        verbalise("x", -0.1)


def test_unknown_scheme_raises():
    with pytest.raises(ValueError):
        verbalise("x", 0.5, scheme="nope")


def test_all_named_schemes_present():
    assert set(SCHEMES) == {"direct", "simplistic", "illustrative"}


def test_verbalise_scores_sorts_by_descending_score():
    scores = {"edema": 0.3, "cardiomegaly": 0.95, "effusion": 0.6}
    out = verbalise_scores(scores)
    lines = out.splitlines()
    assert lines[0] == "- Definitely has cardiomegaly"
    assert lines[1] == "- Likely to have effusion"
    assert lines[2] == "- Small possibility of edema"


def test_verbalise_scores_tie_break_is_label_ascending():
    scores = {"b": 0.7, "a": 0.7}
    out = verbalise_scores(scores)
    assert out.splitlines() == ["- Likely to have a", "- Likely to have b"]


def test_verbalise_scores_sort_by_label():
    scores = {"b": 0.95, "a": 0.1}
    out = verbalise_scores(scores, sort="label")
    assert out.splitlines()[0] == "- No sign of a"


def test_verbalise_scores_deterministic():
    scores = {"p": 0.2, "q": 0.8, "r": 0.55}
    assert verbalise_scores(scores) == verbalise_scores(dict(scores))


def test_custom_band_scheme():
    scheme = BandScheme(
        "risk",
        [Band(0.0, 0.5, "{label}: safe"), Band(0.5, 1.0, "{label}: RISK")],
    )
    assert verbalise("tol", 0.9, scheme=scheme) == "tol: RISK"
    assert verbalise("tol", 0.1, scheme=scheme) == "tol: safe"


def test_non_contiguous_bands_rejected():
    with pytest.raises(ValueError):
        BandScheme("bad", [Band(0.0, 0.4, "a"), Band(0.5, 1.0, "b")])


def test_empty_scheme_rejected():
    with pytest.raises(ValueError):
        BandScheme("empty", [])
