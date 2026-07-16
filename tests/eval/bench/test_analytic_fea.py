"""The closed-form FEA oracles must equal their own formulas.

These tests re-derive each value from arithmetic written here, independently of
the module's own helpers wherever practical, so a bug in the module cannot make
its own tests pass. That is the point of an oracle: the check does not consult
the thing being checked.
"""

from __future__ import annotations

import math

import pytest

from harnesscad.eval.bench import analytic_fea as af

E = 210000.0
L = 100.0
I_STRONG = 20000.0 / 3.0          # 10 * 20^3 / 12
I_WEAK = 5000.0 / 3.0             # 20 * 10^3 / 12


def test_section_properties_are_exact_fractions():
    i, c, _z, a = af.BEAM.strong
    assert math.isclose(i, I_STRONG)
    assert (c, a) == (10.0, 200.0)
    iw, cw, _zw, _aw = af.BEAM.weak
    assert math.isclose(iw, I_WEAK)
    assert cw == 5.0


def test_weak_axis_is_four_times_softer_than_strong():
    """I falls by (20/10)^2 = 4 when the same beam is turned 90 degrees."""
    strong = af.case("cantilever_end_load", "max_displacement").value
    weak = af.case("cantilever_end_load_lateral", "max_displacement").value
    assert math.isclose(weak / strong, 4.0)


@pytest.mark.parametrize("case_id,metric,expected", [
    # Hand-worked from the formula each row cites; see the module docstring.
    ("tension_rod", "max_displacement", 1.0 / 210.0),
    ("tension_rod", "max_von_mises_stress", 10.0),
    ("cantilever_end_load", "max_displacement", 1.0 / 42.0),
    ("cantilever_end_load", "max_von_mises_stress", 15.0),
    ("cantilever_udl", "max_displacement", 1.0 / 112.0),
    ("cantilever_udl", "max_von_mises_stress", 7.5),
    ("cantilever_midspan_load", "max_displacement", 62500000.0 / 8.4e9),
    ("cantilever_midspan_load", "max_von_mises_stress", 7.5),
    ("cantilever_end_load_lateral", "max_displacement", 1e8 / 1.05e9),
    ("cantilever_end_load_lateral", "max_von_mises_stress", 30.0),
    ("fixed_fixed_udl", "max_displacement", 1e8 / 5.376e11),
    ("fixed_fixed_udl", "max_von_mises_stress", 1.25),
    ("fixed_fixed_center_load", "max_displacement", 1e8 / 2.688e11),
    ("column_buckling", "lowest_buckling_factor", math.pi ** 2 * 8.75),
])
def test_value_equals_hand_computed_closed_form(case_id, metric, expected):
    assert math.isclose(af.case(case_id, metric).value, expected, rel_tol=1e-12)


def test_modal_fundamental():
    f1 = af.case("cantilever_modal", "first_natural_frequency_hz").value
    expected = (1.875104 ** 2 / (2.0 * math.pi)) * math.sqrt(3.5e8 / 157.0)
    assert math.isclose(f1, expected, rel_tol=1e-12)
    assert 835.0 < f1 < 836.0


def test_midspan_formula_reduces_to_end_load_at_a_equals_L():
    from harnesscad.eval.verifiers.simulation import beam_max_deflection
    assert math.isclose(
        af.cantilever_point_load_tip_deflection(100.0, L, L, E, I_STRONG),
        beam_max_deflection(100.0, L, E, I_STRONG, "cantilever"))


def test_modal_frequency_scales_as_inverse_length_squared():
    a = af.clamped_free_first_frequency(E, I_WEAK, 7.85e-9, 200.0, 100.0)
    b = af.clamped_free_first_frequency(E, I_WEAK, 7.85e-9, 200.0, 200.0)
    assert math.isclose(a / b, 4.0)


def test_every_row_is_cited_and_positive():
    for row in af.cases():
        assert row.value > 0.0
        assert row.formula and row.citation
        assert row.tolerance_percent > 0.0


def test_tolerance_band_binds_both_ways():
    row = af.case("cantilever_end_load", "max_displacement")  # +/- 10%
    assert row.within_tolerance(row.value)
    assert row.within_tolerance(row.value * 1.09)
    assert not row.within_tolerance(row.value * 1.11)
    assert not row.within_tolerance(row.value * 0.89)


# --------------------------------------------------------------------------- #
# the honesty contract
# --------------------------------------------------------------------------- #
def test_exactly_the_two_fixed_fixed_displacements_are_not_oracles():
    non = sorted((r.case_id, r.metric) for r in af.cases() if not r.is_oracle)
    assert non == [("fixed_fixed_center_load", "max_displacement"),
                   ("fixed_fixed_udl", "max_displacement")]
    assert len(af.oracles()) == len(af.cases()) - 2


def test_oracles_excludes_every_non_oracle():
    assert all(r.is_oracle for r in af.oracles())


def test_upstream_disagreements_are_exactly_the_non_oracles():
    """A stored number that is not its own formula must never be an oracle.

    This is the regression guarding the finding: if someone later "fixes" the
    cross-check by adopting upstream's 2.63e-4, this fails.
    """
    bad = af.disagreements()
    if not bad:                      # vendored data unreadable -> nothing to say
        pytest.skip("upstream reference data not resolvable")
    assert {x.case_id for x in bad} == {"fixed_fixed_udl",
                                        "fixed_fixed_center_load"}
    for x in bad:
        assert not af.case(x.case_id, x.metric).is_oracle
        assert abs(x.deviation_percent) > 1.0


def test_the_two_documented_deviations_have_not_moved():
    checks = {(x.case_id, x.metric): x for x in af.crosscheck()}
    if not checks:
        pytest.skip("upstream reference data not resolvable")
    udl = checks[("fixed_fixed_udl", "max_displacement")]
    ctr = checks[("fixed_fixed_center_load", "max_displacement")]
    assert udl.upstream == pytest.approx(0.000263)
    assert udl.deviation_percent == pytest.approx(41.39, abs=0.01)
    assert ctr.upstream == pytest.approx(0.000657)
    assert ctr.deviation_percent == pytest.approx(76.60, abs=0.01)


def test_agreeing_rows_really_agree():
    for x in af.crosscheck():
        if af.case(x.case_id, x.metric).is_oracle:
            assert x.agrees, str(x)


# --------------------------------------------------------------------------- #
# provenance + degradation
# --------------------------------------------------------------------------- #
def test_manifest_is_mit_and_vendored_bytes_verify():
    m = af.manifest()
    assert m.license == "MIT"
    assert m.verify_vendored() == []
    assert m.availability()["total"] == 10


def test_oracles_need_no_files_at_all(monkeypatch):
    """The answer key is arithmetic: it survives a wheel with no data dir."""
    monkeypatch.setattr(af, "upstream_reference", lambda _cid: None)
    monkeypatch.setattr(af, "_corpus_order", lambda: [])
    rows = af.cases()
    assert len(rows) == 15
    assert af.crosscheck() == []
    assert math.isclose(af.case("cantilever_end_load", "max_displacement").value,
                        1.0 / 42.0)


def test_selfcheck_exits_zero():
    assert af.main(["--selfcheck"]) == 0
