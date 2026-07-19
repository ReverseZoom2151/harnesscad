"""Tests for the Onshape ConstraintType integer-id fact table."""

from harnesscad.domain.geometry.sketch import constraint_type_ids as cti
from harnesscad.domain.geometry.sketch.constraint_satisfaction import CONSTRAINT_TYPES


def test_unique_ids():
    ids = [int(m.value) for m in cti.ConstraintType]
    assert len(ids) == len(set(ids))


def test_core_contiguous_plus_sentinel():
    ids = {int(m.value) for m in cti.ConstraintType}
    assert set(range(0, 30)) <= ids
    assert 101 in ids


def test_name_id_round_trip_total():
    for member in cti.ConstraintType:
        assert cti.id_for_name(member.name) == int(member.value)
        assert cti.name_for_id(int(member.value)) == member.name


def test_case_insensitive_lookup():
    assert cti.id_for_name("COINCIDENT") == 0
    assert cti.id_for_name("coincident") == 0


def test_histcad_names_are_subset():
    # Every HistCAD evaluation-side name has an authoritative integer id.
    for name in CONSTRAINT_TYPES:
        assert name in cti.HISTCAD_TO_ID
        assert isinstance(int(cti.HISTCAD_TO_ID[name]), int)


def test_specific_ids_match_source():
    assert cti.ConstraintType.Coincident == 0
    assert cti.ConstraintType.Angle == 17
    assert cti.ConstraintType.Rho == 28
    assert cti.ConstraintType.Unknown == 29
    assert cti.ConstraintType.Subnode == 101


def test_has_parameters():
    assert cti.has_parameters(cti.ConstraintType.Distance)
    assert cti.has_parameters(cti.ConstraintType.Radius)
    assert not cti.has_parameters(cti.ConstraintType.Coincident)


def test_selfcheck_exits_zero():
    assert cti._selfcheck() == 0
