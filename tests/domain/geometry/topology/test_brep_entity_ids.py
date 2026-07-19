"""Tests for the JoinABLe B-rep entity-type + convexity fact tables."""

from harnesscad.domain.geometry.topology import brep_entity_ids as bei


def test_entity_ids_contiguous_and_unique():
    ids = [int(m.value) for m in bei.EntityType]
    assert len(ids) == len(set(ids))
    assert sorted(ids) == list(range(16))


def test_surface_curve_split():
    assert len(bei.SURFACE_TYPES) == 8
    assert len(bei.CURVE_TYPES) == 8
    assert bei.is_surface_type(bei.EntityType.NurbsSurfaceType)
    assert bei.is_curve_type(bei.EntityType.Line3DCurveType)
    assert bei.is_curve_type(bei.EntityType.Degenerate3DCurveType)


def test_entity_name_id_round_trip():
    for member in bei.EntityType:
        assert bei.entity_name_to_id[member.name] == int(member.value)
        assert bei.entity_id_to_name[int(member.value)] == member.name


def test_entity_ids_match_source():
    assert bei.EntityType.PlaneSurfaceType == 0
    assert bei.EntityType.NurbsSurfaceType == 7
    assert bei.EntityType.Line3DCurveType == 8
    assert bei.EntityType.Degenerate3DCurveType == 15


def test_convexity_six_states_contiguous():
    ids = [int(m.value) for m in bei.Convexity]
    assert sorted(ids) == list(range(6))


def test_convexity_wire_names():
    assert bei.convexity_name_to_id["None"] == 0
    assert bei.convexity_name_to_id["Convex"] == 1
    assert bei.convexity_name_to_id["Concave"] == 2
    assert bei.convexity_name_to_id["Smooth"] == 3
    assert bei.convexity_name_to_id["Non-manifold"] == 4
    assert bei.convexity_name_to_id["Degenerate"] == 5


def test_classify_covers_all_states():
    states = {bei.classify(w) for w in bei.convexity_name_to_id}
    assert len(states) == 6
    assert bei.is_convex(bei.classify("Convex"))
    assert bei.is_concave(bei.classify("Concave"))


def test_classify_case_insensitive():
    assert bei.classify("non-manifold") is bei.Convexity.Nonmanifold


def test_continuous_labels_are_subset():
    for label in ("convex", "concave", "smooth"):
        assert label in bei.EDGE_CONVEXITY_TO_ID


def test_analytic_surface_bridge():
    for cls in ("Plane", "Cylinder", "Cone", "Sphere", "Torus"):
        assert cls in bei.ANALYTIC_SURFACE_TO_ID


def test_axis_thresholds_and_predicate():
    assert bei.AXIS_ANGLE_TOL_DEG == 10.0
    assert bei.AXIS_DISTANCE_TOL == 1e-2
    assert bei.axis_lines_colinear(9.9, 9e-3)
    assert not bei.axis_lines_colinear(10.0, 0.0)
    assert not bei.axis_lines_colinear(0.0, 1e-2)


def test_selfcheck_exits_zero():
    assert bei._selfcheck() == 0
