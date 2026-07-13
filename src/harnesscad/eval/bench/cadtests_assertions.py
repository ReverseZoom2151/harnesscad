"""CADTEST assertion primitives (Mallis et al., "Text-to-CAD Evaluation with
CADTESTS", Sec. 4 and supplementary Sec. A).

A CADTEST is a property-based test ``T_i : M -> {0, 1}`` -- a boolean predicate
over a B-rep model that verifies one geometric or topological requirement of the
prompt. Each test carries a textual description clarifying the property it checks
and, crucially, returns an *interpretable* log message in both the pass and fail
cases (contribution (3), Sec. 1; Sec. 6, "informative message in both success and
failure cases"). Because tests are executed on generated geometry, a test can
also raise at runtime, in which case it is *invalid* and excluded from a
well-formed suite.

This module provides:

  * :class:`CadTest` -- a named, categorised predicate with a description and an
    optional prompt-requirement group, plus :meth:`CadTest.evaluate` which runs
    it against an injected model and returns a :class:`TestResult` (catching
    runtime errors as invalidity rather than propagating them).
  * The six paper categories (:data:`CATEGORIES`) used for per-category accuracy.
  * A library of assertion *factories* -- ``assert_valid_solid``,
    ``assert_num_solids``, ``assert_face_count``, ``assert_typed_face_count``,
    ``assert_has_geometry_type``, ``assert_bbox_dimension``,
    ``assert_aspect_ratio``, ``assert_largest_axis``, ``assert_volume``,
    ``assert_fill_factor``, ``assert_center_of_mass``, ``assert_symmetry``,
    ``assert_coaxial`` -- each returning a ready-to-run :class:`CadTest`.

Deterministic, stdlib-only. Models are queried through
:mod:`bench.cadtests_model`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

# The six CADTEST categories used for per-category accuracy (supplementary A).
SOLID_SHELL_VALIDITY = "solid_shell_validity"
TOPOLOGY = "topology"
GEOMETRIC_TYPES = "geometric_types"
DIMENSIONS_RATIOS = "dimensions_ratios"
VOLUMETRIC = "volumetric"
SPATIAL = "spatial"

CATEGORIES = (SOLID_SHELL_VALIDITY, TOPOLOGY, GEOMETRIC_TYPES,
              DIMENSIONS_RATIOS, VOLUMETRIC, SPATIAL)


@dataclass(frozen=True)
class TestResult:
    """Outcome of running one CADTEST on a model.

    ``passed``   True iff the predicate held.
    ``message``  interpretable log line (populated in both pass and fail cases).
    ``error``    a runtime-error string when the test raised (=> invalid test).
    """
    name: str
    category: str
    passed: bool
    message: str
    requirement: Optional[str] = None
    error: Optional[str] = None

    @property
    def valid(self):
        """A test is valid on this model if it executed without raising."""
        return self.error is None


@dataclass(frozen=True)
class CadTest:
    """A property-based test: a boolean predicate over a CAD model.

    ``predicate`` takes a model and returns either a bool or a
    ``(bool, message)`` pair. When only a bool is returned, a default message is
    synthesised from the description and outcome.
    """
    name: str
    category: str
    description: str
    predicate: Callable[[Any], Any]
    requirement: Optional[str] = None

    def __post_init__(self):
        if self.category not in CATEGORIES:
            raise ValueError("unknown category: %r" % (self.category,))

    def evaluate(self, model):
        """Run the test against ``model`` -> :class:`TestResult`.

        A raised exception is captured as ``error`` (invalid test) and counts as
        a failure, never propagating -- generated CAD models may execute with
        runtime errors (Sec. 6).
        """
        try:
            outcome = self.predicate(model)
        except Exception as exc:  # noqa: BLE001 -- runtime error => invalid test
            return TestResult(
                name=self.name, category=self.category, passed=False,
                message="%s: runtime error: %s" % (self.description, exc),
                requirement=self.requirement, error="%s: %s"
                % (type(exc).__name__, exc))
        if isinstance(outcome, tuple):
            passed, message = bool(outcome[0]), str(outcome[1])
        else:
            passed = bool(outcome)
            message = "%s -> %s" % (self.description,
                                    "PASS" if passed else "FAIL")
        return TestResult(
            name=self.name, category=self.category, passed=passed,
            message=message, requirement=self.requirement)


def _tol_ok(actual, expected, abs_tol, rel_tol):
    diff = abs(float(actual) - float(expected))
    return diff <= max(abs_tol, rel_tol * abs(float(expected))), diff


def _name(prefix, requirement):
    return prefix if requirement is None else "%s[%s]" % (prefix, requirement)


# ---------------------------------------------------------------------------
# Solid & shell validity
# ---------------------------------------------------------------------------
def assert_valid_solid(*, requirement=None):
    """The model is a single, non-degenerate, positive-volume solid."""
    def pred(m):
        ok = m.is_valid_solid()
        return ok, ("valid solid (solids=%d, volume=%.6g)"
                    % (m.num_solids(), m.get_volume()) if ok else
                    "not a valid solid (solids=%d, volume=%.6g)"
                    % (m.num_solids(), m.get_volume()))
    return CadTest(_name("assert_valid_solid", requirement),
                   SOLID_SHELL_VALIDITY,
                   "model forms a single valid solid body", pred, requirement)


def assert_num_solids(n=None, *, at_least=None, at_most=None, requirement=None):
    """Number of solid bodies is exactly ``n`` (or within [at_least, at_most])."""
    def pred(m):
        c = m.num_solids()
        if n is not None:
            return c == n, "num_solids=%d (expected %d)" % (c, n)
        lo = at_least if at_least is not None else 0
        hi = at_most if at_most is not None else c
        return lo <= c <= hi, "num_solids=%d (expected in [%s, %s])" % (
            c, at_least, at_most)
    return CadTest(_name("assert_num_solids", requirement),
                   SOLID_SHELL_VALIDITY,
                   "number of solid bodies constraint", pred, requirement)


def assert_positive_volume(*, requirement=None):
    def pred(m):
        v = m.get_volume()
        return v > 0.0, "volume=%.6g (must be > 0)" % v
    return CadTest(_name("assert_positive_volume", requirement),
                   SOLID_SHELL_VALIDITY, "model has positive volume", pred,
                   requirement)


# ---------------------------------------------------------------------------
# Topology checks
# ---------------------------------------------------------------------------
def _count_test(getter, label, prefix, category, description):
    def factory(n=None, *, at_least=None, at_most=None, requirement=None):
        def pred(m):
            c = getter(m)
            if n is not None:
                return c == n, "%s=%d (expected %d)" % (label, c, n)
            lo = at_least if at_least is not None else 0
            hi = at_most if at_most is not None else c
            return lo <= c <= hi, "%s=%d (expected in [%s, %s])" % (
                label, c, at_least, at_most)
        return CadTest(_name(prefix, requirement), category, description, pred,
                       requirement)
    return factory


assert_face_count = _count_test(lambda m: m.num_faces(), "num_faces",
                                "assert_face_count", TOPOLOGY,
                                "number of faces constraint")
assert_edge_count = _count_test(lambda m: m.num_edges(), "num_edges",
                                "assert_edge_count", TOPOLOGY,
                                "number of edges constraint")
assert_vertex_count = _count_test(lambda m: m.num_vertices(), "num_vertices",
                                  "assert_vertex_count", TOPOLOGY,
                                  "number of vertices constraint")


def assert_typed_face_count(face_type, n=None, *, at_least=None, at_most=None,
                            requirement=None):
    """Number of faces of a given surface type (e.g. cylindrical faces)."""
    def pred(m):
        c = m.count_faces_of_type(face_type)
        if n is not None:
            return c == n, "%s faces=%d (expected %d)" % (face_type, c, n)
        lo = at_least if at_least is not None else 0
        hi = at_most if at_most is not None else c
        return lo <= c <= hi, "%s faces=%d (expected in [%s, %s])" % (
            face_type, c, at_least, at_most)
    return CadTest(_name("assert_typed_face_count", requirement), TOPOLOGY,
                   "typed face-count constraint (%s)" % face_type, pred,
                   requirement)


def assert_more_faces_than(baseline, *, requirement=None):
    """More faces than a simple primitive baseline (supplementary A)."""
    def pred(m):
        c = m.num_faces()
        return c > baseline, "num_faces=%d (must exceed %d)" % (c, baseline)
    return CadTest(_name("assert_more_faces_than", requirement), TOPOLOGY,
                   "more faces than baseline %d" % baseline, pred, requirement)


# ---------------------------------------------------------------------------
# Geometric types
# ---------------------------------------------------------------------------
def assert_has_geometry_type(kind, *, edge=False, requirement=None):
    """Presence of a geometric element type (planar face, circular edge, ...)."""
    def pred(m):
        present = m.has_edge_type(kind) if edge else m.has_face_type(kind)
        return present, "%s %s present=%s" % (
            kind, "edge" if edge else "face", present)
    return CadTest(_name("assert_has_geometry_type", requirement),
                   GEOMETRIC_TYPES,
                   "presence of %s %s" % (kind, "edge" if edge else "face"),
                   pred, requirement)


def assert_no_geometry_type(kind, *, edge=False, requirement=None):
    """Absence of a geometric element type."""
    def pred(m):
        present = m.has_edge_type(kind) if edge else m.has_face_type(kind)
        return not present, "%s %s present=%s (expected absent)" % (
            kind, "edge" if edge else "face", present)
    return CadTest(_name("assert_no_geometry_type", requirement),
                   GEOMETRIC_TYPES,
                   "absence of %s %s" % (kind, "edge" if edge else "face"),
                   pred, requirement)


# ---------------------------------------------------------------------------
# Dimensions & ratios
# ---------------------------------------------------------------------------
def assert_bbox_dimension(axis, value, *, abs_tol=1e-6, rel_tol=0.0,
                          requirement=None):
    """Bounding-box extent along an axis is within tolerance of ``value``."""
    def pred(m):
        d = m.dimension(axis)
        ok, diff = _tol_ok(d, value, abs_tol, rel_tol)
        return ok, "dim[%s]=%.6g (expected %.6g, |diff|=%.3g)" % (
            axis, d, value, diff)
    return CadTest(_name("assert_bbox_dimension", requirement),
                   DIMENSIONS_RATIOS,
                   "bounding-box dimension along %s" % axis, pred, requirement)


def assert_aspect_ratio(axis_a, axis_b, ratio, *, abs_tol=1e-6, rel_tol=0.0,
                        requirement=None):
    """Ratio of two bounding-box extents is within tolerance of ``ratio``."""
    def pred(m):
        r = m.aspect_ratio(axis_a, axis_b)
        ok, diff = _tol_ok(r, ratio, abs_tol, rel_tol)
        return ok, "aspect[%s/%s]=%.6g (expected %.6g, |diff|=%.3g)" % (
            axis_a, axis_b, r, ratio, diff)
    return CadTest(_name("assert_aspect_ratio", requirement), DIMENSIONS_RATIOS,
                   "aspect ratio %s/%s" % (axis_a, axis_b), pred, requirement)


def assert_largest_axis(axis, *, requirement=None):
    """The longest bounding-box extent is along ``axis`` (pose-invariant)."""
    from harnesscad.eval.bench.cadtests_model import _axis_index

    def pred(m):
        want = _axis_index(axis)
        got = m.largest_axis()
        return got == want, "largest_axis=%d (expected %d)" % (got, want)
    return CadTest(_name("assert_largest_axis", requirement), DIMENSIONS_RATIOS,
                   "largest axis is %s" % axis, pred, requirement)


def assert_face_area(index, value, *, abs_tol=1e-6, rel_tol=0.0,
                     requirement=None):
    def pred(m):
        a = m.face_area(index)
        ok, diff = _tol_ok(a, value, abs_tol, rel_tol)
        return ok, "face_area[%d]=%.6g (expected %.6g, |diff|=%.3g)" % (
            index, a, value, diff)
    return CadTest(_name("assert_face_area", requirement), DIMENSIONS_RATIOS,
                   "area of face %d" % index, pred, requirement)


# ---------------------------------------------------------------------------
# Volumetric checks
# ---------------------------------------------------------------------------
def assert_volume(value, *, abs_tol=1e-6, rel_tol=0.0, requirement=None):
    """Measured volume is within tolerance of ``value``."""
    def pred(m):
        v = m.get_volume()
        ok, diff = _tol_ok(v, value, abs_tol, rel_tol)
        return ok, "volume=%.6g (expected %.6g, |diff|=%.3g)" % (v, value, diff)
    return CadTest(_name("assert_volume", requirement), VOLUMETRIC,
                   "measured volume matches expected", pred, requirement)


def assert_fill_factor(value, *, abs_tol=1e-6, rel_tol=0.0, requirement=None):
    """Volume-to-bounding-box fill factor within tolerance (shape factor)."""
    def pred(m):
        f = m.fill_factor()
        ok, diff = _tol_ok(f, value, abs_tol, rel_tol)
        return ok, "fill_factor=%.6g (expected %.6g, |diff|=%.3g)" % (
            f, value, diff)
    return CadTest(_name("assert_fill_factor", requirement), VOLUMETRIC,
                   "fill factor (volume / bbox volume)", pred, requirement)


# ---------------------------------------------------------------------------
# Spatial arrangement
# ---------------------------------------------------------------------------
def assert_center_of_mass(point, *, abs_tol=1e-6, requirement=None):
    """Center of mass is within tolerance of ``point`` (pose-dependent)."""
    def pred(m):
        com = m.get_center_of_mass()
        worst = max(abs(com[i] - float(point[i])) for i in range(3))
        return worst <= abs_tol, "com=%s (expected %s, max|diff|=%.3g)" % (
            tuple(round(c, 6) for c in com), tuple(point), worst)
    return CadTest(_name("assert_center_of_mass", requirement), SPATIAL,
                   "center of mass at point", pred, requirement)


def assert_coaxial(*, abs_tol=1e-6, requirement=None):
    """Center of mass coincides with the bounding-box center (concentric)."""
    def pred(m):
        com = m.get_center_of_mass()
        ctr = m.bbox_center()
        worst = max(abs(com[i] - ctr[i]) for i in range(3))
        return worst <= abs_tol, ("com vs bbox-center max|diff|=%.3g "
                                  "(coaxial/concentric)" % worst)
    return CadTest(_name("assert_coaxial", requirement), SPATIAL,
                   "center of mass aligned with bounding-box center", pred,
                   requirement)


def assert_symmetry(axis, *, abs_tol=1e-6, requirement=None):
    """Center of mass lies on the bounding-box mid-plane along ``axis``."""
    from harnesscad.eval.bench.cadtests_model import _axis_index

    def pred(m):
        i = _axis_index(axis)
        com = m.get_center_of_mass()[i]
        ctr = m.bbox_center()[i]
        diff = abs(com - ctr)
        return diff <= abs_tol, "com[%s]=%.6g vs center %.6g (|diff|=%.3g)" % (
            axis, com, ctr, diff)
    return CadTest(_name("assert_symmetry", requirement), SPATIAL,
                   "symmetric about %s mid-plane" % axis, pred, requirement)
