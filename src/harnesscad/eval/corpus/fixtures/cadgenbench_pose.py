"""cadgenbench's pose-invariance STEP twins + the open-shell invalid oracle.

Source: cadgenbench (resources/cad_repos/cadgenbench-main, Copyright 2026
Hugging Face, Apache-2.0), ``tests/fixtures/``. Vendored under
``fixtures/cadgenbench/`` with per-file SHA-256 provenance in
``MANIFEST.json`` and attribution in ``LICENSE-NOTICE.txt``.

What the fixtures are:

* THREE TWIN PAIRS -- the same solid exported twice, once in a reference pose
  and once rigidly moved: ``box_10_10_10`` vs ``box_10_10_10_cube_rot73``
  (rotation), ``box_10_20_30`` vs ``box_10_20_30_rot37_t10`` (rotation +
  translation), ``l_bracket`` vs ``l_bracket_l_rot45`` (rotation). Any
  pose-invariant observable MUST agree across a pair; that is exactly the
  real-file regression data ``eval/quality/geometry/invariance.py`` and
  ``canonical_pose.py`` were built without.
* SINGLETONS -- ``box_20_20_20`` (cube: fully ambiguous extents),
  ``sphere_10`` (everything ambiguous), ``tapered_box`` (distinct extents):
  the graded subjects for ``canonical_pose.pose_report``'s ambiguity flag.
* ``open_shell.step`` -- an explicitly NON-solid shell: a known-bad oracle.
  A validity checker that calls it a solid is wrong; an importer that
  crashes on it is worse.

How they plug into the existing modules (both kernel-free, both injected):

* :func:`invariance_contract` builds an ``invariance.InvarianceContract``
  whose subject is a STEP path, whose "transform" simply substitutes the
  pre-transformed twin file, and whose default measure is
  :func:`point_cloud_diameter` over the file's ``CARTESIAN_POINT`` entities
  -- the maximum pairwise distance of a rigidly-moved point set is invariant,
  so the twins must agree. Callers with a kernel inject their own measure
  (volume, area) through the same contract.
* :func:`step_cartesian_points` yields the 3D point set that
  ``canonical_pose.bounding_box`` / ``pose_report`` consume directly.

The text-level ``CARTESIAN_POINT`` parse is deliberately naive -- it reads
every 3D point in the file (B-spline control points, axis placements, all of
it). That is exactly what makes its diameter a rigid-motion invariant of the
whole exported model, and it needs no kernel.

Stdlib only (plus the kernel-free invariance module), deterministic.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from harnesscad.eval.corpus.fixtures import Manifest, load_manifest
from harnesscad.eval.quality.geometry.invariance import (
    ContractMetadata,
    InvarianceContract,
    PerturbationCase,
)

__all__ = [
    "StepFixture",
    "TwinPair",
    "manifest",
    "twin_pairs",
    "singletons",
    "open_shell_canary",
    "step_cartesian_points",
    "point_cloud_diameter",
    "invariance_contract",
    "evaluate_pairs",
    "main",
]

_SOURCE = "cadgenbench"

#: (pair name, baseline entry, transformed entry, transformation, note)
_PAIRS: Tuple[Tuple[str, str, str, str, str], ...] = (
    ("box_10_10_10_rot",
     "box_10_10_10", "box_10_10_10_cube_rot73", "rotation",
     "10 mm cube, reference pose vs rotated 73 degrees."),
    ("box_10_20_30_rot_trans",
     "box_10_20_30", "box_10_20_30_rot37_t10", "rotation",
     "10x20x30 box, reference pose vs rotated 37 degrees and translated "
     "10 mm. Transformation metadata says 'rotation' because the contract "
     "vocabulary is single-axis; the twin is rotated AND translated, and an "
     "invariant observable must survive both."),
    ("l_bracket_rot",
     "l_bracket", "l_bracket_l_rot45", "rotation",
     "L-bracket, reference pose vs rotated 45 degrees about its long leg."),
)

_SINGLETONS: Tuple[str, ...] = ("box_20_20_20", "sphere_10", "tapered_box")

#: Diameters between twins agree to float/export noise; a broken export or a
#: scaled part misses by orders of magnitude more than this.
REL_TOL = 1e-6

_POINT_RE = re.compile(
    r"CARTESIAN_POINT\s*\(\s*'[^']*'\s*,\s*\(([^)]*)\)", re.IGNORECASE)

Point3 = Tuple[float, float, float]


@dataclass(frozen=True)
class StepFixture:
    name: str
    path: Optional[Path]
    sha256: str

    @property
    def available(self) -> bool:
        return self.path is not None


@dataclass(frozen=True)
class TwinPair:
    """One solid in two poses. Pose-invariant observables MUST agree."""

    name: str
    transformation: str
    baseline: StepFixture
    transformed: StepFixture
    description: str

    @property
    def available(self) -> bool:
        return self.baseline.available and self.transformed.available

    def as_perturbation_case(self) -> PerturbationCase:
        """The twin file IS the perturbation: parameter = transformed path."""
        return PerturbationCase(name=self.name,
                                parameter=self.transformed.path)


def manifest() -> Manifest:
    return load_manifest(_SOURCE)


def _fixture(m: Manifest, name: str) -> StepFixture:
    e = m.by_name(name)
    if e is None:
        return StepFixture(name, None, "")
    return StepFixture(name, m.resolve(e), e.sha256)


def twin_pairs() -> List[TwinPair]:
    m = manifest()
    return [
        TwinPair(name, transformation, _fixture(m, a), _fixture(m, b), note)
        for name, a, b, transformation, note in _PAIRS
    ]


def singletons() -> List[StepFixture]:
    """Symmetry-graded canonical-pose subjects (cube, sphere, tapered box)."""
    m = manifest()
    return [_fixture(m, name) for name in _SINGLETONS]


def open_shell_canary() -> StepFixture:
    """An explicitly non-solid shell: the known-bad validity oracle."""
    return _fixture(manifest(), "open_shell")


def step_cartesian_points(path: Path) -> List[Point3]:
    """Every 3D ``CARTESIAN_POINT`` in a STEP file, by pure text parse."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    points: List[Point3] = []
    for match in _POINT_RE.finditer(text):
        parts = [p.strip() for p in match.group(1).split(",")]
        if len(parts) != 3:
            continue
        try:
            points.append((float(parts[0]), float(parts[1]), float(parts[2])))
        except ValueError:
            continue
    return points


def point_cloud_diameter(points: Sequence[Point3]) -> float:
    """Max pairwise distance: invariant under any rigid motion of the set."""
    if len(points) < 2:
        return 0.0
    best = 0.0
    for i in range(len(points) - 1):
        xi, yi, zi = points[i]
        for j in range(i + 1, len(points)):
            xj, yj, zj = points[j]
            d = (xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2
            if d > best:
                best = d
    return math.sqrt(best)


def _measure_path_diameter(path: Path) -> float:
    return point_cloud_diameter(step_cartesian_points(path))


def invariance_contract(measure=None, rel_tol: float = REL_TOL,
                        ) -> InvarianceContract:
    """A pose-invariance contract over STEP FILES rather than op streams.

    Subject and perturbation parameter are both file paths; the "transform"
    substitutes the pre-transformed twin. Default measure is the kernel-free
    point-cloud diameter; callers with a kernel inject volume/area instead.
    """
    return InvarianceContract(
        ContractMetadata(
            name="cadgenbench_pose_twin",
            transformation="rotation",
            relation="invariant",
            observable="point_cloud_diameter",
            description="the same exported solid, re-posed: any "
                        "pose-invariant observable must agree across twins",
        ),
        transform=lambda _subject, twin_path: twin_path,
        measure=measure or _measure_path_diameter,
        rel_tol=rel_tol,
        abs_tol=1e-9,
    )


def evaluate_pairs(measure=None, rel_tol: float = REL_TOL) -> List[dict]:
    """Run the invariance contract over every AVAILABLE twin pair."""
    contract = invariance_contract(measure=measure, rel_tol=rel_tol)
    out: List[dict] = []
    for pair in twin_pairs():
        if not pair.available:
            out.append({"pair": pair.name, "skipped": "fixture not present"})
            continue
        report = contract.evaluate(pair.baseline.path,
                                   [pair.as_perturbation_case()])
        d = report.to_dict()
        d["pair"] = pair.name
        out.append(d)
    return out


def _selfcheck() -> int:
    m = manifest()
    assert m.license == "Apache-2.0", m.license
    problems = m.verify_vendored()
    assert not problems, "; ".join(problems)
    assert len(m.entries) == 10, "expected 10 entries, got %d" % len(m.entries)

    pairs = twin_pairs()
    assert len(pairs) == 3
    canary = open_shell_canary()
    assert canary.name == "open_shell"

    available = [p for p in pairs if p.available]
    if not available and not canary.available:
        print("SELFCHECK OK: manifest valid (10 entries); no fixture "
              "resolvable, corpus degrades to empty as designed")
        return 0

    for pair in available:
        base_pts = step_cartesian_points(pair.baseline.path)
        twin_pts = step_cartesian_points(pair.transformed.path)
        assert len(base_pts) >= 8, "%s baseline parsed no points" % pair.name
        assert len(twin_pts) >= 8, "%s twin parsed no points" % pair.name

    results = evaluate_pairs()
    ran = [r for r in results if "skipped" not in r]
    for r in ran:
        assert r["passed"], (
            "pose-invariance twin disagreed: %s -> %s" % (r["pair"], r))
    if canary.available:
        pts = step_cartesian_points(canary.path)
        assert pts, "open_shell parsed no points"
    print("SELFCHECK OK: %d/3 twin pairs present and diameter-invariant; "
          "open_shell canary %s" % (
              len(ran), "present" if canary.available else "absent"))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="cadgenbench pose-invariance STEP twins + open-shell "
                    "invalid-solid oracle (vendored, Apache-2.0).")
    parser.add_argument("--selfcheck", action="store_true",
                        help="validate manifest/hashes and run the "
                             "kernel-free diameter-invariance contract over "
                             "every available twin pair.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.selfcheck:
        parser.print_help()
        return 0
    try:
        return _selfcheck()
    except AssertionError as exc:
        print("SELFCHECK FAILED: %s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
