"""cadgenbench's known-bad mating-feature jigs with PINNED reference scores.

Source: cadgenbench (resources/cad_repos/cadgenbench-main, Copyright 2026
Hugging Face, Apache-2.0), ``tests/fixtures/jig_metric/`` and the pinned
``_EXPECTED_SCORES`` table in ``tests/eval/test_interface_match.py``.
Manifest-only (nothing vendored): the 32 STEP files total ~768 KB and are
regenerable only with a geometry kernel, so they are read from resources/
when present and this module carries the SHA-256 provenance plus the pinned
scores. Provenance in ``cadgenbench_broken/MANIFEST.json``.

License call on the underlying data: CLEAN Apache-2.0, no CADPrompt strings
attached. cadgenbench's prompt corpus derives from CADPrompt (whose own MIT
file may not cover the underlying part data), but these jig_metric fixtures
do NOT: every STEP here is procedurally authored by the committed
``tests/fixtures/jig_metric/generate.py`` (build123d primitives -- plates,
holes, slots, an L-bracket, a hex boss), an original work of the cadgenbench
authors. So the Apache-2.0 grant covers them outright; the CADPrompt
ambiguity is why we still vendor NOTHING and resolve from resources.

What the fixtures are -- a regression substrate of BROKEN mating features:

* FOUR TEST CASES, each a ground-truth part (``gt.step``), a GT-identical
  ``candidates/correct.step``, one or more canonical sub-volume "jig" regions
  (``jig_<context>__<index>__<KOR|KIR>.step`` -- the region R the metric
  scores), and a set of ``candidates/broken_*.step`` -- deliberate,
  predictable failures.
* ELEVEN BROKEN CANDIDATES across the four adversarial builders
  (``generate_test_1..4``): a shrunk hole, an offset hole, a filled-in hole,
  a wrong bolt spacing, a missing hole, a wrong diameter, a cylinder-for-hex
  boss, a rotated boss, shifted holes, a narrow slot, an offset slot. Each is
  engineered to drop at least one sub-volume's pose-searched IoU below the
  pass threshold.

THE VALUE IS THE PINNED SCORES. cadgenbench commits an ``_EXPECTED_SCORES``
table -- the single-number ``interface_score`` for every (test, candidate)
pair, reproducible from its deterministic pose search + soft ramp. We import
that table verbatim (:data:`PINNED_SCORES`) as a regression oracle: a change
that silently improves or degrades a re-implemented grader is caught the
moment its output stops matching these committed reference numbers.

The scoring pipeline (all kernel-bound EXCEPT the final ramp, which is pure
arithmetic and IS ported here as :func:`iou_to_interface_score`):

  raw sub-volume IoU  (max over a bounded deterministic pose search: +/-1% of
  the GT bbox diagonal per translation axis, +/-1 deg per rotation axis)
    -> soft pass/fail ramp: IoU >= 0.95 -> 1.0, <= 0.80 -> 0.0, linear between
    -> aggregate: mean over contexts of (min over the context's ramped
       sub-volume scores).

The adversarial-mesh builders (``generate_test_1..4``) are NOT ported: they
need the build123d kernel to CSG plates/holes/slots, so the defect meshes
cannot be regenerated stdlib-only. What IS portable -- and what the harness
needs to guard the grader -- is the ramp and the pinned reference numbers,
both here. :func:`verify_against` is the injection seam: a caller WITH a
kernel passes its own ``interface_score(candidate, fixture_dir)`` and this
module checks it against the pins, exactly as the pose loader injects a
volume/area measure into its invariance contract.

Stdlib only. Deterministic. No geometry kernel in this module.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from harnesscad.eval.corpus.fixtures import Manifest, load_manifest

__all__ = [
    "ScoredFixture",
    "BrokenCase",
    "PINNED_SCORES",
    "INTERFACE_FULL_SCORE_IOU",
    "INTERFACE_ZERO_SCORE_IOU",
    "DEFAULT_IOU_THRESHOLD",
    "PASS_SCORE",
    "manifest",
    "iou_to_interface_score",
    "pass_cases",
    "broken_cases",
    "scored_fixtures",
    "sub_volume_fixtures",
    "verify_against",
    "main",
]

_SOURCE = "cadgenbench_broken"

# --------------------------------------------------------------------------- #
# Scoring constants -- copied verbatim from cadgenbench's interface_match.py.
# The ramp and thresholds ARE the grader contract these pins guard.
# --------------------------------------------------------------------------- #

#: Per-sub-volume IoU at/above which a fit is a full pass (ramped score 1.0).
INTERFACE_FULL_SCORE_IOU = 0.95
#: Per-sub-volume IoU at/below which a fit is a clean fail (ramped score 0.0).
INTERFACE_ZERO_SCORE_IOU = 0.80
#: The per-sub-volume IoU pass threshold used by the discrimination check.
DEFAULT_IOU_THRESHOLD = 0.95
#: A candidate passes the aggregate metric iff it scores this (GT-identical).
PASS_SCORE = 1.000


def iou_to_interface_score(iou: float) -> float:
    """Map a raw sub-volume IoU through cadgenbench's soft pass/fail ramp.

    Pure arithmetic, ported verbatim from ``interface_match``: at/above 0.95
    -> 1.0, at/below 0.80 -> 0.0, linear in between. This is the one piece of
    the grader that needs no kernel, so it is checked directly in --selfcheck.
    """
    if iou >= INTERFACE_FULL_SCORE_IOU:
        return 1.0
    if iou <= INTERFACE_ZERO_SCORE_IOU:
        return 0.0
    span = INTERFACE_FULL_SCORE_IOU - INTERFACE_ZERO_SCORE_IOU
    return (iou - INTERFACE_ZERO_SCORE_IOU) / span


# --------------------------------------------------------------------------- #
# The pinned reference: cadgenbench's committed _EXPECTED_SCORES, verbatim.
#
# key = (test, candidate-path-relative-to-the-test-dir); value = the pinned
# single-number interface_score. gt.step and correct.step are the passing
# oracles (1.0); every broken_* is a deliberate failure with a score < 1.0.
# --------------------------------------------------------------------------- #

PINNED_SCORES: Dict[Tuple[str, str], float] = {
    ("test_1", "gt.step"): 1.000,
    ("test_1", "candidates/correct.step"): 1.000,
    ("test_1", "candidates/broken_1_small_hole.step"): 0.063,
    ("test_1", "candidates/broken_2_offset_hole.step"): 0.000,
    ("test_1", "candidates/broken_3_no_hole.step"): 0.000,
    ("test_2", "gt.step"): 1.000,
    ("test_2", "candidates/correct.step"): 1.000,
    ("test_2", "candidates/broken_1_wrong_spacing.step"): 0.000,
    ("test_2", "candidates/broken_2_missing_hole.step"): 0.000,
    ("test_2", "candidates/broken_3_wrong_diameter.step"): 0.000,
    ("test_3", "gt.step"): 1.000,
    ("test_3", "candidates/correct.step"): 1.000,
    ("test_3", "candidates/broken_1_cylinder_boss.step"): 0.402,
    ("test_3", "candidates/broken_2_rotated_boss.step"): 0.644,
    ("test_3", "candidates/broken_3_shifted_holes.step"): 0.000,
    ("test_4", "gt.step"): 1.000,
    ("test_4", "candidates/correct.step"): 1.000,
    ("test_4", "candidates/broken_1_narrow_slot.step"): 0.667,
    ("test_4", "candidates/broken_2_slot_offset.step"): 0.667,
}

#: Per-broken defect note + the ``generate_test_N`` builder that authors it.
#: (defect summary, builder id). Straight from generate.py's docstrings.
_BROKEN_DEFECTS: Dict[Tuple[str, str], Tuple[str, str]] = {
    ("test_1", "broken_1_small_hole"):
        ("hole shrunk to O9; R extends outside candidate free space",
         "generate_test_1"),
    ("test_1", "broken_2_offset_hole"):
        ("hole shifted 5 mm, far past the +/-0.5 mm pose bound; IoU ~0",
         "generate_test_1"),
    ("test_1", "broken_3_no_hole"):
        ("hole filled solid; keep-out region has no free space; IoU 0",
         "generate_test_1"),
    ("test_2", "broken_1_wrong_spacing"):
        ("bolt pattern re-pitched; no single rigid pose fits all four holes",
         "generate_test_2"),
    ("test_2", "broken_2_missing_hole"):
        ("one of four holes filled solid; that region scores exact 0",
         "generate_test_2"),
    ("test_2", "broken_3_wrong_diameter"):
        ("one hole shrunk to O8; that region extends beyond free space",
         "generate_test_2"),
    ("test_3", "broken_1_cylinder_boss"):
        ("hex boss replaced by O17 cylinder; hex region pokes past solid",
         "generate_test_3"),
    ("test_3", "broken_2_rotated_boss"):
        ("hex boss rotated 15 deg; keep-in region misaligns with solid",
         "generate_test_3"),
    ("test_3", "broken_3_shifted_holes"):
        ("bolt holes shifted (+1,+1) mm; no rigid pose fits holes AND boss",
         "generate_test_3"),
    ("test_4", "broken_1_narrow_slot"):
        ("slot height 9 not 12 mm; slot region extends past free space",
         "generate_test_4"),
    ("test_4", "broken_2_slot_offset"):
        ("slot centre shifted (0,8) mm; tiny overlap with free space; IoU ~0",
         "generate_test_4"),
}

_BUILDERS = ("generate_test_1", "generate_test_2",
             "generate_test_3", "generate_test_4")


@dataclass(frozen=True)
class ScoredFixture:
    """A committed STEP candidate carrying its PINNED expected score.

    ``passes`` is the candidate's verdict under the aggregate metric: a
    GT-identical part scores :data:`PASS_SCORE`; a broken one does not.
    """

    test: str
    candidate_rel: str            # path relative to the test dir, e.g. gt.step
    role: str                     # gt | candidate_correct | broken
    expected_score: float
    path: Optional[Path]
    sha256: str
    defect: Optional[str] = None
    builder: Optional[str] = None

    @property
    def available(self) -> bool:
        return self.path is not None

    @property
    def passes(self) -> bool:
        return self.expected_score >= PASS_SCORE

    @property
    def entry_name(self) -> str:
        """The MANIFEST entry name: test dir + candidate path, sans .step."""
        stem = self.candidate_rel[:-5] if self.candidate_rel.endswith(".step") \
            else self.candidate_rel
        return "%s/%s" % (self.test, stem)


# Backwards-friendly alias: broken candidates are ScoredFixtures too.
BrokenCase = ScoredFixture


def manifest() -> Manifest:
    return load_manifest(_SOURCE)


def _scored_fixture(m: Manifest, test: str, candidate_rel: str,
                    score: float) -> ScoredFixture:
    stem = candidate_rel[:-5] if candidate_rel.endswith(".step") \
        else candidate_rel
    name = "%s/%s" % (test, stem)
    entry = m.by_name(name)
    path = m.resolve(entry) if entry is not None else None
    sha = entry.sha256 if entry is not None else ""
    base = stem.split("/")[-1]
    if base == "gt":
        role = "gt"
    elif base == "correct":
        role = "candidate_correct"
    else:
        role = "broken"
    defect = builder = None
    if role == "broken":
        defect, builder = _BROKEN_DEFECTS.get((test, base), (None, None))
    return ScoredFixture(test, candidate_rel, role, score, path, sha,
                         defect, builder)


def scored_fixtures() -> List[ScoredFixture]:
    """Every (test, candidate) pair that carries a pinned score."""
    m = manifest()
    return [_scored_fixture(m, test, rel, score)
            for (test, rel), score in PINNED_SCORES.items()]


def pass_cases() -> List[ScoredFixture]:
    """The passing oracles: each ``gt.step`` and ``correct.step`` (score 1.0)."""
    return [f for f in scored_fixtures() if f.passes]


def broken_cases() -> List[ScoredFixture]:
    """The known-bad candidates: every ``broken_*`` with its pinned score."""
    return [f for f in scored_fixtures() if f.role == "broken"]


def sub_volume_fixtures() -> List[ScoredFixture]:
    """The canonical jig sub-volume regions (no pinned score of their own)."""
    m = manifest()
    out: List[ScoredFixture] = []
    for e in m.by_role("sub_volume"):
        test = e.name.split("/", 1)[0]
        rel = e.name.split("/", 1)[1] + ".step"
        out.append(ScoredFixture(test, rel, "sub_volume", float("nan"),
                                 m.resolve(e), e.sha256))
    return out


def verify_against(
    scorer: Callable[[Path, Path], float],
    abs_tol: float = 0.005,
) -> List[dict]:
    """Run a caller-supplied kernel scorer over available fixtures vs the pins.

    ``scorer(candidate_path, fixture_dir)`` mirrors cadgenbench's
    ``interface_score``. Kernel-free callers cannot run this; a kernel-backed
    caller uses it to prove its grader still reproduces the committed numbers.
    Absent fixtures are skipped, never an error.
    """
    m = manifest()
    root = None
    out: List[dict] = []
    for fx in scored_fixtures():
        if not fx.available or fx.path is None:
            out.append({"case": fx.entry_name, "skipped": "fixture absent"})
            continue
        # The fixture dir is the test dir: parent of gt.step / of candidates/.
        fixture_dir = fx.path.parent
        if fixture_dir.name == "candidates":
            fixture_dir = fixture_dir.parent
        got = scorer(fx.path, fixture_dir)
        out.append({
            "case": fx.entry_name,
            "expected": fx.expected_score,
            "got": got,
            "passed": abs(got - fx.expected_score) <= abs_tol,
        })
    del m, root
    return out


def _selfcheck() -> int:
    m = manifest()
    assert m.license == "Apache-2.0", m.license
    assert len(m.entries) == 32, "expected 32 entries, got %d" % len(m.entries)

    # Manifest-only by design: nothing vendored, so there is nothing to SHA
    # locally, but every entry must still carry a resource path + hash.
    assert not m.verify_vendored(), "no vendored files were expected"
    for e in m.entries:
        assert e.vendored is None, "unexpected vendored file: %s" % e.name
        assert e.resource, "entry %s has no resource path" % e.name
        assert len(e.sha256) == 64, "entry %s has no sha256" % e.name

    # Role census: 4 gt + 4 correct + 11 broken + 13 sub_volume = 32.
    roles: Dict[str, int] = {}
    for e in m.entries:
        roles[e.role] = roles.get(e.role, 0) + 1
    assert roles == {"gt": 4, "candidate_correct": 4,
                     "broken": 11, "sub_volume": 13}, roles

    # The ported ramp -- the kernel-free piece of the grader -- is correct.
    assert iou_to_interface_score(1.00) == 1.0
    assert iou_to_interface_score(0.95) == 1.0
    assert iou_to_interface_score(0.80) == 0.0
    assert iou_to_interface_score(0.50) == 0.0
    mid = iou_to_interface_score(0.875)
    assert abs(mid - 0.5) < 1e-9, mid
    # Monotonic non-decreasing across the ramp.
    prev = -1.0
    for i in range(0, 101):
        v = iou_to_interface_score(i / 100.0)
        assert v >= prev - 1e-12, "ramp not monotonic at %.2f" % (i / 100.0)
        assert 0.0 <= v <= 1.0, v
        prev = v

    fixtures = scored_fixtures()
    assert len(fixtures) == len(PINNED_SCORES) == 19, len(fixtures)

    passes = pass_cases()
    broken = broken_cases()
    assert len(passes) == 8, len(passes)     # 4 gt + 4 correct
    assert len(broken) == 11, len(broken)

    # Every pinned key resolves to a real manifest entry (name integrity).
    names = {e.name for e in m.entries}
    for fx in fixtures:
        assert fx.entry_name in names, "pinned key %s not in manifest" % (
            fx.entry_name,)

    # Internal consistency of the pinned reference:
    #  1. every score is a valid metric output in [0, 1];
    #  2. every passing oracle scores exactly PASS_SCORE;
    #  3. every broken case scores strictly below the pass threshold -- both
    #     below every passing score AND below the IoU pass threshold, so the
    #     grader would have flagged each deliberate failure.
    for fx in fixtures:
        assert 0.0 <= fx.expected_score <= 1.0, (fx.entry_name, fx.expected_score)
    for fx in passes:
        assert fx.expected_score == PASS_SCORE, (fx.entry_name, fx.expected_score)
    worst_pass = min(f.expected_score for f in passes)
    for fx in broken:
        assert fx.expected_score < worst_pass, (
            "broken case did not score below the pass floor: %s -> %.3f"
            % (fx.entry_name, fx.expected_score))
        assert fx.expected_score < DEFAULT_IOU_THRESHOLD, (
            "broken case at/above the IoU pass threshold: %s -> %.3f"
            % (fx.entry_name, fx.expected_score))

    # Every broken case names a defect and one of the four builders; the
    # eleven brokens exercise all four adversarial builders.
    seen_builders = set()
    for fx in broken:
        assert fx.defect, "broken %s has no defect note" % fx.entry_name
        assert fx.builder in _BUILDERS, (fx.entry_name, fx.builder)
        seen_builders.add(fx.builder)
    assert seen_builders == set(_BUILDERS), seen_builders

    # File resolution census -- honest about manifest-only degrade-to-empty.
    avail = m.availability()
    present = avail["present"]
    if present == 0:
        print("SELFCHECK OK: manifest valid (32 entries, manifest-only); "
              "pinned scores internally consistent (8 pass @ 1.000, 11 broken "
              "< pass threshold, all in [0,1], ramp verified); no resources "
              "checkout, corpus degrades to empty as designed")
        return 0
    print("SELFCHECK OK: 32 entries, %d resolvable from resources; pinned "
          "scores internally consistent (8 pass @ 1.000, 11 broken below the "
          "pass threshold across all 4 builders); ramp ported and verified"
          % present)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="cadgenbench known-bad mating-feature jigs with pinned "
                    "reference scores (manifest-only, Apache-2.0).")
    parser.add_argument("--selfcheck", action="store_true",
                        help="validate the manifest, the ported scoring ramp, "
                             "and the internal consistency of the pinned "
                             "reference scores (every broken case below the "
                             "pass threshold).")
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
