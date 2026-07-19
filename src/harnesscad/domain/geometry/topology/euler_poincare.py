"""Euler-Poincare B-rep count balance: the handle a closedness check cannot see.

``topology/sew.py`` sews faces into shells and reports two validity signals:
``free_edges`` (edges without exactly two opposite coedges) and tolerance
monotonicity. Both are LOCAL: they interrogate one edge, or one vertex/edge/face
triple, at a time. A shell can pass both and still be the wrong solid, because
neither signal can count.

The gap is provable on the harness's own sewing (both cases are asserted in
:func:`main` below, built through ``sew_faces`` -- no hand-written numbers):

    a sewn cube          V=8  E=12 F=6  S=1  is_closed=True  free_edges=()
    a sewn square torus  V=16 E=32 F=16 S=1  is_closed=True  free_edges=()

The torus is a genus-1 solid -- it has a hole through it. ``sew.py`` reports it
as clean, because it IS clean: every edge has exactly two opposite coedges. Only
a global count balance separates them:

    V - E + F  =  2  (cube)        V - E + F  =  0  (torus)

That is the Euler-Poincare formula, and it is the cheapest global topology
oracle there is: five integers and no geometry.

THE FORMULA, AND THE RING TERM
For a B-rep solid the generalised statement is

    V - E + F - R = 2 * (S - G)

with R = rings (inner loops -- the holes IN faces, e.g. the annular top of a
bored block), S = shells, G = genus (handles THROUGH the solid). Counting loops
L rather than rings gives R = L - F (every face has exactly one outer loop), so
the residual under a genus-0 assumption is ``V - E + 2F - L - 2S``.

The ring term is not decoration, and dropping it is a trap that fails SILENTLY
in the dangerous direction. Take a cube with a square hole bored through it,
modelled the natural way -- top and bottom are each ONE face carrying an outer
loop plus an inner ring:

    V=16  E=24  F=10  L=12  S=1   (so R = L - F = 2)

    naive:  V - E + F      = 16 - 24 + 10     =  2   -> "genus 0"  WRONG
    full:   V - E + F - R  = 16 - 24 + 10 - 2 =  0   -> genus 1    CORRECT

The naive form returns exactly 2 -- indistinguishable from a plain cube -- so a
bored block PASSES a naive check while being a torus. The ring term is the only
thing standing between those two answers. (Split each annulus into four quads
instead and R collapses to 0, which is why the sewn torus above reads 0 rather
than 2: same solid, different face decomposition, and both are handled here.)

WHAT THIS PRESUMES: EVERY FACE IS A DISK
The formula models faces that are topological disks. A SEAM face -- a sphere as
one periodic face, a full-revolve tube -- has a loop that bounds no disk, and
the formula simply does not describe it. The honest response is to EXCLUDE such
faces from the check, not to tune a fudge factor: :func:`check_euler_poincare`
takes ``disk_faces=False`` and refuses to report a genus rather than reporting a
wrong one.

This matters less for ``sew.py`` than it might: ``SewFace`` is an ordered
boundary polygon and ``TopFace`` carries exactly ONE loop, so every sewn face is
a planar polygon (a disk) and every sewn solid has L = F and R = 0 by
construction. :func:`counts_from_sew_result` therefore sets L = F, and that is a
fact about the harness's B-rep, not an assumption about the caller's.

Attribution (facts and design only; NO code or text is copied):
  * The Euler-Poincare relation for B-reps is classical solid-modelling theory
    (Braid; Mantyla, "An Introduction to Solid Modeling") and is nobody's IP.
  * The genus-0 RESIDUAL framing -- expressing the check as the single integer
    ``V - E + 2F - L - 2S`` that must be 0, counting total loops L instead of
    rings R, and pairing it with structural cleanliness and orientation
    consistency to form a full topology contract -- is the design used by
    Roshera-CAD (Roshera-CAD-main
    ``roshera-backend/geometry-engine/src/harness/brep_integrity.rs``:
    ``BRepIntegrityReport::euler_poincare_genus0_residual`` and
    ``is_genus0_manifold``).
  * The disk-face exclusion is Roshera's documented scar, from the same file's
    ``every_operation_is_orientation_consistent_and_euler_balanced`` test: its
    ``EULER_SKIP`` list excludes a torus and a full-revolve tube (where "the
    genus-0 residual is correctly -2*genus") and the sphere ("a single SEAMLESS
    face whose loop bounds no disk, which the disk-face Euler-Poincare formula
    does not model"). Its ``tests/boolean_fuzz_survey.rs`` reinforces the point
    by calibrating a UV-sphere's intrinsic nonzero residual from seam and poles
    before asserting on it, and by treating EULER as a SOFT signal for that
    reason.
  * Roshera-CAD is licensed FSL-1.1 (Functional Source License, non-compete --
    NOT permissive; Copyright (c) 2025-2026 Varun Sharma). Nothing from it is
    copied: the formula is classical, the residual/skip facts are restated in
    this module's own words with citation, and every line below is written here.

Never raises. Pure stdlib, deterministic; no kernel, no geometry -- five
integers in, a verdict out.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

__all__ = [
    "TopologyCounts",
    "counts_from_sew_result",
    "check_euler_poincare",
    "implied_genus",
]


@dataclass(frozen=True)
class TopologyCounts:
    """The five integers the Euler-Poincare relation is stated over.

    ``loops`` counts ALL loops (outer + inner) across every face, so rings are
    ``loops - faces``. Counting loops rather than rings is what makes the number
    readable straight off a B-rep traversal, which never has to decide which
    loop of a face is "the outer one".
    """

    vertices: int
    edges: int
    faces: int
    loops: int
    shells: int

    @property
    def rings(self) -> int:
        """Inner loops: ``loops - faces`` (one outer loop per face)."""
        return self.loops - self.faces

    @property
    def euler_characteristic(self) -> int:
        """``V - E + F - R`` -- the ring-corrected characteristic.

        This is the value that equals ``2*(S - G)``. It is NOT ``V - E + F``:
        see the module docstring's bored-cube case, where the two differ by
        exactly the ring count and only this one is right.
        """
        return self.vertices - self.edges + self.faces - self.rings

    @property
    def naive_characteristic(self) -> int:
        """``V - E + F``, ignoring rings. Kept only to expose the trap."""
        return self.vertices - self.edges + self.faces

    def genus0_residual(self) -> int:
        """``V - E + 2F - L - 2S``: 0 iff the counts describe a genus-0 solid.

        Equivalent to ``euler_characteristic - 2*shells``. For a solid of genus
        G the residual is exactly ``-2*G``, which is why it reads -2 for a
        torus rather than merely "nonzero".
        """
        return self.euler_characteristic - 2 * self.shells

    def to_dict(self) -> Dict[str, int]:
        return {
            "vertices": self.vertices, "edges": self.edges,
            "faces": self.faces, "loops": self.loops,
            "shells": self.shells, "rings": self.rings,
        }


def _counts_sane(c: TopologyCounts) -> Optional[str]:
    """Why these counts cannot describe any B-rep, or ``None`` if they might."""
    for name in ("vertices", "edges", "faces", "loops", "shells"):
        if getattr(c, name) < 0:
            return f"{name} is negative ({getattr(c, name)})"
    if c.shells < 1:
        return "a solid has at least one shell (shells=0)"
    if c.rings < 0:
        return (f"loops ({c.loops}) < faces ({c.faces}): every face carries at "
                f"least one outer loop, so loops >= faces always")
    return None


def implied_genus(counts: TopologyCounts) -> Optional[int]:
    """Genus implied by the counts, or ``None`` if they imply no integer.

    From ``V - E + F - R = 2*(S - G)``: ``G = S - (V - E + F - R)/2``. An ODD
    characteristic has no integer solution, which means the counts are
    inconsistent (something was miscounted) rather than that the solid is
    exotic -- so this returns ``None`` instead of rounding to a plausible lie.
    """
    chi = counts.euler_characteristic
    if chi % 2 != 0:
        return None
    return counts.shells - chi // 2


def check_euler_poincare(
    counts: TopologyCounts,
    *,
    expect_genus: Optional[int] = 0,
    disk_faces: bool = True,
) -> Dict[str, Any]:
    """Check a B-rep's count balance. Never raises.

    ``expect_genus=0`` demands a simply-connected solid (the usual contract for
    a sewn or booleaned result); ``expect_genus=None`` asks only that the counts
    be self-consistent, and reports whatever genus they imply.

    ``disk_faces=False`` declares that some face is a seam/periodic face whose
    loop bounds no disk. The formula does not model that, so the check REFUSES
    (``ok`` is None, ``status="not_modelled"``) instead of returning a verdict
    it cannot justify. That refusal is the whole point: a wrong genus asserted
    confidently is worse than no genus at all.

    Returns a dict with ``ok`` (True/False/None), ``status``, ``residual``,
    ``genus``, ``counts`` and ``notes``.
    """
    base: Dict[str, Any] = {
        "ok": False, "status": "error", "residual": None, "genus": None,
        "counts": None, "notes": [],
    }
    if not isinstance(counts, TopologyCounts):
        base["notes"] = [f"expected TopologyCounts, got {type(counts).__name__}"]
        return base
    base["counts"] = counts.to_dict()

    bad = _counts_sane(counts)
    if bad is not None:
        base["status"] = "invalid_counts"
        base["notes"] = [bad]
        return base

    residual = counts.genus0_residual()
    genus = implied_genus(counts)
    base["residual"] = residual

    if not disk_faces:
        base["ok"] = None
        base["status"] = "not_modelled"
        base["notes"] = [
            "a seam/periodic face (sphere-as-one-face, full-revolve tube) has a "
            "loop bounding no disk; the disk-face Euler-Poincare formula does "
            "not model it, so no genus is reported",
            f"residual would have been {residual}, but it is not meaningful here",
        ]
        return base

    if genus is None:
        base["status"] = "inconsistent"
        base["notes"] = [
            f"characteristic V-E+F-R = {counts.euler_characteristic} is odd, so "
            f"2*(S-G) has no integer solution: the counts are miscounted",
        ]
        return base
    base["genus"] = genus

    notes = []
    if counts.rings and counts.naive_characteristic != counts.euler_characteristic:
        notes.append(
            f"rings={counts.rings}: the naive V-E+F is "
            f"{counts.naive_characteristic}, the ring-corrected value is "
            f"{counts.euler_characteristic} -- only the latter is the genus")
    if genus < 0:
        base["status"] = "inconsistent"
        notes.append(f"implied genus {genus} is negative: counts are miscounted")
        base["notes"] = notes
        return base

    if expect_genus is None:
        base["ok"] = True
        base["status"] = "consistent"
        notes.append(f"counts are self-consistent; solid has genus {genus}")
        base["notes"] = notes
        return base

    if genus == expect_genus:
        base["ok"] = True
        base["status"] = "balanced"
        notes.append(f"balanced: genus {genus} as expected (residual {residual})")
    else:
        base["status"] = "unbalanced"
        notes.append(
            f"expected genus {expect_genus} but the counts imply {genus} "
            f"(residual {residual}; a genus-G solid has residual -2*G). "
            f"On a result that should be simply connected this means missing or "
            f"extra topology -- a dropped face, or an unexpected handle")
    base["notes"] = notes
    return base


def counts_from_sew_result(result: Any) -> TopologyCounts:
    """Read :class:`TopologyCounts` off ``topology/sew.py``'s ``SewResult``.

    ``loops`` is set to ``len(faces)``: ``sew.py``'s ``TopFace`` carries exactly
    one loop, so a sewn B-rep can never express a ring (R = 0). That is a
    property of that module's face type, not an assumption -- a sewn face is
    always a planar boundary polygon, hence always a disk.
    """
    return TopologyCounts(
        vertices=len(result.vertices),
        edges=len(result.edges),
        faces=len(result.faces),
        loops=len(result.faces),
        shells=len(result.shells),
    )


# --------------------------------------------------------------------------- #
# selfcheck
# --------------------------------------------------------------------------- #

def _sew_cube():
    from harnesscad.domain.geometry.topology.sew import SewFace, sew_faces
    p = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
         (4, 4, 4), (4, 4, 4), (4, 4, 4), (4, 4, 4)]
    p = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
         (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    quads = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
             [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    return sew_faces([SewFace(boundary=[p[i] for i in q]) for q in quads],
                     tol=1e-6)


def _sew_square_torus():
    """A genus-1 frame: annuli split into quads, so no face carries a ring."""
    from harnesscad.domain.geometry.topology.sew import SewFace, sew_faces
    a = [(0, 0, 0), (4, 0, 0), (4, 4, 0), (0, 4, 0)]
    b = [(1, 1, 0), (3, 1, 0), (3, 3, 0), (1, 3, 0)]
    c = [(x, y, 1) for (x, y, _) in a]
    d = [(x, y, 1) for (x, y, _) in b]
    quads = []
    for i in range(4):
        j = (i + 1) % 4
        quads.append([a[i], b[i], b[j], a[j]])   # bottom annulus
        quads.append([c[i], c[j], d[j], d[i]])   # top annulus
        quads.append([a[i], a[j], c[j], c[i]])   # outer wall
        quads.append([b[i], d[i], d[j], b[j]])   # inner wall
    return sew_faces([SewFace(boundary=f) for f in quads], tol=1e-6)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Euler-Poincare B-rep count balance (V-E+F-R = 2(S-G)): the "
                    "global topology oracle sew.py's closedness check cannot be.")
    parser.add_argument("--selfcheck", action="store_true",
                        help="prove the balance on real sew_faces() output and "
                             "on the ring trap")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.selfcheck:
        parser.print_help()
        return 0

    # 1. A real sewn cube: genus 0, balanced.
    cube = _sew_cube()
    cc = counts_from_sew_result(cube)
    assert (cc.vertices, cc.edges, cc.faces, cc.shells) == (8, 12, 6, 1), cc
    assert all(s.is_closed for s in cube.shells) and not cube.free_edges
    assert cc.euler_characteristic == 2 and cc.rings == 0
    r = check_euler_poincare(cc)
    assert r["ok"] is True and r["status"] == "balanced" and r["genus"] == 0
    assert r["residual"] == 0
    print("[selfcheck] sewn cube: V=8 E=12 F=6 S=1 -> V-E+F=2, genus 0, "
          "residual 0 (balanced)")

    # 2. THE POINT. A real sewn genus-1 torus that sew.py calls perfectly clean:
    #    every edge has two opposite coedges, zero free edges. Only the count
    #    balance sees the handle.
    torus = _sew_square_torus()
    tc = counts_from_sew_result(torus)
    assert (tc.vertices, tc.edges, tc.faces, tc.shells) == (16, 32, 16, 1), tc
    assert all(s.is_closed for s in torus.shells), "torus should sew closed"
    assert torus.free_edges == (), "torus should have no free edges"
    t = check_euler_poincare(tc)
    assert t["ok"] is False and t["status"] == "unbalanced"
    assert t["genus"] == 1 and t["residual"] == -2, t
    print("[selfcheck] sewn torus: is_closed=True, free_edges=() -- sew.py sees "
          "NOTHING wrong; Euler residual -2 => genus 1 (the handle)")
    # A genus-G solid has residual exactly -2G, not merely "nonzero".
    assert t["residual"] == -2 * t["genus"]
    # Asked only for consistency, the same counts are fine -- a torus is a real
    # solid, it just is not simply connected.
    t2 = check_euler_poincare(tc, expect_genus=None)
    assert t2["ok"] is True and t2["status"] == "consistent" and t2["genus"] == 1
    assert check_euler_poincare(tc, expect_genus=1)["ok"] is True
    print("[selfcheck] same torus passes expect_genus=1/None: the check reports "
          "genus, it does not moralise about it")

    # 3. The ring trap, hand-counted (sew.py cannot express a ring, so this is
    #    the case the harness would meet only from a kernel/import).
    #    Bored cube, annular top+bottom: V=16 E=24 F=10 L=12 S=1 -> R=2.
    bored = TopologyCounts(vertices=16, edges=24, faces=10, loops=12, shells=1)
    assert bored.rings == 2
    assert bored.naive_characteristic == 2, "the trap: naive V-E+F reads 2"
    assert bored.euler_characteristic == 0, "ring-corrected reads 0"
    b = check_euler_poincare(bored)
    assert b["ok"] is False and b["genus"] == 1 and b["residual"] == -2
    assert any("naive" in n for n in b["notes"])
    # A plain cube and a bored cube are INDISTINGUISHABLE to the naive form...
    assert bored.naive_characteristic == cc.naive_characteristic == 2
    # ...and separated by the ring term alone.
    assert bored.euler_characteristic != cc.euler_characteristic
    print("[selfcheck] ring trap: bored cube V-E+F=2 (identical to a plain "
          "cube!) but V-E+F-R=0 => genus 1. The ring term is the only "
          "difference")

    # 4. Seam faces are refused, not guessed. (Roshera's EULER_SKIP scar.)
    sphere = TopologyCounts(vertices=0, edges=0, faces=1, loops=1, shells=1)
    s = check_euler_poincare(sphere, disk_faces=False)
    assert s["ok"] is None and s["status"] == "not_modelled" and s["genus"] is None
    assert any("bounding no disk" in n for n in s["notes"])
    # The same counts, wrongly declared disk-faced, are caught a SECOND way:
    # V-E+F-R = 1 is odd, so 2*(S-G) has no integer solution and the genus is
    # refused again -- as `inconsistent` rather than `not_modelled`. Two
    # independent refusals, and neither invents a number. (This assertion
    # originally demanded `genus is not None`, expecting the mis-declared case
    # to report a wrong genus; the parity check gets there first, which is the
    # stronger behaviour. The test was weaker than the code.)
    s_bad = check_euler_poincare(sphere, disk_faces=True)
    assert s_bad["ok"] is False and s_bad["status"] == "inconsistent"
    assert s_bad["genus"] is None and s_bad["residual"] == -1
    print("[selfcheck] seam face (sphere as one periodic face): refused with "
          "ok=None/not_modelled; mis-declared as disk-faced it is refused "
          "AGAIN by parity (odd characteristic), never with a guessed genus")

    # 5. Never-raise contract, incl. inconsistent counts.
    odd = TopologyCounts(vertices=1, edges=0, faces=0, loops=0, shells=1)
    o = check_euler_poincare(odd)
    assert o["ok"] is False and o["status"] == "inconsistent"
    assert o["genus"] is None and implied_genus(odd) is None
    assert check_euler_poincare(
        TopologyCounts(-1, 0, 0, 0, 1))["status"] == "invalid_counts"
    assert check_euler_poincare(
        TopologyCounts(8, 12, 6, 3, 1))["status"] == "invalid_counts"  # L < F
    assert check_euler_poincare(
        TopologyCounts(8, 12, 6, 6, 0))["status"] == "invalid_counts"  # S = 0
    assert check_euler_poincare("not counts")["status"] == "error"
    for bad in (None, 42, [], TopologyCounts(0, 0, 0, 0, 0)):
        assert isinstance(check_euler_poincare(bad), dict)
    print("[selfcheck] never raises: odd characteristic -> 'inconsistent' "
          "(not a rounded lie); negative/L<F/S=0/garbage -> dicts")

    # 6. Two shells (a solid with a void) balance at residual 0 for genus 0:
    #    a cube with a cube-shaped cavity is two closed genus-0 shells.
    hollow = TopologyCounts(vertices=16, edges=24, faces=12, loops=12, shells=2)
    assert hollow.rings == 0
    assert hollow.euler_characteristic == 4 == 2 * hollow.shells
    h = check_euler_poincare(hollow)
    assert h["ok"] is True and h["genus"] == 0 and h["residual"] == 0
    print("[selfcheck] hollow cube (2 shells, a void): chi=4=2S, genus 0, "
          "balanced -- shells are counted, not assumed to be 1")

    print("[selfcheck] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
