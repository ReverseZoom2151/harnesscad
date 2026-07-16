"""manifold's crash-reproducer meshes as adversarial verifier inputs.

Source: the manifold geometry kernel (resources/cad_repos/manifold-master,
https://github.com/elalish/manifold, Apache-2.0), ``test/models/*.obj`` and
``test/polygons/polygon_corpus.txt``. Files under 100 KB are vendored into
``fixtures/manifold/`` (SHA-256 provenance in ``MANIFEST.json``, attribution
in ``LICENSE-NOTICE.txt``); the large ones (the self-intersection pair, the
hull pair, one boolean twin, ``Offset1``) stay manifest-only and resolve
against ``resources/`` when present.

Why these meshes matter: every one of them is a REAL regression artifact from
a production geometry kernel -- a mesh that crashed OpenSCAD, a pair whose
boolean broke manifold, self-intersecting shells from the wild. The harness's
defect-injection benchmark (:mod:`harnesscad.eval.quality.geometry
.defect_injection`) scores a verifier against four SYNTHETIC lie classes;
this loader supplies the natural complement: found-in-the-wild pathology.

Verifier interface (defect_injection's, verbatim): a verifier is any
``callable(Mesh) -> bool`` returning ``True`` when it judges the mesh SOUND,
over ``defect_injection.Mesh`` (vertices + face index tuples).
:func:`score_verifier` runs one over every available adversarial mesh and
reports, per mesh, the verdict against the mesh's labeled expectation:

* ``self_intersection`` -- geometrically self-intersecting shells. The
  expectation is honest: ``expected_sound=None``. Self-intersection is a
  GEOMETRIC lie, invisible to purely topological checks (defect_injection's
  own ``topology_verifier`` passes these), so a topological verifier is not
  charged a miss -- but a verifier CLAIMING geometric soundness checks should
  catch them, and the report shows exactly who does.
* ``non_manifold_crash`` -- the mesh that crashed OpenSCAD via a non-manifold
  configuration: ``expected_sound=False``. Measured here, not assumed: its
  INDEX topology is clean (every index edge has incidence 2 -- it fools
  ``defect_injection.topology_verifier``), but it carries 717 vertices on
  only 658 distinct positions, and welding the coincident vertices exposes 18
  edges with incidence 4. The pathology is POSITIONAL, which is exactly what
  makes this fixture worth more than any synthetic injector: it separates
  verifiers that look at index topology from verifiers that look at the
  geometry. :func:`weld_vertices` is provided so a topological verifier can
  be lifted to the positional view.
* ``boolean_operand`` -- left/right operands of boolean regressions: sound
  inputs (``expected_sound=True``) whose COMBINATION broke a kernel; also
  useful directly as boolean stressor pairs.
* ``offset_regression`` -- meshes from manifold's offset regression suite:
  ``expected_sound=True``.

``polygon_corpus.txt`` is manifold's triangulation fuzz corpus: named
multi-contour polygons (holes, degeneracies, near-epsilon slivers) with the
expected triangle count. :func:`polygon_samples` parses it; it is a ready
adversarial battery for any 2D triangulator or sketch-profile validator.

Stdlib only (plus the kernel-free defect_injection module). Deterministic.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from harnesscad.eval.corpus.fixtures import Manifest, load_manifest
from harnesscad.eval.quality.geometry.defect_injection import (
    Mesh,
    topology_verifier,
)

__all__ = [
    "AdversarialMesh",
    "PolygonSample",
    "manifest",
    "parse_obj",
    "weld_vertices",
    "adversarial_meshes",
    "boolean_pairs",
    "score_verifier",
    "polygon_samples",
    "main",
]

_SOURCE = "manifold"

#: role -> expected_sound. None = expectation depends on what the verifier
#: claims to check (self-intersection is invisible to pure topology).
_EXPECTED_SOUND: Dict[str, Optional[bool]] = {
    "self_intersection": None,
    "non_manifold_crash": False,
    "boolean_operand": True,
    "offset_regression": True,
}


@dataclass(frozen=True)
class AdversarialMesh:
    """One labeled pathological (or stressor) mesh from manifold's suite."""

    name: str
    role: str
    path: Optional[Path]
    expected_sound: Optional[bool]
    sha256: str

    @property
    def available(self) -> bool:
        return self.path is not None

    def load(self) -> Mesh:
        if self.path is None:
            raise FileNotFoundError(
                "mesh %r is not vendored and resources/ is absent" % self.name)
        return parse_obj(self.path)


@dataclass(frozen=True)
class PolygonSample:
    """One entry of manifold's triangulation fuzz corpus."""

    name: str
    expected_num_triangles: int
    precision: float
    polygons: Tuple[Tuple[Tuple[float, float], ...], ...]


def manifest() -> Manifest:
    return load_manifest(_SOURCE)


def parse_obj(path: Path) -> Mesh:
    """A minimal OBJ reader: ``v`` and ``f`` records into a Mesh.

    Handles ``f v/vt/vn`` slash syntax and negative (relative) indices; the
    output face tuples are 0-based, which is what defect_injection expects.
    """
    vertices: List[Tuple[float, float, float]] = []
    faces: List[Tuple[int, ...]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append(
                    (float(parts[1]), float(parts[2]), float(parts[3])))
            elif parts[0] == "f" and len(parts) >= 4:
                idx: List[int] = []
                for token in parts[1:]:
                    raw = token.split("/", 1)[0]
                    i = int(raw)
                    idx.append(i - 1 if i > 0 else len(vertices) + i)
                faces.append(tuple(idx))
    return Mesh.of(vertices, faces)


def weld_vertices(mesh: Mesh) -> Mesh:
    """Merge exactly-coincident vertices so positional defects become
    topological.

    A mesh can carry duplicate vertices at the same position; its index
    topology then looks closed while the actual surface is torn or
    non-manifold (the OpenSCAD crash mesh does exactly this). Welding maps
    every face index to the first vertex at that position, after which an
    index-topology verifier sees the real incidence structure.
    """
    first_at: Dict[Tuple[float, float, float], int] = {}
    remap: List[int] = []
    welded: List[Tuple[float, float, float]] = []
    for v in mesh.vertices:
        if v in first_at:
            remap.append(first_at[v])
        else:
            first_at[v] = len(welded)
            remap.append(len(welded))
            welded.append(v)
    faces = tuple(tuple(remap[i] for i in f) for f in mesh.faces)
    return Mesh(tuple(welded), faces)


def adversarial_meshes() -> List[AdversarialMesh]:
    m = manifest()
    out: List[AdversarialMesh] = []
    for e in m.entries:
        if e.format != "obj":
            continue
        out.append(AdversarialMesh(
            name=e.name,
            role=e.role,
            path=m.resolve(e),
            expected_sound=_EXPECTED_SOUND.get(e.role),
            sha256=e.sha256,
        ))
    return out


def boolean_pairs() -> List[Tuple[AdversarialMesh, AdversarialMesh]]:
    """The left/right operand pairs of manifold's boolean regressions."""
    operands = {a.name: a for a in adversarial_meshes()
                if a.role == "boolean_operand"}
    pairs: List[Tuple[AdversarialMesh, AdversarialMesh]] = []
    for name, left in sorted(operands.items()):
        if not name.endswith("_left"):
            continue
        right = operands.get(name[:-len("_left")] + "_right")
        if right is not None:
            pairs.append((left, right))
    # hull-body / hull-mask are a pair with their own naming scheme.
    body, mask = operands.get("hull-body"), operands.get("hull-mask")
    if body is not None and mask is not None:
        pairs.append((body, mask))
    return pairs


def score_verifier(
    verifier: Callable[[Mesh], bool] = topology_verifier,
) -> Dict[str, dict]:
    """Run a defect_injection-style verifier over every available mesh.

    Returns ``{name: {role, judged_sound, expected_sound, agrees}}`` where
    ``agrees`` is None when the expectation itself is None (self-intersection
    against a verifier whose scope is unknown to this loader).
    """
    results: Dict[str, dict] = {}
    for a in adversarial_meshes():
        if not a.available:
            results[a.name] = {"role": a.role, "skipped": "not present"}
            continue
        try:
            judged = bool(verifier(a.load()))
        except Exception as exc:  # noqa: BLE001 - a crash IS a verdict here
            results[a.name] = {"role": a.role, "crashed": repr(exc),
                               "expected_sound": a.expected_sound,
                               "agrees": False}
            continue
        agrees = None if a.expected_sound is None else (
            judged == a.expected_sound)
        results[a.name] = {"role": a.role, "judged_sound": judged,
                           "expected_sound": a.expected_sound,
                           "agrees": agrees}
    return results


def polygon_samples() -> List[PolygonSample]:
    """Parse manifold's triangulation fuzz corpus (empty if absent)."""
    m = manifest()
    e = m.by_name("polygon_corpus")
    path = m.resolve(e) if e is not None else None
    if path is None:
        return []
    tokens: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        tokens.extend(line.split())
    samples: List[PolygonSample] = []
    i = 0
    while i < len(tokens):
        name = tokens[i]
        expected = int(tokens[i + 1])
        precision = float(tokens[i + 2])
        n_polys = int(tokens[i + 3])
        i += 4
        polys: List[Tuple[Tuple[float, float], ...]] = []
        for _ in range(n_polys):
            n_pts = int(tokens[i])
            i += 1
            pts = tuple(
                (float(tokens[i + 2 * k]), float(tokens[i + 2 * k + 1]))
                for k in range(n_pts))
            i += 2 * n_pts
            polys.append(pts)
        samples.append(PolygonSample(name, expected, precision, tuple(polys)))
    return samples


def _selfcheck() -> int:
    m = manifest()
    assert m.license == "Apache-2.0", m.license
    problems = m.verify_vendored()
    assert not problems, "; ".join(problems)
    meshes = adversarial_meshes()
    assert len(meshes) == 17, "expected 17 obj entries, got %d" % len(meshes)
    roles = {a.role for a in meshes}
    assert roles == set(_EXPECTED_SOUND), roles

    available = [a for a in meshes if a.available]
    if not available:
        print("SELFCHECK OK: manifest valid (%d entries); no mesh "
              "resolvable, corpus degrades to empty as designed"
              % len(m.entries))
        return 0

    # Every available mesh must parse into a non-trivial Mesh.
    for a in available:
        mesh = a.load()
        assert len(mesh.vertices) >= 4 and len(mesh.faces) >= 4, a.name
    pairs = boolean_pairs()
    assert pairs, "no boolean operand pair resolvable"

    # Score the built-in index-topology verifier over the corpus, and prove
    # the crash mesh's documented trap: index topology says SOUND, welded
    # (positional) topology says UNSOUND. Both assertions are the fixture's
    # value, executable.
    scored = score_verifier(topology_verifier)
    for name, r in scored.items():
        assert "crashed" not in r, "%s crashed the verifier: %r" % (name, r)
    crash = scored.get("openscad-nonmanifold-crash")
    if crash is not None and "skipped" not in crash:
        assert crash["judged_sound"] is True, (
            "the crash mesh no longer fools index topology -- the fixture's "
            "documented trap changed: %r" % crash)
        crash_mesh = next(a for a in available
                          if a.name == "openscad-nonmanifold-crash").load()
        assert topology_verifier(weld_vertices(crash_mesh)) is False, (
            "welded topology failed to expose the crash mesh's "
            "non-manifold edges")

    corpus = polygon_samples()
    if corpus:
        assert len(corpus) >= 10, "polygon corpus parsed only %d" % len(corpus)
        for s in corpus:
            assert s.polygons and all(len(p) >= 3 for p in s.polygons), s.name
    print("SELFCHECK OK: %d/%d meshes present (%d boolean pairs), crash-mesh "
          "trap verified (index topology fooled, welded topology catches it), "
          "%d polygon fuzz samples parsed"
          % (len(available), len(meshes), len(pairs), len(corpus)))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="manifold crash-reproducer meshes + triangulation fuzz "
                    "corpus as adversarial verifier inputs (Apache-2.0).")
    parser.add_argument("--selfcheck", action="store_true",
                        help="validate manifest/hashes, parse every "
                             "available mesh and score the built-in "
                             "topology verifier over the corpus.")
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
