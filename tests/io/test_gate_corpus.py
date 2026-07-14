"""The total-correctness corpus for the output gate.

Split out of ``tests/io/test_gate.py`` because it is the one test in the suite
that meshes 200 parts, and meshing got 5.5x more expensive when the default
mesher became dual contouring (the QEF needs Hermite data -- field samples --
where marching cubes needs only the grid it already has). Together the two were
120 s in one module, on a 120 s CI budget; apart they are ~40 s and ~78 s and
NOTHING was traded away to get there. Shrinking the corpus was tried and rejected:
at 140 streams the rare refusal paths (self-intersecting, shell-wrong-wall) stop
appearing at all and the "streams reached the gate" floor breaks. The rare paths
are exactly the paths worth having.

The fixtures live in ``tests.io.test_gate``; this module owns only the corpus.
"""

from __future__ import annotations

import os
import random
import unittest

from harnesscad.io import gate
from harnesscad.io.backends.frep import FRepBackend
from harnesscad.io.formats import registry as formats
from harnesscad.io.formats import stl as stl_codec

from tests.io.test_gate import (
    CORPUS_MESHERS,
    CORPUS_RESOLUTION,
    CORPUS_SEED,
    CORPUS_SIZE,
    TempDirTest,
)
from harnesscad.core.cisp.ops import (
    AddCircle, AddRectangle, Boolean, Chamfer, Extrude, Fillet, Hole,
    LinearPattern, Mirror, NewSketch, Shell,
)


# ---------------------------------------------------------------------------
# 7. THE TOTAL-CORRECTNESS CLAIM.
# ---------------------------------------------------------------------------

def random_stream(rng):
    """A randomly-parameterised but structurally-plausible op stream.

    Deliberately includes parameters that CANNOT work -- a wall thicker than the
    part, a hole wider than the stock -- because those must be REFUSED, not
    shipped, and a corpus of only-valid parts would prove nothing.
    """
    ops = [NewSketch(plane=rng.choice(["XY", "YZ", "XZ"]))]
    if rng.random() < 0.3:
        ops.append(AddCircle(sketch="sk1", cx=0.0, cy=0.0, r=rng.uniform(4.0, 40.0)))
    else:
        ops.append(AddRectangle(sketch="sk1", x=0.0, y=0.0,
                                w=rng.uniform(8.0, 80.0), h=rng.uniform(8.0, 80.0)))
    ops.append(Extrude(sketch="sk1", distance=rng.uniform(1.0, 40.0)))

    for _ in range(rng.randint(0, 2)):
        pick = rng.random()
        if pick < 0.30:
            ops.append(Shell(faces=[], thickness=rng.uniform(0.5, 12.0)))
        elif pick < 0.55:
            ops.append(Hole(face_or_sketch="f1", x=0.0, y=0.0,
                            diameter=rng.uniform(2.0, 50.0),
                            depth=rng.uniform(1.0, 30.0),
                            through=rng.random() < 0.5, kind="simple"))
        elif pick < 0.70:
            ops.append(Fillet(edges=["f1"], radius=rng.uniform(0.5, 10.0)))
        elif pick < 0.82:
            ops.append(Chamfer(edges=["f1"], distance=rng.uniform(0.5, 8.0)))
        elif pick < 0.92:
            ops.append(Mirror(feature_or_body="",
                              plane=rng.choice(["XY", "YZ", "XZ"])))
        else:
            ops.append(LinearPattern(
                feature="f1",
                direction=rng.choice([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]),
                count=rng.randint(2, 4), spacing=rng.uniform(5.0, 40.0)))
    return ops


class PropertyTotalCorrectnessTest(TempDirTest):
    """The soundness claim, as a test rather than a promise."""

    def test_the_third_outcome_never_occurs(self):
        """Over the corpus, no file is ever written that fails measurement.

        For every stream the harness is asked to export, EXACTLY ONE of:

            (a) a file is written -- and re-measuring that file, read back off the
                disk, finds nothing wrong with it; or
            (b) a typed refusal is raised, and NO file exists.

        The forbidden third outcome -- a file on disk that fails the measured
        checks -- is asserted never to occur.
        """
        rng = random.Random(CORPUS_SEED)
        shipped = refused = plan_rejected = 0
        third_outcome = []
        refusal_codes = {}

        for i in range(CORPUS_SIZE):
            # The mesher is PINNED, and it alternates. This test is about the
            # GATE, but the gate only ever sees geometry a mesher produced, and
            # the two meshers fail differently: over this corpus dual contouring
            # is the only one that produces a SELF-INTERSECTING mesh (MC: 0/200,
            # DC: 1/200), and marching cubes is the cheaper way to hit the
            # winding/manifold paths. Leaving it on the default would test only
            # whichever mesher is currently default -- which is exactly how the
            # self-intersecting refusal path went untested until the flip.
            # Alternating covers both defect profiles and keeps the corpus inside
            # the module's runtime budget (DC-only: 110 s; MC-only: 38 s;
            # alternating: ~75 s).
            mesher = CORPUS_MESHERS[i % len(CORPUS_MESHERS)]
            backend = FRepBackend(resolution=CORPUS_RESOLUTION, mesher=mesher)
            backend.reset()

            if not all(backend.apply(op).ok for op in random_stream(rng)):
                plan_rejected += 1        # the backend refused the plan: never
                continue                  # reached the gate, so nothing was emitted

            out = self.path("part_%03d.stl" % i)
            try:
                formats.export_session(backend, out)
            except gate.InvalidArtifact as exc:
                refused += 1
                self.assertFalse(
                    os.path.exists(out),
                    "stream %d: REFUSED, but the file was written anyway" % i)
                self.assertTrue(exc.failures, "a refusal must be typed")
                for f in exc.failures:
                    refusal_codes[f.check] = refusal_codes.get(f.check, 0) + 1
                continue
            except formats.FormatError:
                refused += 1              # also a refusal: typed, nothing written
                self.assertFalse(os.path.exists(out))
                continue

            # (a) A file was written. It must survive an INDEPENDENT re-measurement
            # read back off the disk. We do not trust the writer.
            self.assertTrue(os.path.exists(out))
            with open(out, "rb") as fh:
                verts, faces = gate._weld(stl_codec.parse_stl(fh.read()))
            residual = gate.measured_failures(gate.measure(verts, faces))
            if residual:
                third_outcome.append((i, [f.check for f in residual]))
            shipped += 1

        self.assertEqual(
            third_outcome, [],
            "THE THIRD OUTCOME OCCURRED: %d artifact(s) were WRITTEN to disk that "
            "fail the measured checks: %r" % (len(third_outcome), third_outcome[:5]))

        # The corpus must actually exercise both outcomes, or the claim is vacuous.
        self.assertGreaterEqual(shipped, 20, "too few parts shipped; corpus too easy")
        self.assertGreaterEqual(
            refused, 1,
            "nothing was refused: the corpus never stressed the gate (codes: %r)"
            % (refusal_codes,))
        self.assertGreaterEqual(shipped + refused, 100,
                                "too few streams reached the gate")
        # COVERAGE PIN. Dual contouring is the only mesher in this corpus that
        # produces a self-intersecting mesh (stream 159), and it is the mesher we
        # ship. If this code stops appearing, the corpus has stopped exercising a
        # refusal path -- do not delete this assertion, work out what changed.
        self.assertIn(
            "self-intersecting", refusal_codes,
            "the corpus no longer exercises the self-intersecting refusal path "
            "(codes: %r)" % (refusal_codes,))
        print("\n  corpus: %d streams | %d shipped-valid | %d refused | "
              "%d rejected pre-gate | THIRD OUTCOME: %d\n  refusal codes: %r"
              % (CORPUS_SIZE, shipped, refused, plan_rejected, len(third_outcome),
                 refusal_codes))


if __name__ == "__main__":
    unittest.main()
