"""Out-of-process worker for a single manifold3d metric boolean.

Why this file exists
--------------------
``metric_booleans`` moved metric intersections onto manifold3d because OCCT
booleans HANG on interface-overlay geometry. That LOWERED the hang probability;
it did not remove it. Upstream manifold's own fuzzers run every boolean under a
10s watchdog, and its ``ExecutionContext`` explicitly CANNOT interrupt a single
large boolean -- so a pathological ``a ^ b`` can still wedge the interpreter with
no timeout, and Python cannot interrupt an in-thread C-extension call.

The only real kill for a wedged C call is a separate PROCESS. This worker is that
process: it reads two meshes, ingests them into manifold3d, computes ``a ^ b`` and
reports the volume as a single JSON line -- exactly the subprocess-isolation
pattern :mod:`harnesscad.io.ingest.step_check` uses for untrusted STEP files. The
parent (:func:`harnesscad.eval.verifiers.metric_booleans.intersection_volume_isolated`)
runs this under a wall-clock budget and KILLS + reaps it on overrun, so a hostile
boolean's process dies while the parent survives.

Contract (identical shape to the step_check worker)
---------------------------------------------------
* Input: an ``.npz`` path (the final positional argument) holding four arrays --
  ``va``/``ta`` (mesh A vertices/triangles) and ``vb``/``tb`` (mesh B).
* Output: the LAST non-empty stdout line is a JSON object with ``status`` and, on
  success, ``volume``. ``status`` is ``ok`` (volume computed), ``empty`` (the
  intersection is empty -> volume 0.0), ``unavailable`` (no manifold3d/numpy) or
  ``error`` (a bad mesh / kernel refusal). Exit 0 for ok/empty, 1 otherwise.
* ``--stall SECONDS`` sleeps before the boolean. It is a TEST hook: it lets the
  self-check point the real worker at a hang so the parent's kill+reap path is
  exercised on the actual boolean worker, not on a stand-in.

Pure stdlib apart from manifold3d + numpy (already the metric path's deps).

ATTRIBUTION
-----------
``_ingest`` below is the same mesh->manifold conversion as
:func:`harnesscad.eval.verifiers.metric_booleans.mesh_to_manifold`, adapted
third-party code from cadgenbench, Copyright 2026 Hugging Face, Apache License
2.0. See the repository root ``THIRD-PARTY.md`` (licence text at
``THIRD-PARTY-LICENSES/Apache-2.0.txt``). The subprocess isolation itself is
HarnessCAD's own code.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import List, Optional, Sequence


def _emit(payload: dict) -> None:
    """Write the single JSON result line the parent reads."""
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _volume(va, ta, vb, tb) -> dict:
    """Ingest two meshes, intersect them, return the overlap volume payload."""
    try:
        import manifold3d as m3d
        import numpy as np
    except Exception as exc:  # noqa: BLE001 - no kernel -> parent degrades to None
        return {"status": "unavailable", "note": "manifold3d/numpy absent: %s" % exc}

    def _ingest(verts, tris):
        mesh = m3d.Mesh(
            vert_properties=np.array(verts, dtype=np.float32, order="C"),
            tri_verts=np.array(tris, dtype=np.uint32, order="C"))
        man = m3d.Manifold(mesh)
        status = man.status
        if hasattr(status, "name") and status.name != "NoError":
            return None
        if man.is_empty():
            return None
        return man

    a = _ingest(va, ta)
    b = _ingest(vb, tb)
    if a is None or b is None:
        return {"status": "error", "note": "a mesh was not a closed 2-manifold"}
    common = a ^ b  # the boolean this whole file exists to time-bound
    if common.is_empty():
        return {"status": "empty", "volume": 0.0}
    return {"status": "ok", "volume": abs(float(common.volume()))}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m harnesscad.eval.verifiers._metric_boolean_worker",
        description="Compute one manifold3d intersection volume, out of process.")
    parser.add_argument("--stall", type=float, default=0.0,
                        help="TEST hook: sleep this many seconds before the "
                             "boolean, so the parent's kill+reap path can be "
                             "exercised on the real worker.")
    parser.add_argument("input", help="path to the .npz holding va/ta/vb/tb")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.stall > 0.0:
        sys.stderr.write("stalling %g s\n" % args.stall)
        sys.stderr.flush()
        time.sleep(args.stall)

    try:
        import numpy as np
        data = np.load(args.input)
        va, ta = data["va"], data["ta"]
        vb, tb = data["vb"], data["tb"]
    except Exception as exc:  # noqa: BLE001
        _emit({"status": "error", "note": "cannot read input: %s" % exc})
        return 1

    payload = _volume(va, ta, vb, tb)
    _emit(payload)
    return 0 if payload.get("status") in ("ok", "empty") else 1


if __name__ == "__main__":
    raise SystemExit(main())
