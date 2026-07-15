"""THE PINNED BACKEND. Every F-rep session this experiment builds comes through
here, and it comes through here so that v1 and v2 are measuring the same ruler.

WHY THIS FILE EXISTS
--------------------
The F-rep backend takes a ``mesher`` and a ``resolution``, and it defaults them
from ``io/backends/frep.py``::

    DEFAULT_RESOLUTION = 48
    DEFAULT_MESHER = "marching_cubes"

Those are MODULE-LEVEL DEFAULTS IN CODE THIS EXPERIMENT DOES NOT OWN, and the
backend's own docstring is already mid-argument with itself about them -- it says
"``dual_contouring`` is THE DEFAULT" three lines above the line that sets the
default to ``marching_cubes``. Dual contouring places its cell vertex by a QEF
solve and is roughly two orders of magnitude more accurate on a sharp edge, so
somebody is going to flip that constant, and they are right to.

But a benchmark whose ruler changes underneath it is not a benchmark. If v2 runs
on a different mesher from v1, then every v1-vs-v2 delta in the report is
confounded with the mesher, and the one question this experiment exists to answer
-- did the arms change -- becomes unanswerable. Volume, bounding box and every SDF
probe are read off the tessellation.

SO: THE EXPERIMENT PINS THE MESHER, AND IT PINS IT TO WHAT v1 RAN.

    marching_cubes @ resolution 48

That is a deliberate choice of COMPARABILITY over ACCURACY, and it costs
something real: marching cubes structurally cannot represent a sharp edge, so
every measurement in this report carries marching-cubes' error. It is the same
error v1 carried, which is the entire point -- it cancels in the delta. A v3 that
wants the accurate ruler should re-run BOTH arms on it and republish both.

If ``DEFAULT_MESHER`` flips tomorrow, nothing here changes. That is the property
being bought.
"""

from __future__ import annotations

from typing import Any

__all__ = ["MESHER", "RESOLUTION", "frep_server", "pin"]

#: THE MESHER: what v1 ran, and therefore what v2 runs. NOT read from the
#: backend's default (which is, as of this run, already ``dual_contouring``).
MESHER = "marching_cubes"

#: THE RESOLUTION: 96, and NOT v1's 48. This is the one confound between v1 and
#: v2 that could not be avoided, and it was FORCED, not chosen.
#:
#: Since v1 ran, the F-rep backend gained a wall-resolution guard: it now REFUSES
#: a shell whose wall spans fewer than 2 grid cells, rather than building a wall
#: the grid cannot represent. That guard is correct and it is an improvement. It
#: also has a consequence nobody checked:
#:
#:     `trap_shell_too_thick` is a 60x40x5 plate. The cavity is empty unless
#:     2t < 5, so a FEASIBLE wall is t < 2.5 mm. At resolution 48 the cell is
#:     60/48 = 1.25 mm, so a BUILDABLE wall is t >= 2.5 mm.
#:
#:     THE TWO WINDOWS DO NOT OVERLAP. At resolution 48, against today's backend,
#:     that brief is unsolvable BY EVERY ANSWER -- including its own hand-written
#:     reference solution. A brief no answer can solve measures nothing; it just
#:     depresses every arm equally and adds noise.
#:
#: At resolution 96 the cell is 0.625 mm and the feasible window t in [1.25, 2.5)
#: is non-empty. Every one of the twelve briefs' reference solutions builds.
#:
#: So: v2's geometry is measured on a FINER grid than v1's, and the v1-vs-v2
#: solve rates are therefore NOT measured with the same ruler. This is stated in
#: the report, up front, and it is the reason v2 re-runs the blind and harness
#: arms rather than quoting v1's numbers for them.
RESOLUTION = 96


def pin(backend: Any) -> Any:
    """Force an F-rep backend onto the pinned ruler. Returns it."""
    if hasattr(backend, "mesher"):
        backend.mesher = MESHER
    if hasattr(backend, "resolution"):
        backend.resolution = RESOLUTION
    return backend


def frep_server(verify_level: str = "core"):
    """A CISPServer on the F-rep backend, pinned. The ONLY way this package
    builds one."""
    from harnesscad.io.surfaces.server import CISPServer

    server = CISPServer(backend="frep", verify_level=verify_level)
    pin(server.backend)
    return server
