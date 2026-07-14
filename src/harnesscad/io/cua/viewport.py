"""Deterministic 3D-viewport interaction: project, click, and VERIFY the pick.

The 3D viewport is the one surface of a CAD GUI that has no accessibility tree.
The ribbon, the dialogs and the feature tree are all enumerable Qt widgets with
names and bounding boxes (see :mod:`harnesscad.io.cua.uia`); the viewport is a
single opaque ``QOpenGLWidget``. ``Face7`` has no name and no box in any DOM
sense, so every GUI-grounding corpus in existence -- all of them harvested by
scraping an a11y tree -- contains exactly zero examples of it. The *harvesting
method*, not merely the model, fails to transfer.

This module is the answer, and it has three parts:

1.  **Never orbit.** An orbit destroys the coordinate frame: the same pixel is a
    different entity five degrees later, and nothing downstream can be replayed.
    Bind instead to the app's NAMED views (:data:`NAMED_VIEWS`) plus zoom-to-fit.
    A named view makes the projection KNOWN, and therefore computable.

2.  **Compute the pixel, never guess it.** With a known orthographic camera the
    screen position of any model-space point is closed-form
    (:meth:`OrthoCamera.project`). We own the B-rep -- we built it -- so we know
    where every face, edge and vertex is in model space. No vision model is asked
    to "find the top face".

3.  **Adjudicate the click with the application's own picker.** A computed pixel
    is a *hypothesis*. The verdict comes from FreeCAD's own ``SoRayPickAction``
    (``View3DInventorPy.getObjectInfo``) -- the very hit-test a real mouse click
    runs -- and the pair is kept only if the app says the click selected the
    entity we intended. Occlusion, depth ordering and pick radius are therefore
    handled *exactly*, because the ground truth IS what will happen when the
    model clicks there. An unverified click is not a click.

The bridge (:class:`FreeCADGuiBridge`) speaks to the GUI's own Python
interpreter over a file channel. That is a deliberate, and dangerous, capability.

.. warning::

   **THE PYTHON CHANNEL IS THE ORACLE. IT MUST NEVER BE PART OF A TRAINING
   ENVIRONMENT'S ACTION SPACE.**

   FreeCAD's GUI hosts a live interpreter with full ``App``/``Gui`` access. If a
   *policy* can reach it -- via this bridge, the Python console panel, the macro
   editor, or any other scripting entry point -- then the optimal policy under any
   geometric reward is "paste a script, terminate": 100% reward, zero GUI
   learning, and a benchmark that measures nothing. The corpus generator in
   :mod:`harnesscad.eval.grounding.corpus` MAY use this channel, because it is
   the labeller and not the agent. A training or evaluation environment MUST NOT
   expose it, and must additionally remove the console/macro menus from the app
   itself. Do not wire this module into an agent's tool list. If you are reading
   this because you are about to: don't.

Everything here is import-safe: no FreeCAD, no GUI and no optional dependency is
touched at import time. The projection maths is pure stdlib and is unit-tested
without any application at all.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# Named views. NEVER ORBIT.
# --------------------------------------------------------------------------
#: The named orthographic views, each bound to (a) the ``View3DInventorPy``
#: method that sets it -- how the *oracle* sets the camera -- and (b) the
#: keystroke a *policy* would press to set it, which is the only navigation
#: action a grounded CAD agent ever needs.
#:
#: FreeCAD's stock bindings are the bare digits (View > Standard views): 0 is
#: axonometric/isometric, 1..6 are the six orthographic faces. There is no orbit
#: key, and that is the point -- these seven views span every entity that is
#: visible from *any* axis-aligned direction, and each one is a projection we can
#: write down.
NAMED_VIEWS: Dict[str, Tuple[str, str]] = {
    "isometric": ("viewAxometric", "0"),
    "front": ("viewFront", "1"),
    "top": ("viewTop", "2"),
    "right": ("viewRight", "3"),
    "rear": ("viewRear", "4"),
    "bottom": ("viewBottom", "5"),
    "left": ("viewLeft", "6"),
}

#: The default view order for a corpus sweep. Isometric first: it is the only one
#: that shows three faces at once and it is what a human actually works in.
VIEW_ORDER: Tuple[str, ...] = ("isometric", "front", "top", "right",
                               "rear", "bottom", "left")


class ViewportError(RuntimeError):
    """The viewport could not be driven, or a pick could not be adjudicated."""


class GuiUnavailable(ViewportError):
    """FreeCAD's GUI binary is not installed (tests SKIP; they never fail)."""


# --------------------------------------------------------------------------
# The projection. Pure maths, no application.
# --------------------------------------------------------------------------
def quat_rotate(q: Sequence[float], v: Sequence[float]) -> Tuple[float, float, float]:
    """Rotate ``v`` by the unit quaternion ``q = (x, y, z, w)``."""
    x, y, z, w = (float(c) for c in q)
    tx = 2.0 * (y * v[2] - z * v[1])
    ty = 2.0 * (z * v[0] - x * v[2])
    tz = 2.0 * (x * v[1] - y * v[0])
    return (v[0] + w * tx + (y * tz - z * ty),
            v[1] + w * ty + (z * tx - x * tz),
            v[2] + w * tz + (x * ty - y * tx))


def quat_inverse(q: Sequence[float]) -> Tuple[float, float, float, float]:
    """The conjugate of a unit quaternion ``(x, y, z, w)``."""
    x, y, z, w = (float(c) for c in q)
    return (-x, -y, -z, w)


def axis_angle_to_quat(axis: Sequence[float], angle: float) -> Tuple[float, float, float, float]:
    """Coin's ``.iv`` files store an orientation as AXIS + ANGLE, not a quaternion.

    ``SoOrthographicCamera.orientation`` reads back through pivy as a quaternion
    but serialises through ``View3DInventorPy.getCamera()`` as four numbers that
    are *axis x y z, angle in radians*. Reading the text as a quaternion is a
    silent, plausible-looking error -- the numbers even have roughly the right
    magnitudes -- so the two forms are converted explicitly and tested against
    each other.
    """
    ax, ay, az = (float(c) for c in axis)
    norm = math.sqrt(ax * ax + ay * ay + az * az)
    if norm <= 0.0:
        return (0.0, 0.0, 0.0, 1.0)
    ax, ay, az = ax / norm, ay / norm, az / norm
    half = float(angle) * 0.5
    s = math.sin(half)
    return (ax * s, ay * s, az * s, math.cos(half))


@dataclass(frozen=True)
class OrthoCamera:
    """A Coin3D ``SoOrthographicCamera`` plus the viewport it renders into.

    ``orientation`` is the quaternion ``(x, y, z, w)`` taking CAMERA space to
    WORLD space; the camera looks down its own -Z with +Y up. ``height`` is the
    full height of the view volume in model units.

    The one subtlety that decides whether the maths is right or merely close:
    Coin's default ``viewportMapping`` is ``ADJUST_CAMERA``, under which the
    camera's own ``aspectRatio`` field is IGNORED at render time and the *viewport's*
    aspect is used instead (``SoCamera::getViewVolume``: ``halfwidth = halfheight *
    vpaspect``, with a ``1/vpaspect`` rescale when the viewport is taller than it
    is wide). FreeCAD reports ``aspectRatio 1`` on a 1534x770 viewport; a
    projection that believed that field would be wrong by 2x in x -- and would
    still look roughly plausible, which is worse.
    """

    position: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float]
    height: float
    width_px: int
    height_px: int

    @property
    def viewport_aspect(self) -> float:
        return float(self.width_px) / float(self.height_px)

    def half_extents(self) -> Tuple[float, float]:
        """``(half_width, half_height)`` of the view volume, in MODEL units."""
        vp = self.viewport_aspect
        hh = self.height * 0.5
        hw = hh * vp
        if vp < 1.0:                      # Coin rescales a portrait viewport
            hh /= vp
            hw /= vp
        return (hw, hh)

    def to_camera_space(self, point: Sequence[float]) -> Tuple[float, float, float]:
        d = (float(point[0]) - self.position[0],
             float(point[1]) - self.position[1],
             float(point[2]) - self.position[2])
        return quat_rotate(quat_inverse(self.orientation), d)

    def project(self, point: Sequence[float]) -> Tuple[float, float, float]:
        """A model-space point as ``(px, py, depth)`` in VIEWPORT-LOCAL pixels.

        ``py`` counts UP from the bottom of the viewport -- OpenGL's convention,
        and the one ``View3DInventorPy.getObjectInfo`` takes. This was settled by
        probe, not by reading: projecting 32 entities of a real solid and
        adjudicating each with the app's own picker, y-up scored 10 exact hits and
        y-down scored 1 (chance). Getting it upside-down would corrupt every label
        in the corpus while still producing a corpus.

        ``depth`` is the camera-space z: it is NEGATIVE in front of the camera and
        more negative further away, so it orders candidates front-to-back.
        """
        cx, cy, cz = self.to_camera_space(point)
        hw, hh = self.half_extents()
        px = (cx / hw + 1.0) * 0.5 * self.width_px
        py = (cy / hh + 1.0) * 0.5 * self.height_px
        return (px, py, cz)

    def in_view(self, px: float, py: float, margin: int = 2) -> bool:
        return (margin <= px <= self.width_px - 1 - margin
                and margin <= py <= self.height_px - 1 - margin)

    def to_image_xy(self, px: float, py: float) -> Tuple[float, float]:
        """Viewport-local (y-up) -> IMAGE coordinates (y-down from the top).

        Screenshots are y-down. The picker is y-up. Carrying both conventions
        implicitly is how a grounding dataset ends up vertically mirrored.
        """
        return (px, float(self.height_px - 1) - py)

    def to_screen_xy(self, px: float, py: float,
                     rect: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """Viewport-local (y-up) -> absolute SCREEN pixels, for a real mouse.

        ``rect`` is the viewport's ``(left, top, width, height)`` on the virtual
        desktop, which the UIA tree hands us for free (never guess it, and never
        select the GL widget by ClassName -- FreeCAD carries a decoy
        ``QOpenGLWidget`` with a stale 100x30 rect outside the window).
        """
        left, top, _w, _h = rect
        ix, iy = self.to_image_xy(px, py)
        return (int(round(left + ix)), int(round(top + iy)))

    def to_dict(self) -> dict:
        return {"position": list(self.position),
                "orientation": list(self.orientation),
                "height": self.height,
                "viewport": [self.width_px, self.height_px]}

    @classmethod
    def from_dict(cls, d: dict) -> "OrthoCamera":
        vp = d.get("viewport") or [d.get("width_px", 1), d.get("height_px", 1)]
        return cls(position=tuple(float(c) for c in d["position"]),
                   orientation=tuple(float(c) for c in d["orientation"]),
                   height=float(d["height"]),
                   width_px=int(vp[0]), height_px=int(vp[1]))


def parse_camera(text: str, width_px: int, height_px: int) -> OrthoCamera:
    """Parse ``View3DInventorPy.getCamera()``'s Inventor-ASCII text.

    This is the pivy-free path, and it is the one the tests use: the whole
    projection can be checked against a recorded camera string with no FreeCAD in
    the room. A ``PerspectiveCamera`` is REFUSED rather than approximated -- exact
    grounding needs an exact projection, and the corpus generator always forces
    the camera to orthographic first.
    """
    if "PerspectiveCamera" in text:
        raise ViewportError(
            "perspective camera: exact viewport grounding requires an "
            "orthographic projection (call set_orthographic() first)")
    if "OrthographicCamera" not in text:
        raise ViewportError("not a camera description: %r" % text[:80])

    def numbers(field_name: str, count: int) -> List[float]:
        idx = text.find(field_name)
        if idx < 0:
            raise ViewportError("camera has no '%s' field" % field_name)
        tail = text[idx + len(field_name):]
        out: List[float] = []
        for tok in tail.replace("\n", " ").split():
            try:
                out.append(float(tok))
            except ValueError:
                break
            if len(out) == count:
                return out
        raise ViewportError("camera field '%s' has %d numbers, wanted %d"
                            % (field_name, len(out), count))

    pos = numbers("position", 3)
    ori = numbers("orientation", 4)
    height = numbers("height", 1)[0]
    quat = axis_angle_to_quat(ori[:3], ori[3])
    return OrthoCamera(position=(pos[0], pos[1], pos[2]), orientation=quat,
                       height=float(height),
                       width_px=int(width_px), height_px=int(height_px))


# --------------------------------------------------------------------------
# Candidate points: where on an entity should we actually click?
# --------------------------------------------------------------------------
#: Parametric fractions sampled along an edge / across a face, in the order they
#: are tried. The midpoint first, then a spread, so the first candidate that the
#: app adjudicates correctly is also the most canonical one.
EDGE_FRACTIONS: Tuple[float, ...] = (0.5, 0.35, 0.65, 0.2, 0.8)
FACE_FRACTIONS: Tuple[Tuple[float, float], ...] = (
    (0.5, 0.5), (0.35, 0.35), (0.65, 0.65), (0.35, 0.65), (0.65, 0.35),
    (0.5, 0.25), (0.5, 0.75), (0.25, 0.5), (0.75, 0.5),
)

#: The centre of mass is NOT a point on the entity for curved geometry -- the
#: centroid of a full cylindrical face lies on its axis, i.e. inside the
#: material, and the centroid of a circular edge is the circle's centre. A
#: corpus built on centroids therefore silently discards every hole and every
#: fillet, which are exactly the entities that matter. Candidates come from the
#: parametric surface/curve instead; the centroid is used only for a planar face,
#: where it is guaranteed to lie in the face (for a convex one) and is the point
#: a human would aim at.
CENTROID_FOR_PLANAR = True


# --------------------------------------------------------------------------
# The bridge to the running GUI. ORACLE ONLY -- see the module warning.
# --------------------------------------------------------------------------
EXECUTABLE_PATTERNS = (
    os.path.join(os.path.expanduser("~"), "AppData", "Local", "Programs",
                 "FreeCAD*", "bin", "freecad.exe"),
    r"C:\Program Files\FreeCAD*\bin\freecad.exe",
    "/usr/bin/freecad",
    "/usr/local/bin/freecad",
    "/Applications/FreeCAD.app/Contents/MacOS/FreeCAD",
)

#: The macro FreeCAD executes at startup. It installs a Qt timer that drains a
#: directory of command files and answers each with a JSON file -- a channel with
#: no socket, no port, no wall clock in a path, and nothing to leak if the process
#: dies. ``RESULT`` is the only thing a command returns.
BOOT_MACRO = r'''
import os, json, traceback
import FreeCAD, FreeCADGui
from PySide import QtCore

CHAN = os.environ["HARNESSCAD_VIEWPORT_CHANNEL"]


def _tick():
    try:
        names = [f for f in os.listdir(CHAN)
                 if f.startswith("cmd_") and f.endswith(".py")]
    except OSError:
        return
    for name in sorted(names):
        seq = name[4:-3]
        path = os.path.join(CHAN, name)
        try:
            with open(path) as fh:
                src = fh.read()
        except OSError:
            continue
        env = {"FreeCAD": FreeCAD, "App": FreeCAD, "Gui": FreeCADGui,
               "FreeCADGui": FreeCADGui, "RESULT": None}
        try:
            exec(compile(src, name, "exec"), env)
            out = {"ok": True, "result": env.get("RESULT")}
        except BaseException as exc:
            out = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc),
                   "traceback": traceback.format_exc()}
        try:
            os.remove(path)
        except OSError:
            pass
        tmp = os.path.join(CHAN, "tmp_" + seq)
        with open(tmp, "w") as fh:
            json.dump(out, fh, default=str)
        os.replace(tmp, os.path.join(CHAN, "out_" + seq + ".json"))


_HARNESSCAD_TIMER = QtCore.QTimer()
_HARNESSCAD_TIMER.timeout.connect(_tick)
_HARNESSCAD_TIMER.start(20)
with open(os.path.join(CHAN, "ready"), "w") as fh:
    fh.write("1")
'''


def find_gui_executable() -> Optional[str]:
    """The FreeCAD **GUI** binary (not ``freecadcmd``). ``HARNESSCAD_FREECAD_GUI`` wins."""
    override = os.environ.get("HARNESSCAD_FREECAD_GUI")
    if override and os.path.isfile(override):
        return override
    import glob
    for name in ("freecad", "FreeCAD"):
        found = shutil.which(name)
        if found:
            return found
    for pattern in EXECUTABLE_PATTERNS:
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[-1]
    return None


def gui_available() -> bool:
    return find_gui_executable() is not None


class FreeCADGuiBridge:
    """A live FreeCAD GUI, driven through its own Python interpreter.

    ORACLE ONLY. See the module warning: this is the labeller, never the agent.

    Safety, because this drives a real GUI on a real machine: the process is
    launched with ``-u`` pointed at a scratch user directory so it cannot touch
    the user's FreeCAD configuration, it opens no user file, it never saves, and
    :meth:`close` kills it. Nothing here writes outside ``workdir``.
    """

    def __init__(self, workdir: Optional[str] = None, timeout: float = 120.0,
                 poll: float = 0.01) -> None:
        exe = find_gui_executable()
        if exe is None:
            raise GuiUnavailable(
                "the FreeCAD GUI binary was not found; install FreeCAD or set "
                "HARNESSCAD_FREECAD_GUI")
        self.executable = exe
        self.timeout = float(timeout)
        self.poll = float(poll)
        self._owns_workdir = workdir is None
        self.workdir = workdir or tempfile.mkdtemp(prefix="harnesscad_vp_")
        self.channel = os.path.join(self.workdir, "channel")
        os.makedirs(self.channel, exist_ok=True)
        self._seq = 0
        self._proc: Optional[subprocess.Popen] = None

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> "FreeCADGuiBridge":
        boot = os.path.join(self.workdir, "harnesscad_boot.py")
        with open(boot, "w", encoding="utf-8") as fh:
            fh.write(BOOT_MACRO)
        env = dict(os.environ)
        env["HARNESSCAD_VIEWPORT_CHANNEL"] = self.channel
        userdir = os.path.join(self.workdir, "userdir")
        os.makedirs(userdir, exist_ok=True)
        self._proc = subprocess.Popen(
            [self.executable, "-u", userdir, boot],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        ready = os.path.join(self.channel, "ready")
        deadline = time.time() + self.timeout
        while not os.path.exists(ready):
            if self._proc.poll() is not None:
                raise ViewportError("FreeCAD exited (%s) before the channel opened"
                                    % self._proc.returncode)
            if time.time() > deadline:
                self.close()
                raise ViewportError("FreeCAD GUI did not open its channel in %.0fs"
                                    % self.timeout)
            time.sleep(0.05)
        return self

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover - a wedged GUI
                pass
        if self._owns_workdir:
            shutil.rmtree(self.workdir, ignore_errors=True)

    def __enter__(self) -> "FreeCADGuiBridge":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- the channel -------------------------------------------------------
    def call(self, source: str, timeout: Optional[float] = None) -> Any:
        """Run ``source`` in the GUI's interpreter; return whatever it set ``RESULT`` to."""
        if self._proc is None:
            raise ViewportError("bridge is not running (call start())")
        self._seq += 1
        seq = "%06d" % self._seq
        tmp = os.path.join(self.channel, "wtmp_" + seq)
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(source)
        os.replace(tmp, os.path.join(self.channel, "cmd_" + seq + ".py"))
        out = os.path.join(self.channel, "out_" + seq + ".json")
        deadline = time.time() + float(timeout or self.timeout)
        while not os.path.exists(out):
            if self._proc.poll() is not None:
                raise ViewportError("FreeCAD exited mid-command (%s)"
                                    % self._proc.returncode)
            if time.time() > deadline:
                raise ViewportError("GUI command timed out: %s" % source.strip()[:70])
            time.sleep(self.poll)
        # The GUI writes the answer with os.replace, which is atomic on NTFS --
        # but a virus scanner (or the indexer) can still hold a brand-new file
        # open for a few milliseconds, and Windows answers that with EACCES, not
        # with a retry. A bare open() here loses an entire corpus run to a race
        # that has nothing to do with us.
        payload = None
        for attempt in range(200):
            try:
                with open(out, encoding="utf-8") as fh:
                    payload = json.load(fh)
                break
            except (PermissionError, ValueError):
                if time.time() > deadline:
                    raise ViewportError("could not read the GUI's answer to: %s"
                                        % source.strip()[:70])
                time.sleep(0.02)
        if payload is None:
            raise ViewportError("could not read the GUI's answer to: %s"
                                % source.strip()[:70])
        try:
            os.remove(out)
        except OSError:
            pass
        if not payload.get("ok"):
            raise ViewportError(payload.get("traceback") or payload.get("error", "?"))
        return payload.get("result")


# --------------------------------------------------------------------------
# Entities and verified picks
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Entity:
    """One topological entity of the solid on screen, with its click candidates.

    ``name`` is FreeCAD's index name (``Face3``) -- which is precisely what a
    reference must never be *stored* as (see
    :mod:`harnesscad.domain.geometry.topology.topological_naming`), but is exactly
    what the app's picker answers with, so it is the right currency for a
    per-click label. ``description`` is the natural-language referent, computed
    from geometry.
    """

    name: str
    kind: str                     # face | edge | vertex
    description: str
    candidates: Tuple[Tuple[float, float, float], ...]
    surface: str = ""
    area: float = 0.0
    length: float = 0.0

    def to_dict(self) -> dict:
        return {"name": self.name, "kind": self.kind,
                "description": self.description, "surface": self.surface,
                "area": self.area, "length": self.length}


@dataclass(frozen=True)
class VerifiedPick:
    """A click the APPLICATION agreed with. The only kind we keep.

    ``verified`` is never set from the projection: it is set from the app's own
    hit-test at exactly the pixel we computed. ``selected`` is what the app said
    was there -- so a rejected pair still carries its evidence, and the reason a
    label was discarded is itself data (it is occlusion, or pick radius, and it is
    measurable).
    """

    entity: str
    kind: str
    description: str
    x: int                        # viewport-local, IMAGE coords (y down)
    y: int
    point: Tuple[float, float, float]
    depth: float
    selected: str = ""
    verified: bool = False
    reason: str = ""

    def to_dict(self) -> dict:
        return {"entity": self.entity, "kind": self.kind,
                "description": self.description, "x": self.x, "y": self.y,
                "point": list(self.point), "depth": self.depth,
                "selected": self.selected, "verified": self.verified,
                "reason": self.reason}


#: The GUI-side program that reads the solid's topology and, for every entity,
#: emits an ORDERED list of candidate 3D points that genuinely lie ON it.
_TOPOLOGY_SOURCE = r'''
import json

obj = App.ActiveDocument.getObject(%(obj)r)
shape = obj.Shape
EDGE_F = %(edge_fracs)r
FACE_F = %(face_fracs)r


def face_points(f):
    pts = []
    try:
        kind = type(f.Surface).__name__.upper()
    except Exception:
        kind = ""
    planar = kind.startswith("PLANE")
    if planar:
        c = f.CenterOfMass
        try:
            u, v = f.Surface.parameter(c)
            if f.isPartOfDomain(u, v):
                pts.append([c.x, c.y, c.z])
        except Exception:
            pts.append([c.x, c.y, c.z])
    u0, u1, v0, v1 = f.ParameterRange
    for (fu, fv) in FACE_F:
        u = u0 + (u1 - u0) * fu
        v = v0 + (v1 - v0) * fv
        try:
            if not f.isPartOfDomain(u, v):
                continue
            p = f.valueAt(u, v)
        except Exception:
            continue
        pts.append([p.x, p.y, p.z])
    return pts


def edge_points(e):
    pts = []
    a, b = e.FirstParameter, e.LastParameter
    for fr in EDGE_F:
        try:
            p = e.valueAt(a + (b - a) * fr)
        except Exception:
            continue
        pts.append([p.x, p.y, p.z])
    return pts


faces, edges, verts = [], [], []
for i, f in enumerate(shape.Faces):
    try:
        kind = type(f.Surface).__name__
    except Exception:
        kind = "Unknown"
    n = None
    try:
        nn = f.normalAt(0.0, 0.0)
        n = [nn.x, nn.y, nn.z]
    except Exception:
        pass
    c = f.CenterOfMass
    faces.append({"name": "Face%%d" %% (i + 1), "kind": "face", "surface": kind,
                  "area": f.Area, "normal": n,
                  "centroid": [c.x, c.y, c.z], "points": face_points(f)})
for i, e in enumerate(shape.Edges):
    try:
        kind = type(e.Curve).__name__
    except Exception:
        kind = "Unknown"
    c = e.CenterOfMass
    edges.append({"name": "Edge%%d" %% (i + 1), "kind": "edge", "surface": kind,
                  "length": e.Length, "centroid": [c.x, c.y, c.z],
                  "points": edge_points(e)})
for i, v in enumerate(shape.Vertexes):
    verts.append({"name": "Vertex%%d" %% (i + 1), "kind": "vertex", "surface": "",
                  "centroid": [v.X, v.Y, v.Z], "points": [[v.X, v.Y, v.Z]]})

bb = shape.BoundBox
RESULT = {"faces": faces, "edges": edges, "vertices": verts,
          "bbox": [bb.XMin, bb.YMin, bb.ZMin, bb.XMax, bb.YMax, bb.ZMax],
          "volume": shape.Volume, "area": shape.Area}
'''


class ViewportController:
    """Named views, analytic projection, and app-adjudicated picks.

    Construction does nothing to the GUI. Every method that changes the camera
    goes through a NAMED view; there is no orbit method, on purpose.
    """

    def __init__(self, bridge: FreeCADGuiBridge, obj_name: str = "Model") -> None:
        self.bridge = bridge
        self.obj_name = obj_name
        self._topology: Optional[dict] = None

    # -- camera ------------------------------------------------------------
    def set_orthographic(self) -> None:
        self.bridge.call(
            "v = Gui.activeDocument().activeView()\n"
            "v.setCameraType('Orthographic')\n"
            "RESULT = v.getCameraType()")

    def set_named_view(self, view: str, tries: int = 40, poll: float = 0.05) -> None:
        """Set one of :data:`NAMED_VIEWS` and zoom to fit. NEVER ORBITS.

        Then WAIT FOR THE CAMERA TO STOP MOVING. ``ViewFit`` is animated and Coin
        applies the new camera over several redraws, so a camera read too early
        returns one that is on its way to the view we asked for -- and every label
        harvested against it is projected through a camera that was never on
        screen. This is not hypothetical: with a fixed 0.15 s sleep the same part
        in the same view yielded 11 verified picks on one run and 0 on the next.
        A sleep is a guess. Polling to a fixed point is an assertion.
        """
        if view not in NAMED_VIEWS:
            raise ViewportError("unknown named view %r (have: %s)"
                                % (view, ", ".join(sorted(NAMED_VIEWS))))
        method, _key = NAMED_VIEWS[view]
        self.bridge.call(
            "v = Gui.activeDocument().activeView()\n"
            "v.setCameraType('Orthographic')\n"
            "v.%s()\n"
            "Gui.SendMsgToActiveView('ViewFit')\n"
            "Gui.updateGui()\n"
            "RESULT = 1" % method)
        previous = None
        stable = 0
        settled = False
        for _ in range(tries):
            current = self.bridge.call(
                "v = Gui.activeDocument().activeView()\n"
                "Gui.updateGui()\n"
                "RESULT = v.getCamera()")
            if current == previous:
                stable += 1
                if stable >= 2:          # two identical reads in a row: settled
                    settled = True
                    break
            else:
                stable = 0
            previous = current
            time.sleep(poll)
        if not settled:
            raise ViewportError("the camera never settled on the '%s' view" % view)
        self.await_picker(tries=tries, poll=poll)

    def await_picker(self, tries: int = 40, poll: float = 0.05) -> None:
        """Block until the app's ray-picker is LIVE, not merely until the camera is.

        ``getObjectInfo`` ray-picks the rendered scene graph. Before the viewport
        has actually drawn a frame it answers ``None`` for every pixel -- not "the
        background", but "I have nothing to pick against". Those Nones are
        indistinguishable, downstream, from a genuinely empty pixel, so an
        un-awaited picker does not crash: it quietly reports that most of the part
        is un-clickable and hands back a corpus that is a subset of the truth.
        Measured: the identical sweep yielded 495 verified pairs without this wait
        and 848 with it, and the 495 were a strict subset of the 848 with the same
        pixels. The corpus was not wrong. It was silently *incomplete*, which for a
        dataset that reports a discard rate is the more dangerous failure.

        The part is always zoom-fitted, so at least one pixel of a coarse probe
        grid MUST be on it. When one is, the picker is live.
        """
        probe = [(x / 6.0, y / 6.0) for x in range(1, 6) for y in range(1, 6)]
        size = self.viewport_size()
        pixels = [(fx * size[0], fy * size[1]) for fx, fy in probe]
        for _ in range(tries):
            if any(hit for hit in self.pick(pixels)):
                return
            self.bridge.call("Gui.updateGui()\nRESULT = 1")
            time.sleep(poll)
        raise ViewportError(
            "the viewport picker never went live: no pixel of a 5x5 probe grid "
            "hits the zoom-fitted solid")

    def camera(self) -> OrthoCamera:
        """The camera as the app actually has it, right now."""
        payload = self.bridge.call(
            "v = Gui.activeDocument().activeView()\n"
            "sz = v.getSize()\n"
            "RESULT = {'camera': v.getCamera(), 'w': int(sz[0]), 'h': int(sz[1])}")
        return parse_camera(payload["camera"], payload["w"], payload["h"])

    def viewport_size(self) -> Tuple[int, int]:
        sz = self.bridge.call(
            "sz = Gui.activeDocument().activeView().getSize()\n"
            "RESULT = [int(sz[0]), int(sz[1])]")
        return (int(sz[0]), int(sz[1]))

    # -- the solid ---------------------------------------------------------
    def topology(self, refresh: bool = False) -> dict:
        if self._topology is None or refresh:
            src = _TOPOLOGY_SOURCE % {"obj": self.obj_name,
                                      "edge_fracs": list(EDGE_FRACTIONS),
                                      "face_fracs": [list(p) for p in FACE_FRACTIONS]}
            self._topology = self.bridge.call(src)
        return self._topology

    def entities(self, kinds: Sequence[str] = ("face", "edge", "vertex")) -> List[Entity]:
        """Every entity of the solid, described in natural language, with candidates."""
        topo = self.topology()
        out: List[Entity] = []
        groups = {"face": topo["faces"], "edge": topo["edges"],
                  "vertex": topo["vertices"]}
        for kind in kinds:
            records = groups.get(kind) or []
            for rec in records:
                out.append(Entity(
                    name=rec["name"], kind=kind,
                    description=describe_entity(rec, topo),
                    candidates=tuple(tuple(float(c) for c in p)
                                     for p in rec.get("points") or []),
                    surface=rec.get("surface", ""),
                    area=float(rec.get("area", 0.0)),
                    length=float(rec.get("length", 0.0))))
        return out

    # -- the picker: the app adjudicates -----------------------------------
    def pick(self, points: Sequence[Tuple[float, float]]) -> List[Optional[str]]:
        """What the app says is under each VIEWPORT-LOCAL (y-up) pixel.

        This is ``SoRayPickAction`` -- the same hit-test, with the same pick
        radius, that a real mouse click runs. Batched: one channel round-trip for
        every candidate of every entity, which is what makes the corpus fast.
        """
        payload = json.dumps([[int(round(x)), int(round(y))] for x, y in points])
        return self.bridge.call(
            "import json\n"
            "v = Gui.activeDocument().activeView()\n"
            "pts = json.loads(%r)\n"
            "out = []\n"
            "for x, y in pts:\n"
            "    info = v.getObjectInfo((x, y))\n"
            "    out.append(None if info is None else info.get('Component'))\n"
            "RESULT = out" % payload)

    def adjudicate(self, entities: Sequence[Entity],
                   camera: Optional[OrthoCamera] = None) -> List[VerifiedPick]:
        """Project every candidate, ask the app, and keep only what it agrees with.

        For each entity the candidates are tried in order and the FIRST one the app
        adjudicates as that entity wins. An entity none of whose candidates the app
        agrees with is returned with ``verified=False`` and a reason -- it is
        occluded from this view, or the pick radius gave it to a neighbour, and
        DISCARDING it is the whole point. That discard rate is a measurement of how
        much of a CAD viewport is genuinely un-clickable from a given camera, and
        no other method can even estimate it.
        """
        cam = camera or self.camera()
        flat: List[Tuple[int, int, Tuple[float, float], Tuple[float, float, float], float]] = []
        for ei, ent in enumerate(entities):
            for ci, point in enumerate(ent.candidates):
                px, py, depth = cam.project(point)
                if not cam.in_view(px, py):
                    continue
                flat.append((ei, ci, (px, py), point, depth))
        verdicts = self.pick([f[2] for f in flat]) if flat else []

        best: Dict[int, VerifiedPick] = {}
        offscreen = {i for i in range(len(entities))}
        for (ei, _ci, (px, py), point, depth), got in zip(flat, verdicts):
            offscreen.discard(ei)
            if ei in best and best[ei].verified:
                continue
            ent = entities[ei]
            ix, iy = cam.to_image_xy(px, py)
            hit = (got == ent.name)
            pick = VerifiedPick(
                entity=ent.name, kind=ent.kind, description=ent.description,
                x=int(round(ix)), y=int(round(iy)), point=point, depth=depth,
                selected=got or "", verified=hit,
                reason="" if hit else ("nothing under the pixel" if not got
                                       else "occluded by %s" % got))
            if hit or ei not in best:
                best[ei] = pick
        out: List[VerifiedPick] = []
        for ei, ent in enumerate(entities):
            if ei in best:
                out.append(best[ei])
            else:
                out.append(VerifiedPick(
                    entity=ent.name, kind=ent.kind, description=ent.description,
                    x=-1, y=-1, point=ent.candidates[0] if ent.candidates else (0, 0, 0),
                    depth=0.0, selected="", verified=False,
                    reason="every candidate projects outside the viewport"
                           if ei in offscreen else "no candidate point on the entity"))
        return out

    # -- the real mouse: does a synthesised click actually select it? -------
    def clear_selection(self) -> None:
        self.bridge.call("Gui.Selection.clearSelection()\nRESULT = 1")

    def selection(self) -> List[str]:
        """``Gui.Selection`` read back as ``['Model.Face3', ...]``."""
        return self.bridge.call(
            "out = []\n"
            "for s in Gui.Selection.getSelectionEx():\n"
            "    for sub in (s.SubElementNames or ('',)):\n"
            "        out.append('%s.%s' % (s.ObjectName, sub))\n"
            "RESULT = out")

    def viewport_rect(self) -> Tuple[int, int, int, int]:
        """The viewport's ``(left, top, width, height)`` on the virtual desktop.

        Read from Qt inside the app (``mapToGlobal``), which is authoritative and
        skips the decoy ``QOpenGLWidget`` the UIA tree also carries. Needed only
        for a REAL mouse click; the corpus itself never leaves viewport-local
        coordinates.
        """
        rect = self.bridge.call(
            "from PySide import QtGui, QtCore\n"
            "mw = Gui.getMainWindow()\n"
            "out = None\n"
            "for w in mw.findChildren(QtGui.QWidget):\n"
            "    if w.metaObject().className() == 'Gui::View3DInventorViewer':\n"
            "        p = w.mapToGlobal(QtCore.QPoint(0, 0))\n"
            "        out = [p.x(), p.y(), w.width(), w.height()]\n"
            "RESULT = out")
        if not rect:
            raise ViewportError("no Gui::View3DInventorViewer widget found")
        return (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))

    def focus_window(self, settle: float = 0.5) -> None:
        """Bring the GUI forward ONCE, and let the window manager finish.

        Raising the window immediately before every click does not work: the
        activation is still in flight when the click arrives and Windows delivers
        it to nothing. Measured, on identical pixels: 0/4 selections when the
        window is raised per-click, 3/4 when it is raised once and then left
        alone. Focus is a precondition, not part of the action.
        """
        self.bridge.call("mw = Gui.getMainWindow()\n"
                         "mw.activateWindow()\n"
                         "mw.raise_()\n"
                         "RESULT = 1")
        time.sleep(settle)

    def mouse_click(self, pick: VerifiedPick, camera: OrthoCamera,
                    rect: Optional[Tuple[int, int, int, int]] = None,
                    settle: float = 0.35) -> List[str]:
        """Synthesise a REAL left click at the pick's pixel; return the selection.

        This is the honesty check on the whole scheme: :meth:`adjudicate` predicts
        what a click will select using the app's ray-picker, but a *policy* will
        move a physical mouse. The two agreeing is a claim, and it is one we can
        test rather than assert. Uses ``SendInput`` (never ``PostMessage``, whose
        keys do not update the async key state and whose success return proves
        nothing).
        """
        import ctypes

        rect = rect or self.viewport_rect()
        # The pick carries IMAGE coords (y down); the viewport rect is y-down too.
        sx, sy = rect[0] + pick.x, rect[1] + pick.y
        self.clear_selection()
        user32 = ctypes.windll.user32          # noqa: F821 - Windows only
        user32.SetCursorPos(int(sx), int(sy))
        time.sleep(0.05)
        user32.mouse_event(0x0002, 0, 0, 0, 0)   # LEFTDOWN
        time.sleep(0.02)
        user32.mouse_event(0x0004, 0, 0, 0, 0)   # LEFTUP
        time.sleep(settle)
        return self.selection()


# --------------------------------------------------------------------------
# Natural-language referents, from geometry. No LLM, no human.
# --------------------------------------------------------------------------
_SURFACE_WORD = {
    "PLANE": "planar", "CYLINDER": "cylindrical", "CONE": "conical",
    "SPHERE": "spherical", "TORUS": "toroidal", "BSPLINESURFACE": "freeform",
    "SURFACEOFREVOLUTION": "revolved",
}
_CURVE_WORD = {"LINE": "straight", "CIRCLE": "circular", "ELLIPSE": "elliptical",
               "BSPLINECURVE": "freeform"}

#: Axis-aligned normals, as a human names them. A face whose normal is within
#: ``_AXIS_TOL`` of one of these gets the word; anything else gets "slanted".
_AXIS_NAMES = (((0, 0, 1), "top"), ((0, 0, -1), "bottom"),
               ((1, 0, 0), "right"), ((-1, 0, 0), "left"),
               ((0, 1, 0), "rear"), ((0, -1, 0), "front"))
_AXIS_TOL = 0.99


def _axis_word(normal: Optional[Sequence[float]]) -> str:
    if not normal:
        return ""
    n = [float(c) for c in normal]
    mag = math.sqrt(sum(c * c for c in n)) or 1.0
    n = [c / mag for c in n]
    for axis, word in _AXIS_NAMES:
        if sum(a * b for a, b in zip(axis, n)) >= _AXIS_TOL:
            return word
    return "slanted"


def describe_entity(rec: dict, topo: dict) -> str:
    """A deterministic natural-language referent for one entity.

    Computed from the B-rep, not from a caption model: the surface kind, the
    face's axis if it has one, and its position in the part's bounding box. This
    is the ``text`` half of a grounding pair, and it is exact by construction --
    which is precisely what a scraped a11y tree gives a web-grounding corpus and
    what no one has ever had for a 3D viewport.
    """
    kind = rec.get("kind")
    surf = str(rec.get("surface", "")).upper()
    c = rec.get("centroid") or [0.0, 0.0, 0.0]
    bb = topo.get("bbox") or [0, 0, 0, 1, 1, 1]
    span = [max(bb[3] - bb[0], 1e-9), max(bb[4] - bb[1], 1e-9), max(bb[5] - bb[2], 1e-9)]
    rel = [(c[i] - bb[i]) / span[i] for i in range(3)]

    def where() -> str:
        parts = []
        if rel[2] > 0.66:
            parts.append("upper")
        elif rel[2] < 0.34:
            parts.append("lower")
        if rel[1] > 0.66:
            parts.append("rear")
        elif rel[1] < 0.34:
            parts.append("front")
        if rel[0] > 0.66:
            parts.append("right")
        elif rel[0] < 0.34:
            parts.append("left")
        return "-".join(parts) if parts else "central"

    if kind == "face":
        word = ""
        for key, val in _SURFACE_WORD.items():
            if surf.upper().startswith(key) or key in surf.upper():
                word = val
                break
        axis = _axis_word(rec.get("normal"))
        if word == "planar" and axis and axis != "slanted":
            return "the %s face of the part" % axis
        if word == "cylindrical":
            return "the cylindrical face at the %s of the part" % where()
        return "the %s face at the %s of the part" % (word or "curved", where())
    if kind == "edge":
        word = ""
        for key, val in _CURVE_WORD.items():
            if key in surf.upper():
                word = val
                break
        return "the %s edge at the %s of the part" % (word or "curved", where())
    return "the vertex at the %s of the part" % where()
