"""The self-labelling CAD-viewport grounding corpus.

Every GUI grounding model in existence -- SeeClick, ShowUI, OS-Atlas, UI-TARS --
was trained on data harvested by scraping an accessibility tree: a
``<button aria-label="Save">`` has a name and a box, for free, in the millions.
A 3D CAD viewport has no accessibility tree. It is ONE opaque node. ``Face7`` has
neither a name nor a box in any DOM sense. So the *harvesting method*, not merely
the model, fails to transfer, and no CAD viewport grounding dataset exists --
because you cannot scrape one and a human would have to label it.

We do not have to label it.

  1. We generated the geometry, so we own the exact B-rep.
  2. We set a NAMED orthographic view, so we know the camera exactly.
  3. We project every face / edge / vertex to screen space ourselves, analytically
     (:class:`~harnesscad.io.cua.viewport.OrthoCamera`).
  4. **We adjudicate every label with the application's own picker.** The click is
     put to FreeCAD's ``SoRayPickAction`` -- the same hit-test a real mouse runs --
     and the pair is KEPT ONLY IF THE APP SELECTED WHAT WE INTENDED.

Step 4 is the whole trick. It handles occlusion, depth ordering and pick radius
automatically, because the ground truth IS what will actually happen when a model
clicks there. The discards are not failures: the discard rate *measures* how much
of a CAD viewport is genuinely un-clickable from a given view, and it is a
number nobody has ever been able to compute.

What this corpus proves, and what it does not
---------------------------------------------
**It proves**: that a click at pixel (x, y) of *this* screenshot selects *that*
entity, in this application, at this camera. That is per-click, exact, and
adjudicated by the thing that will be adjudicating the model's clicks too.

**It does not prove** that the part is the part the brief asked for. The
harness's measured gate is many-to-one -- volume, bbox and genus do not pin down
a part, and two different solids can share all three. For GROUNDING that matters
much less than it does for the end-to-end task, because the label here is
per-click and is not derived from the measurement vector at all; but it is not
zero, and it must not be overclaimed. A corpus of verified clicks on an
*under-determined* part is still a corpus of verified clicks; it is simply not
evidence that the CAD task was solved.

**It does not prove** transfer to SolidWorks, Fusion or NX. The projection maths
is universal; the picker, the pick radius and the entity naming are FreeCAD's.

Cold start
----------
This module is a COMPILER, not a filter. It does not sample an agent policy and
keep the successes -- at cold start a general VLM's success rate in a CAD viewport
is a few percent, and you cannot filter what you never sample. Every pair here is
correct by construction, p(success) = 1.0, because the geometry and the camera are
inputs and not predictions. That is the difference between rejection sampling
(which is unaffordable) and compilation (which is a loop).

.. warning::

   The generator drives FreeCAD's GUI through its own Python interpreter. That
   channel is THE ORACLE and MUST NEVER BE IN A TRAINING ENVIRONMENT'S ACTION
   SPACE -- see the warning in :mod:`harnesscad.io.cua.viewport`. If a policy can
   reach a Python console, the optimal policy is "paste a script, terminate":
   full reward, zero GUI learning.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from harnesscad.io.cua import viewport as vp

#: The views a corpus sweep visits. Isometric shows three faces at once; the six
#: orthographic views between them expose every axis-aligned face of a convex
#: part. There is NO orbit: an orbit destroys the coordinate frame.
DEFAULT_VIEWS: Tuple[str, ...] = ("isometric", "front", "top", "right")

#: The entity kinds harvested. Vertices are included because they are the hardest
#: (they sit on silhouettes, and the pick radius usually gives them to an edge) --
#: their discard rate is the interesting one.
DEFAULT_KINDS: Tuple[str, ...] = ("face", "edge", "vertex")


@dataclass(frozen=True)
class GroundingPair:
    """One ``(screenshot, description) -> (x, y)`` example. Verified, or discarded.

    ``x``/``y`` are IMAGE coordinates (origin top-left) into ``screenshot``, which
    is the viewport render at exactly the resolution the camera was read at -- so
    there is no downscale, no letterbox and no DPI in the label. (Both of the
    vision CUA repos surveyed get this wrong: one resizes 16:9 into 16:10 with
    ``fit: fill``, the other applies a single geometric-mean scale factor to both
    axes after a non-uniform resize. An anamorphic screenshot makes circles into
    ellipses, which for a CAD grounder is indefensible.)
    """

    sample: str
    view: str
    screenshot: str
    entity: str
    kind: str
    description: str
    x: int
    y: int
    point: Tuple[float, float, float]
    verified: bool
    selected: str = ""
    reason: str = ""
    brief: str = ""
    width: int = 0                # the screenshot's size, carried WITH the label:
    height: int = 0               # a coordinate without its frame is not a label

    def to_dict(self) -> dict:
        return {"sample": self.sample, "view": self.view,
                "screenshot": self.screenshot, "entity": self.entity,
                "kind": self.kind, "description": self.description,
                "x": self.x, "y": self.y, "point": list(self.point),
                "verified": self.verified, "selected": self.selected,
                "reason": self.reason, "brief": self.brief,
                "width": self.width, "height": self.height}


@dataclass
class CorpusStats:
    """The numbers that make the corpus honest.

    ``discard_rate`` is not a defect metric. It is a *measurement*: the fraction
    of a part's topology that is genuinely un-clickable from a given camera, which
    is a property of CAD viewports that nobody has ever quantified because nobody
    else can adjudicate a click without a human.
    """

    samples: int = 0
    views: int = 0
    candidates: int = 0
    verified: int = 0
    discarded: int = 0
    elapsed: float = 0.0
    by_kind: Dict[str, List[int]] = field(default_factory=dict)   # kind -> [ok, total]
    by_view: Dict[str, List[int]] = field(default_factory=dict)
    by_reason: Dict[str, int] = field(default_factory=dict)

    def record(self, pair: GroundingPair) -> None:
        self.candidates += 1
        for table, key in ((self.by_kind, pair.kind), (self.by_view, pair.view)):
            slot = table.setdefault(key, [0, 0])
            slot[1] += 1
            if pair.verified:
                slot[0] += 1
        if pair.verified:
            self.verified += 1
        else:
            self.discarded += 1
            reason = pair.reason.split(" by ")[0] if pair.reason else "unknown"
            self.by_reason[reason] = self.by_reason.get(reason, 0) + 1

    @property
    def discard_rate(self) -> float:
        return 0.0 if not self.candidates else self.discarded / float(self.candidates)

    @property
    def pairs_per_minute(self) -> float:
        if self.elapsed <= 0.0:
            return 0.0
        return self.verified * 60.0 / self.elapsed

    def to_dict(self) -> dict:
        return {"samples": self.samples, "views": self.views,
                "candidates": self.candidates, "verified": self.verified,
                "discarded": self.discarded,
                "discard_rate": round(self.discard_rate, 4),
                "elapsed_s": round(self.elapsed, 2),
                "pairs_per_minute": round(self.pairs_per_minute, 1),
                "by_kind": {k: {"verified": v[0], "total": v[1]}
                            for k, v in sorted(self.by_kind.items())},
                "by_view": {k: {"verified": v[0], "total": v[1]}
                            for k, v in sorted(self.by_view.items())},
                "by_reason": dict(sorted(self.by_reason.items(),
                                         key=lambda kv: -kv[1]))}


# --------------------------------------------------------------------------
# The parts. Seeded CISP op streams, over the existing ParametricSampler.
# --------------------------------------------------------------------------
#
# The part family is chosen for its TOPOLOGY, not its realism: grounding is about
# telling entities apart, so what matters is having faces that are hard to name
# (a bore's cylindrical wall, a fillet's blend, a chamfer's bevel), faces that
# occlude one another, and edges that a pick radius will fight over. A corpus of
# nothing but boxes would report a wonderful discard rate and teach a model
# nothing.
#
# These are built from CISP ops directly rather than from
# ``data.datagen.generators``' templates: those templates carry ``Constrain`` ops
# whose arity the backend now (correctly) rejects, and a sketch constraint is
# invisible in a rendered viewport anyway. The seeded ``ParametricSampler`` -- the
# machinery that makes a synthetic dataset reproducible -- is reused as-is.
def _families():
    from harnesscad.core.cisp.ops import (
        AddCircle, AddRectangle, Boolean, Chamfer, Extrude, Fillet, NewSketch,
        Shell,
    )

    def plate(rng):
        w, h, t = rng.dim(40, 120), rng.dim(30, 90), rng.dim(4, 16)
        ops = [NewSketch(plane="XY"), AddRectangle(sketch="sk1", x=0, y=0, w=w, h=h),
               Extrude(sketch="sk1", distance=t)]
        return ("a flat rectangular plate %g x %g x %g mm" % (w, h, t), ops)

    def bored_plate(rng):
        w, h, t = rng.dim(50, 120), rng.dim(40, 90), rng.dim(6, 18)
        r = rng.dim(4, 12)
        ops = [NewSketch(plane="XY"), AddRectangle(sketch="sk1", x=0, y=0, w=w, h=h),
               Extrude(sketch="sk1", distance=t),
               NewSketch(plane="XY"),
               AddCircle(sketch="sk2", cx=w / 2.0, cy=h / 2.0, r=r),
               Extrude(sketch="sk2", distance=t),
               Boolean(kind="cut", target="f1", tool="f2")]
        return ("a %g x %g x %g mm plate with a central %g mm bore"
                % (w, h, t, 2 * r), ops)

    def four_hole_plate(rng):
        w, h, t = rng.dim(70, 140), rng.dim(50, 100), rng.dim(6, 14)
        r = rng.dim(3, 7)
        m = rng.dim(10, 18)
        ops = [NewSketch(plane="XY"), AddRectangle(sketch="sk1", x=0, y=0, w=w, h=h),
               Extrude(sketch="sk1", distance=t)]
        body, feat, sk = "f1", 1, 1
        for cx, cy in ((m, m), (w - m, m), (m, h - m), (w - m, h - m)):
            sk += 1
            ops.append(NewSketch(plane="XY"))
            ops.append(AddCircle(sketch="sk%d" % sk, cx=cx, cy=cy, r=r))
            ops.append(Extrude(sketch="sk%d" % sk, distance=t))
            feat += 1
            ops.append(Boolean(kind="cut", target=body, tool="f%d" % feat))
            feat += 1
            body = "f%d" % feat
        return ("a %g x %g x %g mm plate with four %g mm mounting holes"
                % (w, h, t, 2 * r), ops)

    # NOTE: the blends take ``edges=()`` -- the UNIFORM blend over every edge --
    # rather than a selector. The FreeCAD backend is composed over the F-rep
    # backend, which owns the op log and refuses a selector outright ("an SDF has
    # no topological edges for a selector to name"). That refusal is right, and it
    # is not this module's to argue with. A uniform blend gives the corpus exactly
    # what it needs anyway -- 12 real fillet faces and 12 real chamfer faces, whose
    # narrow, curved, mutually-occluding geometry is the hardest thing in the
    # viewport to ground on.
    def filleted_block(rng):
        w, h, t = rng.dim(40, 90), rng.dim(40, 90), rng.dim(15, 35)
        r = rng.dim(3, 8)
        ops = [NewSketch(plane="XY"), AddRectangle(sketch="sk1", x=0, y=0, w=w, h=h),
               Extrude(sketch="sk1", distance=t),
               Fillet(edges=(), radius=r)]
        return ("a %g x %g x %g mm block with every edge rounded to a %g mm radius"
                % (w, h, t, r), ops)

    def chamfered_block(rng):
        w, h, t = rng.dim(40, 90), rng.dim(40, 90), rng.dim(15, 35)
        d = rng.dim(2, 6)
        ops = [NewSketch(plane="XY"), AddRectangle(sketch="sk1", x=0, y=0, w=w, h=h),
               Extrude(sketch="sk1", distance=t),
               Chamfer(edges=(), distance=d)]
        return ("a %g x %g x %g mm block with a %g mm chamfer on every edge"
                % (w, h, t, d), ops)

    def shelled_box(rng):
        w, h, t = rng.dim(50, 100), rng.dim(40, 80), rng.dim(20, 40)
        wall = rng.dim(2, 5)
        ops = [NewSketch(plane="XY"), AddRectangle(sketch="sk1", x=0, y=0, w=w, h=h),
               Extrude(sketch="sk1", distance=t),
               Shell(faces=("top",), thickness=wall)]
        return ("an open-topped %g x %g x %g mm box with a %g mm wall"
                % (w, h, t, wall), ops)

    def boss(rng):
        w, h, t = rng.dim(50, 100), rng.dim(50, 100), rng.dim(8, 16)
        r, bh = rng.dim(10, 20), rng.dim(15, 30)
        br = rng.dim(3, 6)
        ops = [NewSketch(plane="XY"), AddRectangle(sketch="sk1", x=0, y=0, w=w, h=h),
               Extrude(sketch="sk1", distance=t),
               NewSketch(plane="XY"),
               AddCircle(sketch="sk2", cx=w / 2.0, cy=h / 2.0, r=r),
               Extrude(sketch="sk2", distance=t + bh),
               Boolean(kind="union", target="f1", tool="f2"),
               NewSketch(plane="XY"),
               AddCircle(sketch="sk3", cx=w / 2.0, cy=h / 2.0, r=br),
               Extrude(sketch="sk3", distance=t + bh),
               Boolean(kind="cut", target="f3", tool="f4")]
        return ("a %g x %g mm base plate with a %g mm cylindrical boss bored "
                "through by a %g mm hole" % (w, h, 2 * r, 2 * br), ops)

    return [("plate", plate), ("bored_plate", bored_plate),
            ("four_hole_plate", four_hole_plate),
            ("filleted_block", filleted_block),
            ("chamfered_block", chamfered_block),
            ("shelled_box", shelled_box), ("boss", boss)]


def sample_parts(count: int, seed: int = 0) -> List[Tuple[str, str, list, dict]]:
    """``(sample_id, brief, ops, params)`` -- seeded, deterministic, no wall clock.

    Same seed, same parts, byte for byte: a corpus that cannot be regenerated is
    not a corpus.
    """
    from harnesscad.data.datagen.generators import ParametricSampler

    families = _families()
    out: List[Tuple[str, str, list, dict]] = []
    for i in range(count):
        name, fn = families[i % len(families)]
        rng = ParametricSampler(seed * 1000 + i)
        brief, ops = fn(rng)
        out.append(("s%04d_%s" % (i, name), brief, list(ops),
                    {"generator": name, "seed": seed * 1000 + i}))
    return out


def build_step(ops: Sequence[Any], workdir: str, name: str) -> str:
    """Build a CISP op stream with the SCRIPTED FreeCAD backend; return a STEP path.

    The scripted backend is the one that already agrees with ANALYTIC on all 20
    CISP ops to 4.5e-16, and it is content-addressed and cached -- so the solid the
    GUI shows is exactly the solid the harness measured, and the corpus inherits
    that guarantee rather than re-deriving it. The GUI then simply *imports* the
    B-rep: the viewport is a display of a solid we already verified, which is the
    whole reason the labels can be exact.

    ``io/backends`` is not touched: this only calls its public surface.
    """
    from harnesscad.io.backends.freecad import FreeCADBackend

    backend = FreeCADBackend()
    backend.reset()
    for op in ops:
        result = backend.apply(op)
        if not result.ok:
            raise vp.ViewportError(
                "op %s rejected: %s" % (getattr(type(op), "OP", "?"),
                                        "; ".join(d.message for d in result.diagnostics)))
    diagnostics = backend.regenerate()
    errors = [d for d in diagnostics if getattr(d.severity, "name", "") == "ERROR"]
    if errors:
        raise vp.ViewportError("model did not build: %s"
                               % "; ".join(d.message for d in errors))
    text = backend.export("step")
    path = os.path.join(workdir, "%s.step" % name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


#: Load a STEP into the running GUI as a single ``Part::Feature`` named ``Model``.
#: A fresh document per sample: no user file is ever opened, nothing is ever
#: saved, and the previous sample cannot leak into the next one.
_LOAD_SOURCE = r'''
import Part
for d in list(App.listDocuments()):
    App.closeDocument(d)
doc = App.newDocument("grounding")
shape = Part.Shape()
shape.read(%(path)r)
solids = shape.Solids
obj = doc.addObject("Part::Feature", "Model")
obj.Shape = solids[0] if len(solids) == 1 else shape
doc.recompute()
Gui.activeDocument().activeView().setCameraType("Orthographic")
Gui.SendMsgToActiveView("ViewFit")
Gui.updateGui()
RESULT = {"faces": len(obj.Shape.Faces), "edges": len(obj.Shape.Edges),
          "volume": obj.Shape.Volume}
'''


class CorpusGenerator:
    """Drives one FreeCAD GUI and emits verified grounding pairs.

    Usage is a context manager, because the GUI must die even if we do::

        with CorpusGenerator(outdir) as gen:
            pairs, stats = gen.run(count=8, seed=0)
    """

    def __init__(self, outdir: str, views: Sequence[str] = DEFAULT_VIEWS,
                 kinds: Sequence[str] = DEFAULT_KINDS,
                 timeout: float = 180.0) -> None:
        self.outdir = outdir
        self.views = tuple(views)
        self.kinds = tuple(kinds)
        self.timeout = timeout
        os.makedirs(os.path.join(self.outdir, "images"), exist_ok=True)
        self.bridge: Optional[vp.FreeCADGuiBridge] = None
        self.controller: Optional[vp.ViewportController] = None
        self._steps: Dict[str, str] = {}      # sample -> the STEP it was built from

    def __enter__(self) -> "CorpusGenerator":
        self.bridge = vp.FreeCADGuiBridge(timeout=self.timeout).start()
        self.controller = vp.ViewportController(self.bridge, obj_name="Model")
        return self

    def __exit__(self, *exc: object) -> None:
        if self.bridge is not None:
            self.bridge.close()
        self.bridge = None
        self.controller = None

    # -- one part, one view -------------------------------------------------
    def screenshot(self, path: str, size: Tuple[int, int]) -> str:
        """Render the viewport to PNG at EXACTLY the size the camera was read at.

        ``View3DInventorPy.saveImage`` renders through the same camera and the same
        scene graph, so the image and the projection cannot disagree about the
        frame -- which is the failure mode that silently corrupts a grounding
        dataset (and which no amount of downstream checking can detect).
        """
        assert self.bridge is not None
        self.bridge.call(
            "v = Gui.activeDocument().activeView()\n"
            "v.saveImage(%r, %d, %d, 'Current')\n"
            "RESULT = 1" % (path, int(size[0]), int(size[1])))
        return path

    def harvest(self, sample: str, brief: str, view: str) -> List[GroundingPair]:
        """Set the view, shoot it, project everything, let the app adjudicate."""
        ctl = self.controller
        assert ctl is not None
        ctl.set_named_view(view)
        camera = ctl.camera()
        image = os.path.join(self.outdir, "images", "%s_%s.png" % (sample, view))
        self.screenshot(image, (camera.width_px, camera.height_px))
        entities = ctl.entities(self.kinds)
        picks = ctl.adjudicate(entities, camera)
        rel = os.path.relpath(image, self.outdir).replace("\\", "/")
        return [GroundingPair(sample=sample, view=view, screenshot=rel,
                              entity=p.entity, kind=p.kind,
                              description=p.description, x=p.x, y=p.y,
                              point=p.point, verified=p.verified,
                              selected=p.selected, reason=p.reason, brief=brief,
                              width=camera.width_px, height=camera.height_px)
                for p in picks]

    # -- the sweep ----------------------------------------------------------
    def load(self, step: str) -> None:
        """Show one solid. A fresh document: the previous part cannot leak in."""
        assert self.bridge is not None and self.controller is not None
        self.bridge.call(_LOAD_SOURCE % {"path": step}, timeout=self.timeout)
        self.controller._topology = None                   # new solid, new topology

    def run(self, count: int = 4, seed: int = 0,
            progress: bool = False) -> Tuple[List[GroundingPair], CorpusStats]:
        assert self.controller is not None and self.bridge is not None
        stats = CorpusStats()
        pairs: List[GroundingPair] = []
        steps = os.path.join(self.outdir, "steps")
        os.makedirs(steps, exist_ok=True)
        t0 = time.perf_counter()
        for sample, brief, ops, _params in sample_parts(count, seed=seed):
            try:
                step = build_step(ops, steps, sample)
            except Exception as exc:                       # noqa: BLE001
                if progress:
                    print("  skip %s: %s" % (sample, exc))
                continue
            self.load(step)
            self._steps[sample] = step
            stats.samples += 1
            for view in self.views:
                got = self.harvest(sample, brief, view)
                stats.views += 1
                for pair in got:
                    stats.record(pair)
                pairs.extend(got)
                if progress:
                    ok = sum(1 for p in got if p.verified)
                    print("  %s/%-9s %3d/%3d verified" % (sample, view, ok, len(got)))
        stats.elapsed = time.perf_counter() - t0
        return pairs, stats

    # -- the honesty check --------------------------------------------------
    def crosscheck_mouse(self, pairs: Sequence[GroundingPair],
                         limit: int = 12) -> Dict[str, Any]:
        """Does a REAL synthesised mouse click select what the ray-picker predicted?

        :meth:`ViewportController.adjudicate` uses the app's ``SoRayPickAction``.
        A trained policy will move a physical mouse instead. That the two agree is
        a CLAIM, and this measures it rather than asserting it: for a sample of
        verified pairs, re-set the view, click the pixel with ``SendInput``, and
        read ``Gui.Selection`` back.

        Only ever runs on our own scratch document, and clears the selection after.
        """
        ctl = self.controller
        assert ctl is not None
        # A pair is only meaningful against the document it was harvested from.
        # Clicking sample 0's pixels while sample 13 is on screen measures
        # nothing at all -- and it looks exactly like a catastrophic failure of
        # the projection, which is how a real result gets thrown away by mistake.
        # Sort by (sample, view) and RELOAD, so each click lands on its own part.
        verified = sorted((p for p in pairs if p.verified),
                          key=lambda p: (p.sample, p.view))[:limit]
        if not verified:
            return {"checked": 0, "agree": 0, "agreement": 0.0, "mismatches": []}
        rect = ctl.viewport_rect()
        ctl.focus_window()          # ONCE. Never per-click: see focus_window().
        agree = 0
        mismatches: List[dict] = []
        current: Tuple[str, str] = ("", "")
        for pair in verified:
            if (pair.sample, pair.view) != current:
                if pair.sample != current[0]:
                    self.load(self._steps[pair.sample])
                ctl.set_named_view(pair.view)
                current = (pair.sample, pair.view)
            camera = ctl.camera()
            pick = vp.VerifiedPick(entity=pair.entity, kind=pair.kind,
                                   description=pair.description, x=pair.x, y=pair.y,
                                   point=pair.point, depth=0.0, verified=True)
            got = ctl.mouse_click(pick, camera, rect)
            names = [g.split(".", 1)[1] for g in got if "." in g]
            if pair.entity in names:
                agree += 1
            else:
                mismatches.append({"entity": pair.entity, "view": pair.view,
                                   "xy": [pair.x, pair.y], "selected": names})
        ctl.clear_selection()
        return {"checked": len(verified), "agree": agree,
                "agreement": round(agree / float(len(verified)), 4),
                "mismatches": mismatches[:6]}


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------
def write_corpus(outdir: str, pairs: Sequence[GroundingPair], stats: CorpusStats,
                 crosscheck: Optional[dict] = None) -> str:
    """Write ``pairs.jsonl`` (VERIFIED only) + ``discards.jsonl`` + ``stats.json``.

    Discards are kept, not thrown away. They are the negative half of the
    measurement and they are what the discard rate is computed from -- a corpus
    that silently drops them cannot report its own coverage.
    """
    os.makedirs(outdir, exist_ok=True)
    keep = os.path.join(outdir, "pairs.jsonl")
    drop = os.path.join(outdir, "discards.jsonl")
    with open(keep, "w", encoding="utf-8") as fh:
        for pair in pairs:
            if pair.verified:
                fh.write(json.dumps(pair.to_dict(), sort_keys=True) + "\n")
    with open(drop, "w", encoding="utf-8") as fh:
        for pair in pairs:
            if not pair.verified:
                fh.write(json.dumps(pair.to_dict(), sort_keys=True) + "\n")
    payload = stats.to_dict()
    if crosscheck is not None:
        payload["mouse_crosscheck"] = crosscheck
    with open(os.path.join(outdir, "stats.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return keep


def read_pairs(path: str) -> List[GroundingPair]:
    out: List[GroundingPair] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append(GroundingPair(
                sample=d["sample"], view=d["view"], screenshot=d["screenshot"],
                entity=d["entity"], kind=d["kind"], description=d["description"],
                x=int(d["x"]), y=int(d["y"]),
                point=tuple(float(c) for c in d["point"]),
                verified=bool(d["verified"]), selected=d.get("selected", ""),
                reason=d.get("reason", ""), brief=d.get("brief", ""),
                width=int(d.get("width", 0)), height=int(d.get("height", 0))))
    return out


def generate(outdir: str, count: int = 4, seed: int = 0,
             views: Sequence[str] = DEFAULT_VIEWS,
             kinds: Sequence[str] = DEFAULT_KINDS,
             crosscheck: int = 0, progress: bool = False) -> CorpusStats:
    """Generate, cross-check, and write a corpus. The one call the CLI needs."""
    with CorpusGenerator(outdir, views=views, kinds=kinds) as gen:
        pairs, stats = gen.run(count=count, seed=seed, progress=progress)
        check = gen.crosscheck_mouse(pairs, limit=crosscheck) if crosscheck else None
    write_corpus(outdir, pairs, stats, check)
    return stats


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("outdir")
    ap.add_argument("--count", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--views", default=",".join(DEFAULT_VIEWS))
    ap.add_argument("--crosscheck", type=int, default=0,
                    help="synthesise N real mouse clicks and compare with the picker")
    args = ap.parse_args(list(argv) if argv is not None else None)
    if not vp.gui_available():
        print("FreeCAD GUI not found; nothing to do.")
        return 2
    stats = generate(args.outdir, count=args.count, seed=args.seed,
                     views=tuple(v.strip() for v in args.views.split(",") if v.strip()),
                     crosscheck=args.crosscheck, progress=True)
    print(json.dumps(stats.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
