"""CADSpot -- a ScreenSpot-style GUI grounding benchmark for a CAD application.

ScreenSpot asks: given a screenshot and a text referent, predict a pixel; score
``point_in_bbox``. It is the number everyone quotes (GPT-4o on raw pixels ~20%,
ShowUI-2B ~75%, OS-Atlas-7B ~82-85%, UI-TARS ~90%). Every one of those numbers is
measured on a surface that has an accessibility tree, because that is the only
surface anyone can build ground truth for.

CADSpot splits a CAD session into the four surfaces a user actually clicks, and
scores each separately, because they are not the same problem:

===============  ==========================================================
region           ground truth
===============  ==========================================================
``toolbar``      the Qt widget's name + rect, free from the UIA tree
``dialog``       ditto -- task-panel spinboxes, checkboxes, OK/Cancel
``tree``         ditto -- the feature tree's items
``viewport``     **nothing gives it to you.** It comes from
                 :mod:`harnesscad.eval.grounding.corpus`: we own the B-rep, we
                 know the camera, we project, and the APPLICATION adjudicates.
===============  ==========================================================

The first three cost a tree walk (~0.2 s for 156 elements) and are, honestly, a
CAD-flavoured re-run of what OS-Atlas already did for Windows. **The fourth is the
contribution.** No CAD GUI grounding dataset exists anywhere, and the reason is
that the viewport cannot be scraped and a human would have to label it. Ours costs
nothing to produce and its labels are exact.

Two metrics, and the difference between them matters
----------------------------------------------------
``point_in_bbox`` -- ScreenSpot's metric. Offline, no application needed. For the
chrome it is exactly right: a button IS a rectangle. For the viewport it is a
PROXY, and a lenient one: an entity's projected bounding box includes pixels that
belong to whatever occludes it, so a model can score a hit on a box while clicking
a face it did not mean. We report it because it is the comparable number, and we
label it as a proxy because it is one.

``selects_expected`` -- the metric the viewport actually deserves, and one that no
web benchmark can have. Put the model's predicted pixel to the application's own
picker and ask what it selected. That is not a proxy for the task; it IS the task.
It needs a live FreeCAD, and it is the number to believe.

Reward hacking
--------------
A benchmark harness may drive the app's Python interpreter -- it is the grader. A
TRAINING environment must not expose it to the policy: with a console in the
action space the optimal policy is "paste a script, terminate", which scores 100%
and learns nothing about a GUI. See the warning in
:mod:`harnesscad.io.cua.viewport`.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from harnesscad.eval.grounding import corpus as corpus_mod
from harnesscad.io.cua import viewport as vp

#: The four surfaces. Their proportions in a real CAD session are roughly
#: 40 / 20 / 10 / 30 (toolbar / dialog / tree / viewport) -- so the region nobody
#: can ground is about a third of every click a CAD agent will ever make.
REGIONS: Tuple[str, ...] = ("toolbar", "dialog", "tree", "viewport")

#: A predictor: ``(image_path, instruction, width, height) -> (x, y)``.
Predictor = Callable[[str, str, int, int], Tuple[float, float]]


@dataclass(frozen=True)
class Target:
    """One grounding question. ``bbox`` is ``(left, top, right, bottom)``, image-local."""

    region: str
    instruction: str
    image: str
    width: int
    height: int
    bbox: Tuple[int, int, int, int]
    entity: str = ""              # viewport only: the entity a click must select
    sample: str = ""
    view: str = ""
    point: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def center(self) -> Tuple[int, int]:
        left, top, right, bottom = self.bbox
        return ((left + right) // 2, (top + bottom) // 2)

    def contains(self, x: float, y: float) -> bool:
        """ScreenSpot's whole metric: ``pointinbbox``."""
        left, top, right, bottom = self.bbox
        return left <= x <= right and top <= y <= bottom

    def to_dict(self) -> dict:
        return {"region": self.region, "instruction": self.instruction,
                "image": self.image, "width": self.width, "height": self.height,
                "bbox": list(self.bbox), "entity": self.entity,
                "sample": self.sample, "view": self.view, "point": list(self.point)}

    @classmethod
    def from_dict(cls, d: dict) -> "Target":
        return cls(region=d["region"], instruction=d["instruction"],
                   image=d["image"], width=int(d["width"]), height=int(d["height"]),
                   bbox=tuple(int(v) for v in d["bbox"]), entity=d.get("entity", ""),
                   sample=d.get("sample", ""), view=d.get("view", ""),
                   point=tuple(float(v) for v in d.get("point") or (0, 0, 0)))


@dataclass
class RegionScore:
    n: int = 0
    hits: int = 0
    selects: int = 0             # viewport only, and only when a live app scored it
    adjudicated: int = 0
    latency: float = 0.0

    @property
    def accuracy(self) -> float:
        return 0.0 if not self.n else self.hits / float(self.n)

    @property
    def error_rate(self) -> float:
        return 1.0 - self.accuracy

    @property
    def select_accuracy(self) -> float:
        return 0.0 if not self.adjudicated else self.selects / float(self.adjudicated)

    def to_dict(self) -> dict:
        out = {"n": self.n, "hits": self.hits,
               "point_in_bbox": round(self.accuracy, 4),
               "error_rate": round(self.error_rate, 4),
               "latency_ms": round(1000.0 * self.latency / max(self.n, 1), 2)}
        if self.adjudicated:
            out["selects_expected"] = round(self.select_accuracy, 4)
            out["adjudicated"] = self.adjudicated
        return out


@dataclass
class Report:
    predictor: str = ""
    regions: Dict[str, RegionScore] = field(default_factory=dict)

    @property
    def overall(self) -> RegionScore:
        total = RegionScore()
        for score in self.regions.values():
            total.n += score.n
            total.hits += score.hits
            total.selects += score.selects
            total.adjudicated += score.adjudicated
            total.latency += score.latency
        return total

    def to_dict(self) -> dict:
        return {"predictor": self.predictor,
                "overall": self.overall.to_dict(),
                "regions": {k: v.to_dict() for k, v in sorted(self.regions.items())}}


# --------------------------------------------------------------------------
# Building the benchmark
# --------------------------------------------------------------------------
#: How a UIA node's automation-id path maps onto a CADSpot region. Checked in
#: order; the first substring that appears in the id wins. This is the whole
#: reason the chrome is free: the Qt object path IS the semantic label.
_REGION_BY_AID: Tuple[Tuple[str, str], ...] = (
    ("Dlg", "dialog"),
    ("Dialog", "dialog"),
    ("TaskBox", "dialog"),
    ("TaskPanel", "dialog"),
    ("TaskAttacher", "dialog"),
    ("Tree", "tree"),
    ("ToolBar", "toolbar"),
)

#: Only ACTIONABLE controls are grounding targets. A ``GroupControl`` named
#: "GroupBox5", a ``PaneControl`` named "Gui::TaskView::TaskBox" and a
#: ``TextControl`` named "TextLabelX" are all in the tree, all have boxes, and are
#: none of them things a user clicks -- a static label is not a target, it is the
#: caption OF one. TuriX classifies a node by the ACTIONS it exposes rather than
#: by its role, which is the right instinct; this is the same test, spelled with
#: the control types Qt's UIA bridge actually emits.
_ACTIONABLE: Tuple[str, ...] = (
    "ButtonControl", "MenuItemControl", "SpinnerControl", "EditControl",
    "CheckBoxControl", "ComboBoxControl", "TreeItemControl", "ListItemControl",
    "RadioButtonControl", "TabItemControl",
)

#: The UIA tree must be walked DEEP. FreeCAD's task-panel fields sit ~18 levels
#: down (MainWindow > dock > TaskView > TaskBox > QFrame > Gui__Dialog__Placement
#: > GroupBox > spinbox), and a depth-12 walk -- the driver's default -- stops
#: short of every one of them. The dialog region then reports n=0, which does not
#: look like a truncated walk; it looks like FreeCAD has no dialogs.
_TREE_DEPTH = 25
_REGION_BY_TYPE: Dict[str, str] = {
    "ButtonControl": "toolbar",
    "MenuItemControl": "toolbar",
    "TreeItemControl": "tree",
    "SpinnerControl": "dialog",
    "EditControl": "dialog",
    "CheckBoxControl": "dialog",
    "ComboBoxControl": "toolbar",
}


def _classify(element: Any) -> str:
    aid = getattr(element, "automation_id", "") or ""
    for needle, region in _REGION_BY_AID:
        if needle in aid:
            return region
    return _REGION_BY_TYPE.get(getattr(element, "control_type", ""), "")


def chrome_targets(pid: int, image: str, window_rect: Tuple[int, int, int, int],
                   viewport_rect: Tuple[int, int, int, int]) -> List[Target]:
    """Harvest toolbar / dialog / tree targets from the UIA tree. Free and exact.

    Uses the CUA core's :class:`~harnesscad.io.cua.uia.UiaDriver` -- the ONE UIA
    driver in this repo; a second one would be a second set of bugs. If the
    ``uiautomation`` extra is absent this returns ``[]`` and the benchmark is
    viewport-only, which still works and still says something.

    Rects come back in virtual-desktop coordinates and are rebased into the
    window-local frame of the screenshot. Anything inside the viewport rect is
    DROPPED: the viewport is one opaque node and its box is not a grounding
    target, it is the region that has no targets, which is the entire point.
    """
    try:
        from harnesscad.io.cua import uia
    except Exception:                                   # noqa: BLE001
        return []
    if not uia.available():
        return []
    driver = uia.UiaDriver(pid=pid, max_depth=_TREE_DEPTH)
    wx, wy, _ww, _wh = window_rect
    vx, vy, vw, vh = viewport_rect
    width = window_rect[2]
    height = window_rect[3]
    seen: Dict[str, Target] = {}
    for element in driver.tree():
        if not element.enabled:
            continue
        if element.control_type not in _ACTIONABLE:
            continue
        region = _classify(element)
        if not region:
            continue
        # In a dialog the UIA *Name* is useless as a referent and the AutomationId
        # is perfect. The placement panel's three position spinboxes are all named
        # "Translation" -- one name, three different fields -- so a corpus keyed on
        # the name keeps ONE of them and silently drops the other two. Their
        # AutomationId leaves are ``xPos`` / ``yPos`` / ``zPos``: distinct, and a
        # typed op-parameter binding handed to us for free.
        aid = (element.automation_id or "").strip()
        leaf = aid.rsplit(".", 1)[-1] if aid else ""
        name = (element.name or "").strip()
        instruction = (leaf or name) if region == "dialog" else (name or leaf)
        if not instruction or instruction.startswith("Q"):   # QToolButton, QFrame
            continue
        left, top, right, bottom = element.rect
        if right <= left or bottom <= top:
            continue
        # Exclude the 3D view by IDENTITY, never by geometry.
        #
        # The obvious filter -- "drop anything whose box falls inside the viewport
        # rect" -- is wrong, and expensively so. FreeCAD's ``Tasks`` dock opens
        # OVER the area the 3D view still reports as its own (measured: the viewer
        # reports (646,365,1534,770), i.e. x up to 2180, while the Tasks dock sits
        # at x=1789), so the geometric test swallows every field of every task
        # panel and the dialog region reports n=0. That does not look like a bad
        # filter. It looks like FreeCAD has no dialogs.
        #
        # The viewport needs no geometric exclusion anyway: it is ONE opaque node
        # with no named children, which is the very fact this whole module exists
        # to work around. Naming it is enough.
        if "View3DInventor" in (element.automation_id or ""):
            continue
        if "GL" in (element.class_name or "") or "View3D" in (element.class_name or ""):
            continue
        box = (left - wx, top - wy, right - wx, bottom - wy)
        if box[0] < 0 or box[1] < 0 or box[2] > width or box[3] > height:
            continue
        key = "%s|%s|%s" % (region, aid or instruction, instruction)
        if key in seen:
            continue
        seen[key] = Target(region=region, instruction=instruction, image=image,
                           width=width, height=height, bbox=box)
    return list(seen.values())


def viewport_targets(pairs: Sequence[corpus_mod.GroundingPair],
                     radius: int = 12) -> List[Target]:
    """Turn VERIFIED corpus pairs into benchmark targets.

    The bbox is a square of ``radius`` px around the verified pixel. It is a
    PROXY and it is deliberately tight: an entity's true clickable region is an
    arbitrary polygon (a fillet is a sliver, a bore's wall is a crescent), and a
    generous box would score clicks on the wrong face as hits. The honest metric
    for this region is :func:`evaluate`'s ``selects_expected``, which puts the
    prediction to the app.
    """
    out: List[Target] = []
    for pair in pairs:
        if not pair.verified:
            continue
        out.append(Target(
            region="viewport", instruction=pair.description, image=pair.screenshot,
            width=pair.width, height=pair.height,
            bbox=(pair.x - radius, pair.y - radius,
                  pair.x + radius, pair.y + radius),
            entity=pair.entity, sample=pair.sample, view=pair.view,
            point=pair.point))
    return out


def save(path: str, targets: Sequence[Target]) -> str:
    with open(path, "w", encoding="utf-8") as fh:
        for t in targets:
            fh.write(json.dumps(t.to_dict(), sort_keys=True) + "\n")
    return path


def load(path: str) -> List[Target]:
    out: List[Target] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(Target.from_dict(json.loads(line)))
    return out


# --------------------------------------------------------------------------
# Baselines. No VLM required -- these are the floor and the ceiling.
# --------------------------------------------------------------------------
def center_predictor(image: str, instruction: str, w: int, h: int) -> Tuple[float, float]:
    """"Always click the middle." The floor, and a real one: on a benchmark whose
    targets cluster in the centre of a zoom-fitted viewport, a centre-clicking
    model can look deceptively competent. Reporting it keeps everyone honest."""
    return (w / 2.0, h / 2.0)


def random_predictor(seed: int = 0) -> Predictor:
    """Seeded uniform noise. Chance. Deterministic, so it is comparable run to run."""
    rng = random.Random(seed)

    def predict(image: str, instruction: str, w: int, h: int) -> Tuple[float, float]:
        return (rng.uniform(0, w), rng.uniform(0, h))
    return predict


def oracle_predictor() -> Predictor:
    """Our own projection, replayed. Not a baseline -- a HARNESS CHECK.

    It must score 1.0 on ``point_in_bbox`` (the label came from here) and, on a
    live app, ~1.0 on ``selects_expected``. If it does not, the benchmark plumbing
    is broken and every other number on the page is worthless. A benchmark that
    cannot detect its own miswiring is not a benchmark -- and this one caught its
    own: the first version keyed the oracle on ``(image, instruction)``, and BOTH
    halves of that key are wrong. The instruction is not unique (a part has four
    edges that are all "the straight edge at the front-left"), and ``evaluate``
    rebases the image path against the corpus root, so the lookup missed and the
    oracle silently degraded to clicking the centre -- scoring 0.0 and looking
    exactly like a broken projection rather than a broken harness.

    So it takes the TARGET, not a guess at the target's identity.
    """
    def predict(target: Target, image: str, instruction: str,
                w: int, h: int) -> Tuple[float, float]:
        return target.center
    predict.wants_target = True                # noqa: SLF001 - see evaluate()
    return predict


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def evaluate(targets: Sequence[Target], predictor: Predictor, name: str = "",
             root: str = "", adjudicator: Optional["Adjudicator"] = None) -> Report:
    """Score a predictor. ``adjudicator``, if given, asks the APP about the viewport."""
    report = Report(predictor=name or getattr(predictor, "__name__", "predictor"))
    wants_target = bool(getattr(predictor, "wants_target", False))
    for target in targets:
        score = report.regions.setdefault(target.region, RegionScore())
        image = os.path.join(root, target.image) if root else target.image
        width = target.width or (target.bbox[2] + 64)
        height = target.height or (target.bbox[3] + 64)
        t0 = time.perf_counter()
        if wants_target:
            x, y = predictor(target, image, target.instruction, width, height)
        else:
            x, y = predictor(image, target.instruction, width, height)
        score.latency += time.perf_counter() - t0
        score.n += 1
        if target.contains(x, y):
            score.hits += 1
        if adjudicator is not None and target.region == "viewport":
            got = adjudicator(target, x, y)
            if got is not None:
                score.adjudicated += 1
                if got == target.entity:
                    score.selects += 1
    return report


class Adjudicator:
    """Asks a LIVE FreeCAD what the model's predicted pixel actually selects.

    This is the metric that makes CADSpot's viewport split different in kind from
    ScreenSpot's. ScreenSpot can only ever ask "is the point in the box we wrote
    down". We can ask the application. The application is the thing the agent will
    ultimately have to convince, so its answer is not an approximation of the truth
    -- it is the truth.

    Groups targets by (sample, view) so each document is loaded once. ORACLE ONLY:
    it drives the app's Python channel, and must never be reachable from a policy.
    """

    def __init__(self, generator: "corpus_mod.CorpusGenerator") -> None:
        self.generator = generator
        self._current: Tuple[str, str] = ("", "")

    def __call__(self, target: Target, x: float, y: float) -> Optional[str]:
        ctl = self.generator.controller
        if ctl is None or target.sample not in self.generator._steps:
            return None
        if (target.sample, target.view) != self._current:
            if target.sample != self._current[0]:
                self.generator.load(self.generator._steps[target.sample])
            ctl.set_named_view(target.view)
            self._current = (target.sample, target.view)
        camera = ctl.camera()
        # The prediction is in IMAGE coords (y down); the picker wants y-up.
        py = float(camera.height_px - 1) - float(y)
        got = ctl.pick([(float(x), py)])
        return (got[0] or "") if got else ""


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------
def build(outdir: str, count: int = 6, seed: int = 0,
          views: Sequence[str] = corpus_mod.DEFAULT_VIEWS,
          progress: bool = False) -> Tuple[List[Target], dict]:
    """Build the whole benchmark: chrome from UIA, viewport from the corpus."""
    os.makedirs(outdir, exist_ok=True)
    targets: List[Target] = []
    meta: Dict[str, Any] = {}
    with corpus_mod.CorpusGenerator(outdir, views=views) as gen:
        pairs, stats = gen.run(count=count, seed=seed, progress=progress)
        meta["corpus"] = stats.to_dict()
        targets.extend(viewport_targets(pairs))

        # Chrome: one screenshot of the live window, one UIA walk.
        ctl = gen.controller
        assert ctl is not None and gen.bridge is not None
        ctl.focus_window()
        window = gen.bridge.call(
            "mw = Gui.getMainWindow()\n"
            "g = mw.frameGeometry()\n"
            "RESULT = [g.x(), g.y(), g.width(), g.height()]")
        window = (int(window[0]), int(window[1]), int(window[2]), int(window[3]))
        shot = os.path.join(outdir, "images", "chrome.png")
        try:
            from PIL import ImageGrab
            box = (window[0], window[1], window[0] + window[2], window[1] + window[3])
            ImageGrab.grab(bbox=box, all_screens=True).save(shot)
        except Exception as exc:                        # noqa: BLE001
            meta["chrome_error"] = "screenshot failed: %s" % exc
        pid = gen.bridge._proc.pid if gen.bridge._proc else 0
        rect = ctl.viewport_rect()
        try:
            chrome = chrome_targets(pid, "images/chrome.png", window, rect)
        except Exception as exc:                        # noqa: BLE001
            meta["chrome_error"] = "UIA walk failed: %s" % exc
            chrome = []
        if not chrome and "chrome_error" not in meta:
            meta["chrome_error"] = "the UIA extra is absent; viewport-only benchmark"
        targets.extend(chrome)

        # The DIALOG region does not exist until a dialog does. Open a real task
        # panel (Part's primitives dialog) so the spinboxes, the checkboxes and
        # OK/Cancel are on screen and in the tree, then close it. Without this the
        # dialog split silently reports n=0 -- a benchmark region that is empty
        # because nobody opened one, which reads as "we have no dialogs".
        # It must be a DOCKED TASK PANEL, not Part's free-floating primitives
        # dialog: a separate top-level QDialog is not a descendant of
        # Gui::MainWindow, so a UIA walk scoped to the app window -- which is what
        # scoping it to the app window MEANS -- cannot see it, and the dialog
        # region silently stays empty. PartDesign's additive box docks into the
        # combo view, and its spinboxes then appear in the tree under their Qt
        # objectName (boxLength / boxWidth / boxHeight).
        # ``Std_Placement`` is the right choice and not an arbitrary one: it is a
        # STANDARD command (always registered, in every workbench and every
        # version), and PartDesign's primitive commands are not -- the additive
        # box lives inside a ``PartDesign_CompPrimitiveAdditive`` command GROUP and
        # has no top-level name to run, so ``PartDesign_AdditiveBox`` raises
        # "No such command". Placement docks into the combo view and brings a full
        # set of spinboxes (Position x/y/z, Angle, Axis x/y/z) with it.
        try:
            gen.bridge.call("Gui.Selection.clearSelection()\n"
                            "doc = App.ActiveDocument\n"
                            "Gui.Selection.addSelection(doc.Name, 'Model')\n"
                            "Gui.runCommand('Std_Placement')\n"
                            "Gui.updateGui()\nRESULT = 1")
            time.sleep(1.5)
            shot2 = os.path.join(outdir, "images", "dialog.png")
            try:
                from PIL import ImageGrab
                box = (window[0], window[1],
                       window[0] + window[2], window[1] + window[3])
                ImageGrab.grab(bbox=box, all_screens=True).save(shot2)
            except Exception:                           # noqa: BLE001
                pass
            # RE-READ the viewport rect. Docking a task panel RESIZES the 3D view,
            # so the rect measured a moment ago is stale -- and since the panel now
            # occupies pixels the viewport used to own, every dialog field tests as
            # "inside the viewport" and is discarded. The dialog region reports
            # n=0, and it looks like the dialog never opened. The viewport rect is
            # not a constant; it is a reading, and it must be taken at the moment
            # it is used.
            rect_now = ctl.viewport_rect()
            dialog = [t for t in chrome_targets(pid, "images/dialog.png",
                                                window, rect_now)
                      if t.region == "dialog"]
            targets.extend(dialog)
            gen.bridge.call("Gui.Control.closeDialog()\nGui.updateGui()\nRESULT = 1")
        except Exception as exc:                        # noqa: BLE001
            meta["dialog_error"] = str(exc)
    save(os.path.join(outdir, "cadspot.jsonl"), targets)
    meta["targets"] = {r: sum(1 for t in targets if t.region == r) for r in REGIONS}
    with open(os.path.join(outdir, "cadspot_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    return targets, meta


def baseline_report(targets: Sequence[Target], root: str = "",
                    live: bool = False) -> dict:
    """Run every baseline. ``live`` additionally adjudicates the viewport in the app."""
    out: Dict[str, Any] = {}
    predictors: List[Tuple[str, Predictor]] = [
        ("random", random_predictor(0)),
        ("center", center_predictor),
        ("projection-oracle", oracle_predictor()),
    ]
    if not live:
        for name, fn in predictors:
            out[name] = evaluate(targets, fn, name=name, root=root).to_dict()
        return out
    with corpus_mod.CorpusGenerator(os.path.join(root or ".", "_adjudicate")) as gen:
        # Rebuild the STEPs the targets refer to, so the app can be asked.
        steps = os.path.join(root, "steps")
        for target in targets:
            if target.sample and target.sample not in gen._steps:
                path = os.path.join(steps, "%s.step" % target.sample)
                if os.path.isfile(path):
                    gen._steps[target.sample] = path
        adj = Adjudicator(gen)
        for name, fn in predictors:
            out[name] = evaluate(targets, fn, name=name, root=root,
                                 adjudicator=adj).to_dict()
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="CADSpot: CAD GUI grounding benchmark")
    ap.add_argument("outdir")
    ap.add_argument("--count", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--live", action="store_true",
                    help="adjudicate viewport predictions in a live FreeCAD")
    args = ap.parse_args(list(argv) if argv is not None else None)
    if not vp.gui_available():
        print("FreeCAD GUI not found; nothing to do.")
        return 2
    targets, meta = build(args.outdir, count=args.count, seed=args.seed,
                          progress=True)
    print(json.dumps({"targets": meta["targets"]}, indent=2))
    report = baseline_report(targets, root=args.outdir, live=args.live)
    with open(os.path.join(args.outdir, "baselines.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
