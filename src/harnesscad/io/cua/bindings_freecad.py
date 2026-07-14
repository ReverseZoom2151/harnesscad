"""bindings_freecad — the CISP-op -> UIA-control table, as DATA.

Everything FreeCAD-specific about driving the GUI lives here, as inert data
structures. The driver (:mod:`uia`) and the environment
(:mod:`environment_freecad`) contain no FreeCAD strings at all. **A second CAD
application is therefore a second TABLE, not a second agent** — which is the
whole reason this is data and not code.

The table is measured, not guessed. Every name and AutomationId below was read
off a live FreeCAD 1.1.1 UIA tree on this machine (see the probe transcript):

* the Part Design toolbar exposes ``Pad``, ``Pocket``, ``Revolve``, ``Groove``,
  ``Hole``, ``Additive Loft``, ``Additive Pipe``, ``Additive Primitive``,
  ``Boolean Operation``, ``Fillet``, ``Chamfer``, ``Draft``, ``Thickness``,
  ``Mirror``, ``Linear Pattern``, ``Polar Pattern`` — a vocabulary that is
  **isomorphic to the CISP op set**, so :data:`OP_TO_BUTTON` is a static dict and
  not a vision problem;
* ``Additive Primitive`` (invoked directly, not through its dropdown) opens the
  Box task panel, whose three quantity fields are exposed under their Qt
  objectNames ``boxLength`` / ``boxWidth`` / ``boxHeight``, with ``OK`` / ``Cancel``
  on a ``QDialogButtonBox``.

What is honestly NOT in here: any binding that needs a click inside the 3D
viewport. A fillet needs an EDGE, a pocket needs a FACE, a boolean needs a
selection in the model tree — those are picks, not parameters, and a table cannot
express them. They are listed in :data:`REQUIRES_VIEWPORT` with the reason, and
the environment refuses them rather than pretending.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: The top-level window, and the one node that needs vision.
WINDOW_CLASS = "Gui::MainWindow"
VIEWPORT_AID_CONTAINS = "View3DInventorViewer"   # NEVER select by ClassName: there
                                                 # is a decoy 100x30 QOpenGLWidget.

#: FreeCAD's own workbench, read off the workbench combo box.
WORKBENCH = "Part Design"
WORKBENCH_COMBO_AID = "Workbench.Gui::WorkbenchComboBox"

#: The title FreeCAD gives an unsaved document. The '*' is our free tripwire.
DIRTY_PREFIX = "*"


@dataclass(frozen=True)
class Control:
    """How to RESOLVE a control. Never a coordinate — the tree has the answer."""

    name: Optional[str] = None
    aid_suffix: Optional[str] = None
    control_type: Optional[str] = None

    def query(self) -> Dict[str, Any]:
        q: Dict[str, Any] = {}
        if self.name is not None:
            q["name"] = self.name
        if self.aid_suffix is not None:
            q["aid_suffix"] = self.aid_suffix
        if self.control_type is not None:
            q["control_type"] = self.control_type
        return q


@dataclass(frozen=True)
class FieldBind:
    """A typed op parameter -> a Qt objectName. Handed to us for free by the tree."""

    aid_suffix: str                 # 'boxLength'
    source: Tuple[str, str]         # ('add_rectangle', 'w') — op tag, field name
    control_type: str = "SpinnerControl"
    unit: str = "mm"


@dataclass(frozen=True)
class Guard:
    """A precondition on the op stream for a recipe to be APPLICABLE.

    The additive Box primitive is always placed at the body origin — the
    attachment-offset spinners are disabled ("inactive - not attached") until a
    reference is picked in the VIEWPORT, which is a pick and not a parameter. So a
    rectangle whose corner is not (0, 0) cannot be built by this recipe, and the
    environment says so instead of quietly building the box in the wrong place.
    """

    source: Tuple[str, str]
    equals: float
    reason: str = ""


@dataclass(frozen=True)
class Recipe:
    """One GUI procedure that realises a CONSECUTIVE run of CISP ops."""

    id: str
    pattern: Tuple[str, ...]
    buttons: Tuple[Control, ...]
    fields: Tuple[FieldBind, ...]
    confirm: Control
    guards: Tuple[Guard, ...] = ()
    note: str = ""


#: Session setup: the ops any recipe needs to have happened first.
NEW_DOCUMENT = Control(name="New Document", control_type="ButtonControl")
NEW_BODY = Control(name="New Body", control_type="ButtonControl")
RECOMPUTE = Control(name="Recompute", control_type="ButtonControl")
OK_BUTTON = Control(name="OK", control_type="ButtonControl")
CANCEL_BUTTON = Control(name="Cancel", control_type="ButtonControl")

#: THE ISOMORPHISM. Every CISP op and the FreeCAD toolbar button that realises it.
#: This is the whole reason a11y grounding beats vision here: op -> button is a
#: static dict.
OP_TO_BUTTON: Dict[str, str] = {
    "extrude": "Pad",
    "revolve": "Revolve",
    "hole": "Hole",
    "fillet": "Fillet",
    "chamfer": "Chamfer",
    "draft": "Draft",
    "shell": "Thickness",
    "loft": "Additive Loft",
    "sweep": "Additive Pipe",
    "boolean": "Boolean Operation",
    "mirror": "Mirror",
    "linear_pattern": "Linear Pattern",
    "circular_pattern": "Polar Pattern",
}

#: Ops whose FreeCAD button exists and is resolvable, but which CANNOT be driven
#: coordinate-free because they need a PICK (an edge, a face, a tree item) rather
#: than a parameter. Declared, with the reason, and refused — never faked.
REQUIRES_VIEWPORT: Dict[str, str] = {
    "extrude": "Pad needs a sketch drawn in the Sketcher, which is viewport drawing",
    "revolve": "needs a sketch profile + an axis picked in the viewport",
    "hole": "needs a face pick in the viewport",
    "fillet": "needs an EDGE selection in the viewport",
    "chamfer": "needs an EDGE selection in the viewport",
    "draft": "needs a FACE selection in the viewport",
    "shell": "Thickness needs a FACE selection in the viewport",
    "loft": "needs two profile sketches drawn in the Sketcher",
    "sweep": "needs a profile + a 3D path sketch",
    "boolean": "needs two bodies selected in the model tree",
    "mirror": "needs a feature selected in the model tree",
    "linear_pattern": "needs a feature selected in the model tree",
    "circular_pattern": "needs a feature selected in the model tree",
    "new_sketch": "the Sketcher is a drawing surface, not a parameter dialog",
    "add_point": "sketch drawing happens in the viewport",
    "add_line": "sketch drawing happens in the viewport",
    "add_circle": "sketch drawing happens in the viewport",
    "add_rectangle": "sketch drawing happens in the viewport",
    "constrain": "constraints are applied to picked sketch entities",
    "set_param": "editing a feature needs a model-tree pick",
    "add_instance": "assembly is a different workbench",
    "mate": "assembly is a different workbench",
}

#: THE RECIPES: op runs the GUI can build coordinate-free, end to end.
#:
#: ``box``: a rectangle sketched at the origin and padded is EXACTLY FreeCAD's
#: additive Box primitive — corner at the body origin, extents (length, width,
#: height). ``AddRectangle(x, y, w, h)`` has (x, y) as the CORNER (see
#: ``frep._Profile``), so the mapping is w->boxLength, h->boxWidth,
#: distance->boxHeight, and the guards demand the corner be the origin.
RECIPES: Tuple[Recipe, ...] = (
    Recipe(
        id="box",
        pattern=("new_sketch", "add_rectangle", "extrude"),
        buttons=(Control(name="Additive Primitive", control_type="ButtonControl"),),
        fields=(
            FieldBind("boxLength", ("add_rectangle", "w")),
            FieldBind("boxWidth", ("add_rectangle", "h")),
            FieldBind("boxHeight", ("extrude", "distance")),
        ),
        confirm=OK_BUTTON,
        guards=(
            Guard(("add_rectangle", "x"), 0.0,
                  "the additive Box primitive is placed at the body origin; its "
                  "attachment-offset fields are disabled until a reference is "
                  "PICKED IN THE VIEWPORT, so a rectangle whose corner is not "
                  "(0,0) cannot be built coordinate-free"),
            Guard(("add_rectangle", "y"), 0.0, ""),
            Guard(("new_sketch", "plane"), "XY",
                  "the primitive's base plane follows the body's XY origin plane"),
        ),
        note="successive additive primitives FUSE into the active body, which is "
             "the same union the op stream's successive extrudes produce",
    ),
)


def recipe_for(tags: Sequence[str]) -> Optional[Recipe]:
    for r in RECIPES:
        if tuple(tags) == r.pattern:
            return r
    return None


def match_recipes(ops: Sequence[Any]) -> Tuple[List[Tuple[Recipe, List[Any]]], List[str]]:
    """Greedily segment an op stream into recipes. Returns (matches, reasons-not).

    An op stream that does not segment cleanly is REPORTED, not partially built:
    half a part is worse than no part, because it looks like a part.
    """
    tags = [getattr(type(op), "OP", "") for op in ops]
    out: List[Tuple[Recipe, List[Any]]] = []
    reasons: List[str] = []
    i = 0
    while i < len(ops):
        hit = None
        for r in RECIPES:
            n = len(r.pattern)
            if tuple(tags[i:i + n]) == r.pattern:
                hit = (r, list(ops[i:i + n]))
                break
        if hit is None:
            tag = tags[i]
            reasons.append("op '%s' at index %d: %s" % (
                tag, i, REQUIRES_VIEWPORT.get(
                    tag, "no GUI recipe binds this op coordinate-free")))
            i += 1
            continue
        out.append(hit)
        i += len(hit[0].pattern)
    return out, reasons


def bind_values(recipe: Recipe, ops: Sequence[Any]) -> Dict[str, float]:
    """Resolve a recipe's ``FieldBind``s against the matched ops -> {aid: value}."""
    by_tag = {getattr(type(op), "OP", ""): op for op in ops}
    values: Dict[str, float] = {}
    for fb in recipe.fields:
        tag, attr = fb.source
        op = by_tag.get(tag)
        if op is None:
            raise KeyError("recipe %r wants op '%s', which is not in the match"
                           % (recipe.id, tag))
        values[fb.aid_suffix] = float(getattr(op, attr))
    return values


def check_guards(recipe: Recipe, ops: Sequence[Any]) -> List[str]:
    """Guard violations, as human-readable reasons. Empty = the recipe applies."""
    by_tag = {getattr(type(op), "OP", ""): op for op in ops}
    bad: List[str] = []
    for g in recipe.guards:
        tag, attr = g.source
        op = by_tag.get(tag)
        if op is None:
            continue
        got = getattr(op, attr)
        want = g.equals
        same = (str(got) == str(want) if isinstance(want, str)
                else abs(float(got) - float(want)) < 1e-12)
        if not same:
            bad.append("%s.%s is %r, must be %r%s"
                       % (tag, attr, got, want,
                          " -- " + g.reason if g.reason else ""))
    return bad
