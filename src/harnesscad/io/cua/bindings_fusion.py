"""bindings_fusion - the CISP-op -> Fusion 360 toolbar-control table, as DATA.

Everything Fusion-specific about driving the GUI lives here, as inert data, the
same way :mod:`bindings_freecad` does for FreeCAD. The environment
(:mod:`environment_fusion`) holds no toolbar strings of its own.

Calibration honesty and Fusion's a11y limits
---------------------------------------------
Fusion 360 draws much of its UI in a Qt/OpenGL surface, so its accessibility tree
is PARTIAL: the top ribbon and command dialogs expose UIAutomation nodes, but the
canvas and some Qt-drawn panels do not. This is stated plainly in the environment
capabilities. The command names below are authored from Fusion's published API
command ids (``adsk``) and marked for LIVE CALIBRATION; they name the INTENT and
must be confirmed against a running Fusion tree before any name is trusted.

The ORACLE is Fusion's own Python API (``adsk.core`` / ``adsk.fusion``): after the
GUI is driven, ``PhysicalProperties`` and ``BoundingBox3D`` are read back through
the API. Like Inventor - and UNLIKE SolidWorks's SI COM - Fusion's API works in
CENTIMETRES (its internal unit), so length converts x10 (cm -> mm), not x1000.
Because that API lives INSIDE Fusion (a script/add-in), the read is an in-process
oracle; in a real deployment the harness reaches it through a control-directory
channel exactly like FreeCAD's macro channel, so the agent never touches it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from harnesscad.core.cisp.ops import Op

#: Fusion has no COM ProgID: its automation surface is the in-process ``adsk``
#: Python API. This constant names the module the oracle imports (and which only
#: imports successfully INSIDE a running Fusion).
API_MODULE = "adsk.core"

#: Fusion's API is in CENTIMETRES (its internal unit), NOT SI metres. Volume is
#: cm^3, area cm^2, lengths cm. CISP authors in mm, so the oracle converts cm ->
#: mm (x10) for length, cm^3 -> mm^3 (x1000) for volume, cm^2 -> mm^2 (x100) for
#: area - explicitly, at the boundary.
CM3_TO_MM3 = 1.0e3
CM2_TO_MM2 = 1.0e2
CM_TO_MM = 1.0e1

#: The top-level frame window class, for the UIA driver where the ribbon is
#: reachable. LIVE CALIBRATION (Fusion's window class differs by release).
WINDOW_CLASS = "Qt5152QWindowIcon"


@dataclass(frozen=True)
class ToolbarControl:
    """How to RESOLVE a Fusion toolbar/command control. Never a coordinate."""

    command: str
    workspace: str = "Design"
    automation_id: Optional[str] = None

    def query(self) -> Dict[str, str]:
        q: Dict[str, str] = {"name": self.command}
        if self.automation_id is not None:
            q["aid_suffix"] = self.automation_id
        return q


@dataclass(frozen=True)
class GuiField:
    """One numeric value written into a Fusion command dialog, read back after."""

    automation_id: str
    source: str
    unit: str = "mm"
    #: Multiplier applied to the op value before typing (a diameter field takes 2x
    #: the op's radius). 1.0 for a field that takes the op value verbatim.
    scale: float = 1.0


@dataclass(frozen=True)
class GuiRecipe:
    """How ONE CISP op is built through the Fusion GUI coordinate-free."""

    op: str
    control: ToolbarControl
    fields: Tuple[GuiField, ...] = ()
    confirm: ToolbarControl = ToolbarControl(command="OK")
    origin_only: bool = False
    note: str = ""


#: THE ISOMORPHISM: every CISP op and the Fusion command that realises it.
OP_TO_COMMAND: Dict[str, str] = {
    "new_sketch": "Create Sketch",
    "add_rectangle": "2-Point Rectangle",
    "add_circle": "Center Diameter Circle",
    "extrude": "Extrude",
    "boolean": "Combine",
    "revolve": "Revolve",
    "fillet": "Fillet",
    "chamfer": "Chamfer",
    "hole": "Hole",
    "shell": "Shell",
    "draft": "Draft",
    "loft": "Loft",
    "sweep": "Sweep",
    "mirror": "Mirror",
    "linear_pattern": "Rectangular Pattern",
    "circular_pattern": "Circular Pattern",
}

CONFIRM = ToolbarControl(command="OK", automation_id="CommandDialogOK")
CANCEL = ToolbarControl(command="Cancel", automation_id="CommandDialogCancel")


#: THE RECIPES: the op subset the GUI can build coordinate-free, end to end.
RECIPES: Dict[str, GuiRecipe] = {
    "new_sketch": GuiRecipe(
        op="new_sketch",
        control=ToolbarControl("Create Sketch", workspace="Design",
                               automation_id="SketchCreate"),
        confirm=CONFIRM,
        origin_only=True,
        note="the sketch is created on the XY origin plane",
    ),
    "add_rectangle": GuiRecipe(
        op="add_rectangle",
        control=ToolbarControl("2-Point Rectangle", workspace="Design",
                               automation_id="SketchRectangle"),
        fields=(GuiField("RectangleWidth", "w"),
                GuiField("RectangleHeight", "h")),
        confirm=CONFIRM,
        origin_only=True,
        note="the rectangle's first corner is placed at the sketch origin",
    ),
    "add_circle": GuiRecipe(
        op="add_circle",
        control=ToolbarControl("Center Diameter Circle", workspace="Design",
                               automation_id="SketchCircle"),
        fields=(GuiField("CircleDiameter", "r", scale=2.0),),
        confirm=CONFIRM,
        origin_only=True,
        note="the circle is centred on the sketch origin; the field is a diameter, "
             "so the recipe runner doubles the op's radius",
    ),
    "extrude": GuiRecipe(
        op="extrude",
        control=ToolbarControl("Extrude", workspace="Design",
                               automation_id="Extrude"),
        fields=(GuiField("Distance", "distance"),),
        confirm=CONFIRM,
        note="Distance extrude of the active sketch profile",
    ),
    "boolean": GuiRecipe(
        op="boolean",
        control=ToolbarControl("Combine", workspace="Design",
                               automation_id="Combine"),
        confirm=CONFIRM,
        note="Combine>Join over the bodies; the only boolean expressible without a "
             "per-body browser pick",
    ),
}


#: Ops whose Fusion command resolves but which need a PICK. Refused, not faked.
REQUIRES_PICK: Dict[str, str] = {
    "add_point": "a sketch point needs a canvas click at a computed pixel",
    "add_line": "a line needs two canvas picks",
    "constrain": "a constraint needs the two sketch entities picked on the canvas",
    "revolve": "revolve needs an axis picked in the sketch",
    "fillet": "fillet needs the target EDGES picked on the canvas",
    "chamfer": "chamfer needs the target EDGES picked on the canvas",
    "hole": "a hole needs a face and a point picked",
    "shell": "shell needs the removed FACES picked",
    "draft": "draft needs faces and a pull direction picked",
    "loft": "loft needs the ordered profiles picked",
    "sweep": "sweep needs a profile and a path picked",
    "mirror": "mirror needs the seed feature and a mirror plane picked",
    "linear_pattern": "pattern needs the seed feature selected in the browser",
    "circular_pattern": "pattern needs the seed feature and an axis picked",
    "add_instance": "assembly is a separate component-insert workflow",
    "mate": "a joint needs two components picked in the assembly",
    "set_param": "editing a feature needs it selected in the timeline/browser first",
}


def value_for(op: Op, source: str) -> float:
    """Pull a field value off an op by attribute name. Raises if absent."""
    if not hasattr(op, source):
        raise KeyError("op %r has no attribute %r for a Fusion GUI binding"
                       % (type(op).__name__, source))
    return float(getattr(op, source))
