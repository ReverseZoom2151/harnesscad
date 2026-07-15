"""bindings_inventor - the CISP-op -> Autodesk Inventor ribbon-control table, DATA.

Everything Inventor-specific about driving the GUI lives here, as inert data, the
same way :mod:`bindings_freecad` does for FreeCAD and :mod:`bindings_solidworks`
does for SolidWorks. The environment (:mod:`environment_inventor`) holds no ribbon
strings of its own.

Calibration honesty
-------------------
Inventor is (almost certainly) not installed here, so the ribbon command names and
AutomationId hints below are authored from Inventor's published API command
vocabulary and marked for LIVE CALIBRATION. They name the INTENT; they must be
confirmed against a running Inventor UIA tree before any name is trusted. Until
then they are honest placeholders.

Inventor's ribbon is WPF and exposes UIAutomation; the command vocabulary (Start
2D Sketch, Rectangle, Circle, Extrude, Combine, Fillet, Chamfer, Hole, Shell,
Draft, Loft, Sweep, Revolve, Mirror, Rectangular/Circular Pattern) is isomorphic
to the CISP op set, so :data:`OP_TO_COMMAND` is a static dict, not a vision
problem.

The ORACLE is the Inventor COM automation object (``Inventor.Application``): after
the GUI is driven, ``ComponentDefinition.MassProperties`` is read back through
COM. A CRITICAL unit fact lives here: Inventor's API works in its internal
DATABASE units, which are CENTIMETRES - not the SI metres SolidWorks reports and
not the millimetres CISP authors in. The conversions below are therefore x10 (cm
-> mm) for length, not x1000.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from harnesscad.core.cisp.ops import Op

#: The COM ProgID of the Inventor application object. This is the ORACLE handle.
COM_PROGID = "Inventor.Application"

#: The Inventor document-type enum value for a Part document
#: (``kPartDocumentObject``). Used to open a harness-owned scratch part.
PART_DOCUMENT_OBJECT = 12290

#: Inventor's API is in CENTIMETRES (its internal database unit), NOT SI metres.
#: Volume is cm^3, area cm^2, lengths cm. CISP authors in mm, so the oracle
#: converts cm -> mm (x10) for length, cm^3 -> mm^3 (x1000) for volume, cm^2 ->
#: mm^2 (x100) for area - explicitly, at the boundary, or the differential
#: compare is silently wrong.
CM3_TO_MM3 = 1.0e3
CM2_TO_MM2 = 1.0e2
CM_TO_MM = 1.0e1

#: The top-level frame window class, for the UIA driver. LIVE CALIBRATION.
WINDOW_CLASS = "InventorFrameClass"

#: Inventor marks an unsaved document with a trailing asterisk in its title.
DIRTY_PREFIX = "*"


@dataclass(frozen=True)
class RibbonControl:
    """How to RESOLVE a ribbon control. Never a coordinate - the tree answers."""

    command: str
    tab: str = ""
    automation_id: Optional[str] = None

    def query(self) -> Dict[str, str]:
        q: Dict[str, str] = {"name": self.command}
        if self.automation_id is not None:
            q["aid_suffix"] = self.automation_id
        return q


@dataclass(frozen=True)
class GuiField:
    """One numeric value written into an Inventor dialog / mini-toolbar field.

    Every field is READ BACK after it is written; a field we cannot prove we set
    is a field the op does not proceed on.
    """

    automation_id: str
    source: str
    unit: str = "mm"
    #: Multiplier applied to the op value before typing (a diameter field takes 2x
    #: the op's radius). 1.0 for a field that takes the op value verbatim.
    scale: float = 1.0


@dataclass(frozen=True)
class GuiRecipe:
    """How ONE CISP op is built through the Inventor GUI coordinate-free."""

    op: str
    control: RibbonControl
    fields: Tuple[GuiField, ...] = ()
    confirm: RibbonControl = RibbonControl(command="OK")
    origin_only: bool = False
    note: str = ""


#: THE ISOMORPHISM: every CISP op and the Inventor command that realises it.
OP_TO_COMMAND: Dict[str, str] = {
    "new_sketch": "Start 2D Sketch",
    "add_rectangle": "Rectangle",
    "add_circle": "Circle",
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

CONFIRM = RibbonControl(command="OK", automation_id="uiOkButton")
CANCEL = RibbonControl(command="Cancel", automation_id="uiCancelButton")


#: THE RECIPES: the op subset the GUI can build coordinate-free, end to end.
RECIPES: Dict[str, GuiRecipe] = {
    "new_sketch": GuiRecipe(
        op="new_sketch",
        control=RibbonControl("Start 2D Sketch", tab="Model",
                              automation_id="Sketch2D"),
        confirm=CONFIRM,
        origin_only=True,
        note="the sketch is started on the XY origin plane",
    ),
    "add_rectangle": GuiRecipe(
        op="add_rectangle",
        control=RibbonControl("Rectangle", tab="Sketch",
                              automation_id="SketchRectangle"),
        fields=(GuiField("RectangleWidth", "w"),
                GuiField("RectangleHeight", "h")),
        confirm=CONFIRM,
        origin_only=True,
        note="the rectangle's first corner is constrained to the sketch origin",
    ),
    "add_circle": GuiRecipe(
        op="add_circle",
        control=RibbonControl("Circle", tab="Sketch", automation_id="SketchCircle"),
        fields=(GuiField("CircleDiameter", "r", scale=2.0),),
        confirm=CONFIRM,
        origin_only=True,
        note="the circle is centred on the sketch origin; the field is a diameter, "
             "so the recipe runner doubles the op's radius",
    ),
    "extrude": GuiRecipe(
        op="extrude",
        control=RibbonControl("Extrude", tab="Model", automation_id="PartExtrude"),
        fields=(GuiField("Distance", "distance"),),
        confirm=CONFIRM,
        note="Distance extrude of the active sketch profile",
    ),
    "boolean": GuiRecipe(
        op="boolean",
        control=RibbonControl("Combine", tab="Model", automation_id="PartCombine"),
        confirm=CONFIRM,
        note="Combine>Join over the solid bodies; the only boolean expressible "
             "without a per-body browser pick",
    ),
}


#: Ops whose Inventor command resolves but which need a PICK. Refused, not faked.
REQUIRES_PICK: Dict[str, str] = {
    "add_point": "a sketch point needs a graphics-window click at a computed pixel",
    "add_line": "a line needs two graphics-window picks",
    "constrain": "a constraint needs the two sketch entities picked in the graphics window",
    "revolve": "revolve needs an axis picked in the sketch",
    "fillet": "fillet needs the target EDGES picked in the graphics window",
    "chamfer": "chamfer needs the target EDGES picked in the graphics window",
    "hole": "a hole needs a face and a location picked",
    "shell": "shell needs the removed FACES picked",
    "draft": "draft needs faces and a pull direction picked",
    "loft": "loft needs the ordered sections picked",
    "sweep": "sweep needs a profile and a path picked",
    "mirror": "mirror needs the seed feature and a mirror plane picked",
    "linear_pattern": "pattern needs the seed feature selected in the browser",
    "circular_pattern": "pattern needs the seed feature and an axis picked",
    "add_instance": "assembly place is a different document type",
    "mate": "a constraint needs two components picked in the assembly",
    "set_param": "editing a feature needs it selected in the browser first",
}


def value_for(op: Op, source: str) -> float:
    """Pull a field value off an op by attribute name. Raises if absent."""
    if not hasattr(op, source):
        raise KeyError("op %r has no attribute %r for an Inventor GUI binding"
                       % (type(op).__name__, source))
    return float(getattr(op, source))
