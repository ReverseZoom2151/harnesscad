"""bindings_solidworks - the CISP-op -> SolidWorks-ribbon-control table, as DATA.

Everything SolidWorks-specific about driving the GUI lives here, as inert data
structures, exactly as :mod:`bindings_freecad` does for FreeCAD. The environment
(:mod:`environment_solidworks`) contains no ribbon strings of its own: **a second
CAD application is a second TABLE, not a second driver.**

Honesty about calibration
-------------------------
FreeCAD's table was read off a live UIA tree on this machine, so every name in it
is measured. SolidWorks is (almost certainly) not installed here, so the ribbon
command names and AutomationId hints below are authored from the published API /
CommandManager vocabulary and are marked for LIVE CALIBRATION: they name the
INTENT and must be confirmed against a running SolidWorks UIA tree once, the same
way FreeCAD's were. Until then they are honest placeholders, never presented as
verified.

The ribbon is MFC/WPF and exposes MSAA/UIAutomation; the SolidWorks CommandManager
command set is isomorphic to the CISP op vocabulary (Sketch, Corner Rectangle,
Circle, Extruded Boss/Base, Combine, Fillet, Chamfer, Hole Wizard, Shell, Draft,
Loft, Sweep, Revolve, Mirror, Linear/Circular Pattern), so :data:`OP_TO_COMMAND`
is a static dict, not a vision problem.

What is honestly NOT here: any binding that needs a click INSIDE the graphics
area. A fillet needs an EDGE, a shell needs a FACE, a boolean picks bodies in the
FeatureManager tree - those are picks, not parameters, and a table cannot express
them. They are listed in :data:`REQUIRES_PICK` with the reason, and the
environment refuses them rather than pretending.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from harnesscad.core.cisp.ops import Op

#: The COM ProgID of the SolidWorks application object (``ISldWorks``). This is
#: the ORACLE handle: after the GUI is driven, mass properties are read back
#: through this same application object over COM - a structured, synchronous read
#: the agent never actuates.
COM_PROGID = "SldWorks.Application"

#: SolidWorks stores model geometry in SI units through the API: volume in m^3,
#: area in m^2, lengths (centre of mass, bounding box) in metres. CISP ops are
#: authored in millimetres and the scripted backends measure in mm, so the oracle
#: converts at its boundary - explicitly - or the differential compare is off by
#: 10^9.
M3_TO_MM3 = 1.0e9
M2_TO_MM2 = 1.0e6
M_TO_MM = 1.0e3

#: The top-level frame window class, for the UIA driver to bind to. Marked for
#: LIVE CALIBRATION (read off the running app's tree; do not trust this string).
WINDOW_CLASS = "SWMainFrame"

#: SolidWorks marks an unsaved document with a trailing asterisk in its title, the
#: same free "the op really mutated the document" tripwire FreeCAD gives us.
DIRTY_PREFIX = "*"


@dataclass(frozen=True)
class RibbonControl:
    """How to RESOLVE a ribbon/CommandManager control. Never a coordinate.

    ``command`` is the human-facing command name (what the tooltip shows);
    ``tab`` is the CommandManager tab it lives on; ``automation_id`` is the UIA
    AutomationId hint when the control exposes one. The driver resolves by these,
    exactly as it does for FreeCAD, and can therefore REFUSE before dispatch.
    """

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
    """One numeric value written into a PropertyManager field, addressed by name.

    ``source`` is the op attribute that supplies the value (e.g. ``"distance"``,
    ``"w"``); ``unit`` is the unit the value is typed in. Every field is READ
    BACK after it is written (the 375mm defence), so a field we cannot prove we
    set is a field the op does not proceed on.
    """

    automation_id: str
    source: str
    unit: str = "mm"


@dataclass(frozen=True)
class GuiRecipe:
    """How ONE CISP op is built through the SolidWorks GUI coordinate-free.

    A CommandManager control opens a PropertyManager; the dialog fields are
    filled and read back; a confirm (the green check / OK) commits the feature.
    ``needs_body_guard`` records that the recipe places geometry at the part
    origin and cannot honour a non-origin placement without a graphics-area pick.
    """

    op: str
    control: RibbonControl
    fields: Tuple[GuiField, ...] = ()
    confirm: RibbonControl = RibbonControl(command="OK")
    origin_only: bool = False
    note: str = ""


#: THE ISOMORPHISM. Every CISP op and the SolidWorks command that realises it.
#: Static dict, not vision. Ops that need a pick still appear here (the command
#: exists and resolves) but are refused via :data:`REQUIRES_PICK`.
OP_TO_COMMAND: Dict[str, str] = {
    "new_sketch": "Sketch",
    "add_rectangle": "Corner Rectangle",
    "add_circle": "Circle",
    "extrude": "Extruded Boss/Base",
    "boolean": "Combine",
    "revolve": "Revolved Boss/Base",
    "fillet": "Fillet",
    "chamfer": "Chamfer",
    "hole": "Hole Wizard",
    "shell": "Shell",
    "draft": "Draft",
    "loft": "Lofted Boss/Base",
    "sweep": "Swept Boss/Base",
    "mirror": "Mirror",
    "linear_pattern": "Linear Pattern",
    "circular_pattern": "Circular Pattern",
}

#: The confirm control shared by the PropertyManager dialogs (the green check).
CONFIRM = RibbonControl(command="OK", automation_id="PropertyManagerOK")
CANCEL = RibbonControl(command="Cancel", automation_id="PropertyManagerCancel")


#: THE RECIPES: the op subset the GUI can build coordinate-free, end to end.
#:
#: These mirror FreeCAD/Onshape's drivable subset - sketch a profile at the
#: origin, then a feature - plus Combine, which SolidWorks can run as "Add" over
#: ALL solid bodies (an operation that needs no per-body pick and is therefore the
#: one boolean expressible coordinate-free).
RECIPES: Dict[str, GuiRecipe] = {
    "new_sketch": GuiRecipe(
        op="new_sketch",
        control=RibbonControl("Sketch", tab="Sketch", automation_id="Sketch"),
        confirm=CONFIRM,
        origin_only=True,
        note="the sketch is opened on the part's front/XY reference plane",
    ),
    "add_rectangle": GuiRecipe(
        op="add_rectangle",
        control=RibbonControl("Corner Rectangle", tab="Sketch",
                              automation_id="SketchRectangle"),
        fields=(GuiField("RectangleWidth", "w"),
                GuiField("RectangleHeight", "h")),
        confirm=CONFIRM,
        origin_only=True,
        note="the rectangle's first corner is dimensioned to the sketch origin; a "
             "non-origin corner needs a graphics-area pick and is refused",
    ),
    "add_circle": GuiRecipe(
        op="add_circle",
        control=RibbonControl("Circle", tab="Sketch", automation_id="SketchCircle"),
        fields=(GuiField("CircleRadius", "r"),),
        confirm=CONFIRM,
        origin_only=True,
        note="the circle is centred on the sketch origin",
    ),
    "extrude": GuiRecipe(
        op="extrude",
        control=RibbonControl("Extruded Boss/Base", tab="Features",
                              automation_id="InsertPadFeature"),
        fields=(GuiField("Depth", "distance"),),
        confirm=CONFIRM,
        note="Blind extrude of the active sketch profile by 'distance'",
    ),
    "boolean": GuiRecipe(
        op="boolean",
        control=RibbonControl("Combine", tab="Features", automation_id="InsertCombine"),
        confirm=CONFIRM,
        note="Combine>Add over all solid bodies; this is the only boolean that "
             "needs no per-body FeatureManager pick and so is coordinate-free",
    ),
}


#: Ops whose SolidWorks command exists and resolves, but which CANNOT be driven
#: coordinate-free because they need a PICK (an edge, a face, a body/feature in
#: the FeatureManager tree) rather than a parameter. Declared, with the reason,
#: and refused - never faked. Half a part looks like a part; that is worse than
#: none.
REQUIRES_PICK: Dict[str, str] = {
    "add_point": "a sketch point needs a graphics-area click at a computed pixel",
    "add_line": "a line needs two graphics-area picks",
    "constrain": "a relation needs the two sketch entities picked in the graphics area",
    "revolve": "revolve needs an axis/centreline picked in the sketch",
    "fillet": "fillet needs the target EDGES picked in the graphics area",
    "chamfer": "chamfer needs the target EDGES picked in the graphics area",
    "hole": "the Hole Wizard needs a face datum and a location picked",
    "shell": "shell needs the open FACES picked in the graphics area",
    "draft": "draft needs faces and a neutral plane picked",
    "loft": "loft needs the ordered profiles picked",
    "sweep": "sweep needs a profile and a path picked",
    "mirror": "mirror needs the seed feature and a mirror plane picked",
    "linear_pattern": "pattern needs the seed feature selected in the FeatureManager",
    "circular_pattern": "pattern needs the seed feature and an axis picked",
    "add_instance": "assembly insert is a different document type",
    "mate": "a mate needs two components picked in the assembly",
    "set_param": "editing a feature needs it selected in the FeatureManager first",
}


def value_for(op: Op, source: str) -> float:
    """Pull a field value off an op by attribute name. Raises if absent - a
    binding we cannot fill is a binding we do not attempt."""
    if not hasattr(op, source):
        raise KeyError("op %r has no attribute %r for a SolidWorks GUI binding"
                       % (type(op).__name__, source))
    return float(getattr(op, source))
