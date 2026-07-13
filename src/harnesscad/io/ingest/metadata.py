"""metadata — STEP AP242 metadata extraction via OCCT XCAF.

Pulls the *non-geometric* payload a modern STEP file carries: the part name, its
material, a bill-of-materials (BOM) rolled up from the assembly's product
structure, PMI (Product Manufacturing Information: tolerances / datums /
annotations) and the assembly tree. This is what turns an imported solid into a
document you can cost, source and manufacture from — not just measure.

OCCT's XCAF document model (``STEPCAFControl_Reader`` -> ``TDocStd_Document`` ->
``XCAFDoc_ShapeTool`` / ``XCAFDoc_MaterialTool`` / ``XCAFDoc_DimTolTool``) is the
standard route; AP242 PMI rides the same document. Everything is imported LAZILY
and GUARDED: with no kernel, a missing file, or a non-STEP file,
:func:`extract_metadata` returns an empty :class:`PartMetadata` with a ``note`` —
it NEVER raises.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PartMetadata:
    """Structured, non-geometric metadata for an imported part / assembly.

    - ``name``          : root product name (empty if unknown).
    - ``material``      : material label when present.
    - ``bom_lines``     : ``[{name, quantity, material?}]`` rolled up from the
                          assembly product structure (a flat BOM).
    - ``pmi``           : list of PMI/tolerance annotation records (or ``None``).
    - ``assembly_tree`` : nested ``{name, children:[...]}`` tree (or ``None``).
    - ``available``     : True only when a document was actually read.
    - ``note``          : why the record is empty / a status message.
    """

    path: str
    name: str = ""
    material: Optional[str] = None
    bom_lines: List[dict] = field(default_factory=list)
    pmi: Optional[List[dict]] = None
    assembly_tree: Optional[dict] = None
    available: bool = False
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.available

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "name": self.name,
            "material": self.material,
            "bom_lines": [dict(b) for b in self.bom_lines],
            "pmi": None if self.pmi is None else [dict(p) for p in self.pmi],
            "assembly_tree": self.assembly_tree,
            "available": self.available,
            "note": self.note,
        }


def _step_ext(path: str) -> bool:
    _, ext = os.path.splitext(path or "")
    return ext.lower() in (".step", ".stp")


def _empty(path: str, note: str) -> PartMetadata:
    return PartMetadata(path=path, note=note)


def extract_metadata(path: str) -> PartMetadata:
    """Extract STEP AP242 metadata via XCAF, degrading gracefully.

    Returns a :class:`PartMetadata`. When the file is missing, is not a STEP
    file, OCCT/XCAF is unavailable, or reading fails, the record is empty with a
    descriptive ``note`` — never an exception.
    """
    if not path or not os.path.exists(path):
        return _empty(path, f"file not found: {path!r}")
    if not os.path.isfile(path):
        return _empty(path, f"not a file: {path!r}")
    if not _step_ext(path):
        return _empty(
            path, "metadata extraction supports STEP (.step/.stp) via XCAF only")

    try:
        return _extract_xcaf(path)
    except Exception as exc:  # noqa: BLE001 - any XCAF/kernel failure -> empty
        return _empty(
            path,
            f"XCAF metadata unavailable ({type(exc).__name__}: {exc}); "
            "install the 'cadquery' extra with OCP for STEP AP242 metadata")


# --------------------------------------------------------------------------- #
# XCAF extraction (only reached when OCP is importable)
# --------------------------------------------------------------------------- #
def _extract_xcaf(path: str) -> PartMetadata:
    from OCP.STEPCAFControl import STEPCAFControl_Reader
    from OCP.TDocStd import TDocStd_Document
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.XCAFApp import XCAFApp_Application
    from OCP.XCAFDoc import XCAFDoc_DocumentTool
    from OCP.IFSelect import IFSelect_ReturnStatus

    app = XCAFApp_Application.GetApplication_s()
    doc = TDocStd_Document(TCollection_ExtendedString("XCAF"))
    app.InitDocument(doc)

    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    reader.SetMatMode(True)
    reader.SetColorMode(True)
    reader.SetGDTMode(True)  # PMI / GD&T
    status = reader.ReadFile(path)
    if status != IFSelect_ReturnStatus.IFSelect_RetDone:
        raise ValueError("STEPCAFControl_Reader failed to read file")
    if not reader.Transfer(doc):
        raise ValueError("STEPCAFControl_Reader failed to transfer document")

    main = doc.Main()
    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(main)

    sole_material = _materials_present(main)
    tree, bom = _walk_assembly(shape_tool, sole_material)
    name = tree.get("name", "") if tree else ""
    material = sole_material or _first_material(bom)
    pmi = _extract_pmi(main)

    available = bool(tree) or bool(bom)
    note = "" if available else "no product structure found in STEP document"
    return PartMetadata(
        path=path, name=name, material=material,
        bom_lines=bom, pmi=pmi, assembly_tree=tree,
        available=available, note=note)


def _label_name(label) -> str:
    from OCP.TDataStd import TDataStd_Name
    attr = TDataStd_Name()
    try:
        if label.FindAttribute(TDataStd_Name.GetID_s(), attr):
            return attr.Get().ToExtString()
    except Exception:  # noqa: BLE001
        return ""
    return ""


def _materials_present(main) -> Optional[str]:
    """Return the sole material name if the document defines exactly one.

    Read via the ``XCAFDoc_MaterialTool`` (the material-definition labels), which
    is safe on this OCP build; a per-shape ``FindAttribute`` into a freshly
    constructed ``XCAFDoc_Material`` hard-crashes it, so we never do that. When
    the document carries a single material we attach it to the BOM; otherwise we
    leave material honestly unresolved (``None``) rather than guess associations.
    """
    try:
        from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_MaterialTool
        from OCP.TDF import TDF_LabelSequence
        mat_tool = XCAFDoc_DocumentTool.MaterialTool_s(main)
        labels = TDF_LabelSequence()
        mat_tool.GetMaterialLabels(labels)
        if labels.Length() != 1:
            return None
        name = _label_name(labels.Value(1))
        return name or None
    except Exception:  # noqa: BLE001
        return None


def _walk_assembly(shape_tool, sole_material=None):
    """Return (assembly_tree, flat_bom) from the XCAF shape tool.

    Uses the static ``XCAFDoc_ShapeTool`` ``*_s`` methods (the instance-method
    aliases are not exposed by this OCP build) and never chains into unbound
    label internals — a wrong native call there hard-crashes the interpreter.
    """
    from OCP.TDF import TDF_LabelSequence, TDF_Label
    from OCP.XCAFDoc import XCAFDoc_ShapeTool

    roots = TDF_LabelSequence()
    shape_tool.GetFreeShapes(roots)
    bom: List[dict] = []

    def visit(label) -> dict:
        name = _label_name(label) or "unnamed"
        node = {"name": name, "children": []}
        entry = {"name": name, "quantity": 1}
        if sole_material and not XCAFDoc_ShapeTool.IsAssembly_s(label):
            entry["material"] = sole_material
        bom.append(entry)
        if XCAFDoc_ShapeTool.IsAssembly_s(label):
            comps = TDF_LabelSequence()
            XCAFDoc_ShapeTool.GetComponents_s(label, comps)
            for i in range(1, comps.Length() + 1):
                comp = comps.Value(i)
                referred = TDF_Label()
                if XCAFDoc_ShapeTool.GetReferredShape_s(comp, referred):
                    node["children"].append(visit(referred))
                else:
                    node["children"].append(visit(comp))
        return node

    if roots.Length() == 0:
        return None, bom
    if roots.Length() == 1:
        tree = visit(roots.Value(1))
    else:
        tree = {"name": "assembly", "children": [
            visit(roots.Value(i)) for i in range(1, roots.Length() + 1)]}
    bom = _rollup_bom(bom)
    return tree, bom


def _rollup_bom(bom: List[dict]) -> List[dict]:
    """Collapse identical (name, material) lines and sum quantities."""
    rolled: dict = {}
    order: List[tuple] = []
    for line in bom:
        key = (line.get("name"), line.get("material"))
        if key not in rolled:
            rolled[key] = dict(line)
            order.append(key)
        else:
            rolled[key]["quantity"] += line.get("quantity", 1)
    return [rolled[k] for k in order]


def _first_material(bom: List[dict]) -> Optional[str]:
    for line in bom:
        if line.get("material"):
            return line["material"]
    return None


def _extract_pmi(main) -> Optional[List[dict]]:
    """Best-effort PMI/GD&T annotation extraction. Returns None if none/failed."""
    try:
        from OCP.XCAFDoc import XCAFDoc_DocumentTool
        from OCP.TDF import TDF_LabelSequence
        dim_tool = XCAFDoc_DocumentTool.DimTolTool_s(main)
        pmi: List[dict] = []
        dims = TDF_LabelSequence()
        dim_tool.GetDimensionLabels(dims)
        for i in range(1, dims.Length() + 1):
            pmi.append({"kind": "dimension", "name": _label_name(dims.Value(i))})
        tols = TDF_LabelSequence()
        dim_tool.GetGeomToleranceLabels(tols)
        for i in range(1, tols.Length() + 1):
            pmi.append({"kind": "tolerance", "name": _label_name(tols.Value(i))})
        datums = TDF_LabelSequence()
        dim_tool.GetDatumLabels(datums)
        for i in range(1, datums.Length() + 1):
            pmi.append({"kind": "datum", "name": _label_name(datums.Value(i))})
        return pmi or None
    except Exception:  # noqa: BLE001 - PMI is optional, never fatal
        return None
