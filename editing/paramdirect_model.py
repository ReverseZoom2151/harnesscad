"""Data models for the two CAD paradigms integrated by Zou (2025),
*Parametric/direct CAD integration* (Technical Report ZJU).

The paper contrasts two solid-modeling paradigms:

* **Parametric (feature-based) modeling** — a *construction history*: an ordered
  sequence of features (2D sketch -> 3D sweep -> boolean combine), each carrying
  embedded parameters and *associativity* (constraints) so that a local
  parameter change propagates automatically. The ordered history defines a
  *model variation space* of admissible edits.
* **Direct (B-rep) modeling** — a *history-free* boundary representation: a
  collection of interconnected faces (carrying surfaces = geometry, connections
  = topology) whose entities are all free to change. Editing is done by grabbing
  a face and *push-pulling* it, which (unlike the older *tweak* operation) is
  allowed to violate the pre-edit topology.

This module provides deterministic, stdlib-only data models for both paradigms
plus the paper's *information-layer* taxonomy from the conclusion: a CAD model
carries information on **topology**, **geometry**, and **constraints**;
"parametric and direct edits work at different layers of information, i.e., the
constraint layer and the geometry layer, respectively." Two edit-operation
classes (:class:`ParameterEdit`, :class:`PushPullEdit`) are provided, and
:func:`classify_edit` / :func:`edit_layer` reproduce that paradigm/layer
classification.

No wall clock, no RNG; dict round-trips exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Paradigm / information-layer taxonomy (Sections 3 and 5)
# ---------------------------------------------------------------------------
class Paradigm(str, Enum):
    """The two integrated modeling paradigms."""

    PARAMETRIC = "parametric"
    DIRECT = "direct"


class InfoLayer(str, Enum):
    """The three information layers a CAD model carries (conclusion).

    Parametric edits act on the CONSTRAINT layer; direct edits on the GEOMETRY
    layer. TOPOLOGY is the connective layer both may disturb.
    """

    TOPOLOGY = "topology"
    GEOMETRY = "geometry"
    CONSTRAINT = "constraint"


_QUANT = 1_000_000  # 1e-6 tolerance for geometric equality


def _q(v: float) -> int:
    return int(round(float(v) * _QUANT))


# ---------------------------------------------------------------------------
# Parametric paradigm: feature tree = construction history
# ---------------------------------------------------------------------------
@dataclass
class ParametricFeature:
    """One step of a construction history (a *feature*).

    ``params`` are the embedded, editable parameters (the parametric handle).
    ``refs`` are ids of earlier features this one is positioned/associated with
    (the associativity / design-intent network). ``direct_edit`` marks a feature
    that originated from a direct edit (a pseudo-feature or a synchronous-tech
    direct-edit feature).
    """

    fid: str
    ftype: str
    params: Dict[str, float] = field(default_factory=dict)
    refs: Tuple[str, ...] = ()
    direct_edit: bool = False

    def to_dict(self) -> Dict:
        return {"fid": self.fid, "ftype": self.ftype,
                "params": dict(self.params), "refs": list(self.refs),
                "direct_edit": self.direct_edit}

    @staticmethod
    def from_dict(d: Dict) -> "ParametricFeature":
        return ParametricFeature(
            d["fid"], d["ftype"], dict(d.get("params", {})),
            tuple(d.get("refs", ())), bool(d.get("direct_edit", False)))


@dataclass
class FeatureTree:
    """An ordered construction history (list order == regeneration order)."""

    features: List[ParametricFeature] = field(default_factory=list)

    # -- lookup -----------------------------------------------------------
    def index_of(self, fid: str) -> int:
        for i, f in enumerate(self.features):
            if f.fid == fid:
                return i
        raise KeyError(fid)

    def get(self, fid: str) -> ParametricFeature:
        return self.features[self.index_of(fid)]

    def parameter(self, fid: str, name: str) -> float:
        return self.get(fid).params[name]

    def set_parameter(self, fid: str, name: str, value: float) -> None:
        f = self.get(fid)
        if name not in f.params:
            raise KeyError(name)
        f.params[name] = value

    # -- variation space (Section 3) --------------------------------------
    def variation_space(self) -> Tuple[Tuple[str, str], ...]:
        """The set of (feature, parameter) handles a simple edit can change.

        The construction history "will restrict the user to what can be edited";
        this enumerates that admissible edit space in deterministic order.
        """
        out: List[Tuple[str, str]] = []
        for f in self.features:
            for name in sorted(f.params):
                out.append((f.fid, name))
        return tuple(out)

    def dependents(self, fid: str) -> Tuple[str, ...]:
        """Features that reference ``fid`` (downstream in the associativity net)."""
        return tuple(f.fid for f in self.features if fid in f.refs)

    def copy(self) -> "FeatureTree":
        return FeatureTree.from_dict(self.to_dict())

    def to_dict(self) -> Dict:
        return {"features": [f.to_dict() for f in self.features]}

    @staticmethod
    def from_dict(d: Dict) -> "FeatureTree":
        return FeatureTree([ParametricFeature.from_dict(f)
                            for f in d.get("features", ())])


# ---------------------------------------------------------------------------
# Direct paradigm: history-free B-rep with push-pull
# ---------------------------------------------------------------------------
@dataclass
class Face:
    """A planar boundary face = carrying surface (geometry) + free offset.

    The carrying plane is ``normal . x = offset``; ``normal`` is a unit-ish
    axis. ``origin`` optionally records the parametric feature that produced the
    face (used by the translating/consistency reconciliation), but the direct
    model treats the face as free to change.
    """

    name: str
    nx: float
    ny: float
    nz: float
    offset: float
    origin: Optional[str] = None

    def normal(self) -> Tuple[float, float, float]:
        return (self.nx, self.ny, self.nz)

    def geometry_key(self) -> Tuple:
        return (_q(self.nx), _q(self.ny), _q(self.nz), _q(self.offset))

    def to_dict(self) -> Dict:
        return {"name": self.name, "nx": self.nx, "ny": self.ny,
                "nz": self.nz, "offset": self.offset, "origin": self.origin}

    @staticmethod
    def from_dict(d: Dict) -> "Face":
        return Face(d["name"], d["nx"], d["ny"], d["nz"], d["offset"],
                    d.get("origin"))


@dataclass
class DirectBRep:
    """A history-free B-rep: free faces (geometry) + face adjacency (topology).

    ``adjacency`` is a set of unordered face-name pairs (shared edges). Push-pull
    only moves a face along its normal; topology is left to the consistency
    checks to police (a push-pull may over-run a neighbour, violating topology).
    """

    faces: Dict[str, Face] = field(default_factory=dict)
    adjacency: List[Tuple[str, str]] = field(default_factory=list)

    def add_face(self, face: Face) -> None:
        self.faces[face.name] = face

    def connect(self, a: str, b: str) -> None:
        key = tuple(sorted((a, b)))
        if key not in {tuple(sorted(p)) for p in self.adjacency}:
            self.adjacency.append(key)  # type: ignore[arg-type]

    def neighbours(self, name: str) -> Tuple[str, ...]:
        out = []
        for a, b in self.adjacency:
            if a == name:
                out.append(b)
            elif b == name:
                out.append(a)
        return tuple(sorted(out))

    def push_pull(self, name: str, distance: float) -> None:
        """Move ``name`` along its own normal by ``distance`` (a direct edit).

        This mutates only the geometry layer (the face offset); no history is
        recorded and associativity is ignored, mirroring direct modeling's
        "make all geometric entities free to change" strategy.
        """
        self.faces[name].offset += float(distance)

    def copy(self) -> "DirectBRep":
        return DirectBRep.from_dict(self.to_dict())

    def to_dict(self) -> Dict:
        return {"faces": {k: v.to_dict() for k, v in self.faces.items()},
                "adjacency": [list(p) for p in self.adjacency]}

    @staticmethod
    def from_dict(d: Dict) -> "DirectBRep":
        b = DirectBRep()
        for k, v in d.get("faces", {}).items():
            b.faces[k] = Face.from_dict(v)
        b.adjacency = [tuple(p) for p in d.get("adjacency", [])]
        return b


# ---------------------------------------------------------------------------
# Edit operations + paradigm / layer classification
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ParameterEdit:
    """A parametric edit: change one feature parameter (constraint layer)."""

    target_fid: str
    param: str
    new_value: float

    def to_dict(self) -> Dict:
        return {"kind": "parameter", "target_fid": self.target_fid,
                "param": self.param, "new_value": self.new_value}


@dataclass(frozen=True)
class PushPullEdit:
    """A direct edit: push-pull a face by a distance (geometry layer)."""

    face_name: str
    distance: float

    def to_dict(self) -> Dict:
        return {"kind": "push_pull", "face_name": self.face_name,
                "distance": self.distance}


def edit_from_dict(d: Dict):
    if d["kind"] == "parameter":
        return ParameterEdit(d["target_fid"], d["param"], d["new_value"])
    if d["kind"] == "push_pull":
        return PushPullEdit(d["face_name"], d["distance"])
    raise ValueError(f"unknown edit kind: {d.get('kind')!r}")


def classify_edit(edit) -> Paradigm:
    """Classify an edit operation by the paradigm it belongs to."""
    if isinstance(edit, ParameterEdit):
        return Paradigm.PARAMETRIC
    if isinstance(edit, PushPullEdit):
        return Paradigm.DIRECT
    raise TypeError(f"not a paramdirect edit: {edit!r}")


def edit_layer(edit) -> InfoLayer:
    """The information layer an edit acts on (conclusion of the paper).

    Parametric edits act on the constraint layer; direct edits on the geometry
    layer.
    """
    para = classify_edit(edit)
    return InfoLayer.CONSTRAINT if para is Paradigm.PARAMETRIC else InfoLayer.GEOMETRY
