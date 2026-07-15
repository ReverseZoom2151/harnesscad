"""OnshapeBackend -- a scriptable GeometryBackend that drives Onshape's geometry
THROUGH the REST/FeatureScript API directly (the API is both actuator AND oracle).

Why this module exists (and how it differs from the CUA environment)
--------------------------------------------------------------------
:mod:`harnesscad.io.cua.environment_onshape` drives Onshape's *browser GUI* as the
actuator and uses the REST API only as a read-only ORACLE -- the two channels are
kept physically separate on purpose (that module's whole thesis). This module is
the complementary piece: here the REST API is the ACTUATOR too. CISP ops are
lowered to Onshape Feature-definition JSON and POSTed to the Part Studio
``features`` endpoint, so Onshape becomes a real :class:`GeometryBackend`
alongside CadQuery/FreeCAD/frep/stub -- not a GUI target but a cloud geometry
kernel we script.

It REUSES, verbatim, the authenticated transport already built for the oracle:
:class:`~harnesscad.io.cua.environment_onshape.OnshapeCredentials` (HMAC-SHA256
request signing, env-only secret handling, redacting ``repr``),
:class:`~harnesscad.io.cua.environment_onshape.OnshapeOracle` (signing, transport,
scratch-document lifecycle, and the mass-properties / bounding-box / feature-list
reads) and its :class:`DocumentRef` / :class:`MassProperties` / :class:`BoundingBox`
value types. Nothing in that stack is rebuilt; this module only ADDS the one thing
the oracle deliberately lacks -- a feature *write* (:class:`OnshapeFeatureClient`).

The op -> Onshape-feature mapping (see :data:`FEATURE_DOC`)
----------------------------------------------------------
A CISP sketch is built incrementally (``new_sketch`` then ``add_rectangle`` /
``add_circle``) but an Onshape sketch is ONE atomic feature carrying all its
entities, so the sketch ops are BUFFERED locally and flushed as a single
``newSketch`` feature the moment a downstream feature (``extrude``) first
references the sketch -- exactly the "sketch ops -> a Sketch feature" mapping the
design calls for. The constructive, coordinate-free subset is honoured with
Feature JSON modelled on Onshape's own documented examples
(https://onshape-public.github.io/docs/api-adv/featureaccess/):

    new_sketch  + add_rectangle / add_circle  ->  one BTMSketch-151 (newSketch)
    extrude                                   ->  BTMFeature-134  featureType=extrude
    boolean                                   ->  BTMFeature-134  featureType=boolean

Every op field is either honoured or the op is REFUSED with a typed diagnostic --
never silently approximated. Ops that in Onshape require a geometry PICK we cannot
fabricate query-free (fillet/chamfer edges, shell/hole faces, revolve axis, loft
path, patterns, assembly mates, ...) are refused with a reason naming exactly what
is missing, the same doctrine the CUA environment applies to viewport picks. A
half-built part is worse than an honest refusal.

Credentials -- a hard constraint (identical to the oracle's)
------------------------------------------------------------
This module NEVER handles, asks for, prints, or writes an API key or secret. It
reads ``ONSHAPE_ACCESS_KEY`` / ``ONSHAPE_SECRET_KEY`` from the environment ONLY,
through :class:`OnshapeCredentials`. When they are absent the backend's
constructor raises :class:`~harnesscad.io.backends.base.BackendUnavailable`
naming the two variables -- so the CISP server falls back and every live test
SKIPS. It never hangs and never prompts. A test may inject a mock client to
exercise the op->feature-JSON mapping fully offline (which is how this backend is
verified here, credentials being absent).

The backend NEVER destroys a user document
-------------------------------------------
Like the oracle, it creates a fresh name-stamped SCRATCH document, drives only
that, and deletes only that on :meth:`close`. No user document is ever opened,
written, or deleted. Document creation is LAZY (first op), so merely constructing
the backend touches no network beyond nothing.

Stdlib only. Absolute imports. Deterministic. Import-safe with no credentials.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Protocol, Tuple

from harnesscad.core.cisp.ops import (
    Op, NewSketch, AddPoint, AddLine, AddCircle, AddRectangle,
    Constrain, Extrude, Fillet, Boolean,
    Revolve, Chamfer, Hole, Shell, Draft,
    Loft, Sweep, LinearPattern, CircularPattern, Mirror,
    AddInstance, Mate, SetParam,
    canonical_json, edit_oplog,
)
from harnesscad.eval.verifiers.verify import Diagnostic, Severity
from harnesscad.io.backends.base import ApplyResult, BackendUnavailable
# Reuse the oracle's entire authenticated stack -- do NOT rebuild it.
from harnesscad.io.cua.environment_onshape import (
    ACCESS_ENV, SECRET_ENV,
    OnshapeCredentials, OnshapeOracle, OnshapeApiError,
    DocumentRef, MassProperties, BoundingBox,
)

#: Onshape's internal length unit is the METRE; CISP ops are authored in
#: millimetres. Sketch geometry fields (xCenter, radius, pnt/dir) are raw numbers
#: in metres, so they are converted here. (Feature *quantities* like an extrude
#: depth are passed as unit-bearing expression strings -- "20 mm" -- which
#: Onshape parses itself, so those need no manual conversion.)
MM_TO_M = 1.0e-3

#: CISP sketch planes -> Onshape's three default datum planes. Onshape's Top plane
#: is the XY plane (normal +Z), Front is XZ (normal +Y), Right is YZ (normal +X).
PLANE_TO_ONSHAPE: Dict[str, str] = {
    "XY": "Top", "YX": "Top",
    "XZ": "Front", "ZX": "Front",
    "YZ": "Right", "ZY": "Right",
    "top": "Top", "front": "Front", "right": "Right",
}

#: Onshape btType tags (from the documented feature-access examples). Named
#: constants so the mapping is legible and one place fixes a version bump.
BT_FEATURE_CALL = "BTFeatureDefinitionCall-1406"
BT_FEATURE = "BTMFeature-134"
BT_SKETCH = "BTMSketch-151"
BT_QUERY_LIST = "BTMParameterQueryList-148"
BT_INDIVIDUAL_QUERY = "BTMIndividualQuery-138"
BT_SKETCH_REGION_QUERY = "BTMIndividualSketchRegionQuery-140"
BT_PARAM_ENUM = "BTMParameterEnum-145"
BT_PARAM_QUANTITY = "BTMParameterQuantity-147"
BT_PARAM_BOOL = "BTMParameterBoolean-144"
BT_SKETCH_CURVE = "BTMSketchCurve-4"           # a full curve (circle)
BT_SKETCH_SEGMENT = "BTMSketchCurveSegment-155"  # a bounded segment (line)
BT_GEOM_CIRCLE = "BTCurveGeometryCircle-115"
BT_GEOM_LINE = "BTCurveGeometryLine-117"

#: Boolean kind -> Onshape BooleanOperationType enum value.
BOOLEAN_OP: Dict[str, str] = {"union": "UNION", "cut": "SUBTRACT",
                              "intersect": "INTERSECT"}

#: The ops this backend maps to real Onshape feature JSON, coordinate-free.
SUPPORTED_OPS: Tuple[str, ...] = (
    "new_sketch", "add_rectangle", "add_circle", "extrude", "boolean",
)

#: Every other op needs a geometry PICK / query this backend will NOT fabricate.
#: Refused with the reason -- the same "never fake a pick" doctrine the CUA
#: environment uses for the viewport, applied here to FeatureScript queries.
REFUSED_OPS: Dict[str, str] = {
    "add_point": "a bare sketch point contributes no region to extrude; Onshape "
                 "needs it constrained, which requires a sketch-solver binding not "
                 "built here",
    "add_line": "an open sketch chain needs closing + a region query to become a "
                "solid; only closed rectangle/circle profiles are mapped",
    "constrain": "a sketch constraint maps to a BTMSketchConstraint that must "
                 "name solved entity ids; not calibrated against a live sketch",
    "revolve": "revolve needs an axis query (a construction line or edge); this "
               "backend will not fabricate the axis geometry query",
    "fillet": "fillet needs an EDGE query; a CadQuery selector like '|Z' has no "
              "faithful Onshape query translation here",
    "chamfer": "chamfer needs an EDGE query; see fillet",
    "hole": "hole needs a FACE datum + a located sketch point (a viewport/query "
            "pick) to place the cut",
    "shell": "shell needs the open FACES named by a geometry query",
    "draft": "draft needs faces + a neutral-plane query",
    "loft": "loft needs the ordered profile region queries across sketches",
    "sweep": "sweep needs a profile region + a path query",
    "linear_pattern": "pattern needs a body query for the seed and a direction "
                      "query (an edge); direction cannot be given query-free",
    "circular_pattern": "pattern needs a body query for the seed and an axis query",
    "mirror": "mirror needs a body query for the seed and a mirror-plane query",
    "add_instance": "an assembly instance is a different element type (Assembly), "
                    "not a Part Studio feature",
    "mate": "a mate lives in an Assembly element and needs two instance picks",
    "set_param": "editing an Onshape feature is a POST to features/featureid/{fid}; "
                 "wired for creation, not yet for in-place edit replay",
    "add_arc": "an arc is a BTMSketchCurveSegment whose region must be closed and "
               "region-queried; only closed rectangle/circle profiles are mapped",
    "add_ellipse": "an ellipse is a BTMSketchCurve (BTCurveGeometryEllipse) whose "
                   "region needs a solver-bound region query not built here",
    "add_polygon": "a free polygon needs its closing loop region-queried; only "
                   "rectangle/circle profiles are mapped",
    "add_spline": "a spline is a BTMSketchCurve (interpolated) needing a "
                  "region query; only rectangle/circle profiles are mapped",
    "primitive": "a solid primitive is modelled in Onshape as a sketch + extrude "
                 "(or a FeatureScript primitive) whose region/plane picks this "
                 "backend will not fabricate coordinate-free",
    "split": "split needs a face/plane geometry query to name the cutting surface",
    "thicken": "thicken needs a face/sheet geometry query to name the surfaces to "
               "offset",
    "hull": "a convex hull needs the body queries for the bodies it encloses; "
            "this backend maps only the coordinate-free sketch+extrude+boolean "
            "subset",
    "minkowski": "a Minkowski sum / offset-solid has no query-free feature "
                 "mapping here; this backend maps only sketch+extrude+boolean",
}

#: Human-readable record of the op->feature mapping, surfaced by
#: ``query('mapping')`` and used in the report. DATA, so it is inspectable.
FEATURE_DOC: Dict[str, str] = {
    "new_sketch": "buffered; flushed with its entities as one BTMSketch-151 "
                  "(featureType=newSketch) on first reference, sketchPlane = "
                  "qCreatedBy(makeId(<Top|Front|Right>), EntityType.FACE)",
    "add_rectangle": "four BTMSketchCurveSegment-155 (BTCurveGeometryLine-117) "
                     "closing a loop, added to the buffered sketch",
    "add_circle": "one BTMSketchCurve-4 (BTCurveGeometryCircle-115) added to the "
                  "buffered sketch",
    "extrude": "BTMFeature-134 featureType=extrude: bodyType=SOLID, "
               "operationType=NEW, entities=BTMIndividualSketchRegionQuery-140 "
               "of the flushed sketch, endBound=BLIND, depth='<d> mm' "
               "(oppositeDirection for a negative distance)",
    "boolean": "BTMFeature-134 featureType=boolean: operationType enum "
               "UNION/SUBTRACT/INTERSECT; tools (and targets for SUBTRACT) are "
               "qCreatedBy(makeId(<featureId>), EntityType.BODY) of the operand "
               "features",
}


def _err(code: str, msg: str, where: Optional[str] = None) -> ApplyResult:
    return ApplyResult(False, [], [Diagnostic(Severity.ERROR, code, msg, where)])


# ---------------------------------------------------------------------------
# the ACTUATOR client -- the oracle plus a feature WRITE (the only new capability)
# ---------------------------------------------------------------------------
class OnshapeClient(Protocol):
    """What the backend needs from its transport. The default implementation is
    :class:`OnshapeFeatureClient` (a real signed REST client); a test injects a
    mock satisfying exactly this surface to prove the mapping offline."""

    def create_scratch_document(self, name: str) -> DocumentRef: ...
    def delete_document(self, did: str) -> None: ...
    def add_feature(self, ref: DocumentRef, feature: dict) -> dict: ...
    def features(self, ref: DocumentRef) -> List[dict]: ...
    def mass_properties(self, ref: DocumentRef) -> MassProperties: ...
    def bounding_box(self, ref: DocumentRef) -> BoundingBox: ...
    def export_stl(self, ref: DocumentRef) -> bytes: ...


class OnshapeFeatureClient(OnshapeOracle):
    """The oracle, EXTENDED with the one write it deliberately omits.

    :class:`OnshapeOracle` is read-only by design (it is an oracle). This subclass
    inherits ALL of it -- HMAC signing, transport, scratch-doc lifecycle, and every
    read -- and adds only ``add_feature`` / ``update_feature`` / ``delete_feature``,
    each a thin POST/DELETE through the inherited signed ``_request``. Nothing about
    the auth or read path is reimplemented.
    """

    def _features_path(self, ref: DocumentRef) -> str:
        return ("/api/%s/partstudios/d/%s/w/%s/e/%s/features"
                % (self.api_version, ref.did, ref.wid, ref.eid))

    def add_feature(self, ref: DocumentRef, feature: dict) -> dict:
        """POST .../partstudios/.../features -- create one feature. Returns the
        created feature payload (its ``feature.featureId`` is the handle)."""
        return self._request("POST", self._features_path(ref), body=feature)

    def update_feature(self, ref: DocumentRef, fid: str, feature: dict) -> dict:
        """POST .../features/featureid/{fid} -- edit an existing feature."""
        return self._request("POST", self._features_path(ref) + "/featureid/%s" % fid,
                             body=feature)

    def delete_feature(self, ref: DocumentRef, fid: str) -> dict:
        """DELETE .../features/featureid/{fid} -- remove a feature."""
        return self._request("DELETE",
                             self._features_path(ref) + "/featureid/%s" % fid)


# ---------------------------------------------------------------------------
# the op -> Feature-JSON builders (pure functions -- this is the mapping, testable)
# ---------------------------------------------------------------------------
def _plane_query(plane: str) -> dict:
    name = PLANE_TO_ONSHAPE[plane]
    return {
        "btType": BT_QUERY_LIST,
        "parameterId": "sketchPlane",
        "queries": [{
            "btType": BT_INDIVIDUAL_QUERY,
            "queryString": 'query=qCreatedBy(makeId("%s"), EntityType.FACE);' % name,
        }],
    }


def _line_segment(eid: str, x0: float, y0: float, x1: float, y1: float) -> dict:
    """One BTMSketchCurveSegment-155 from (x0,y0) to (x1,y1), coords in mm."""
    ax0, ay0 = x0 * MM_TO_M, y0 * MM_TO_M
    ax1, ay1 = x1 * MM_TO_M, y1 * MM_TO_M
    dx, dy = ax1 - ax0, ay1 - ay0
    length = (dx * dx + dy * dy) ** 0.5
    ux, uy = (dx / length, dy / length) if length else (1.0, 0.0)
    return {
        "btType": BT_SKETCH_SEGMENT,
        "entityId": eid,
        "startPointId": eid + ".start",
        "endPointId": eid + ".end",
        "startParam": 0.0,
        "endParam": length,
        "geometry": {
            "btType": BT_GEOM_LINE,
            "pntX": ax0, "pntY": ay0,   # a point on the line (the start)
            "dirX": ux, "dirY": uy,     # unit direction
        },
    }


def _rectangle_entities(prefix: str, x: float, y: float, w: float, h: float) -> List[dict]:
    """Four closed line segments for an axis-aligned rectangle, coords in mm."""
    corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    ents = []
    for i in range(4):
        x0, y0 = corners[i]
        x1, y1 = corners[(i + 1) % 4]
        ents.append(_line_segment("%s.line%d" % (prefix, i), x0, y0, x1, y1))
    return ents


def _circle_entity(eid: str, cx: float, cy: float, r: float) -> dict:
    return {
        "btType": BT_SKETCH_CURVE,
        "entityId": eid,
        "centerId": eid + ".center",
        "geometry": {
            "btType": BT_GEOM_CIRCLE,
            "radius": r * MM_TO_M,
            "xCenter": cx * MM_TO_M, "yCenter": cy * MM_TO_M,
            "xDir": 1.0, "yDir": 0.0, "clockwise": False,
        },
    }


def build_sketch_feature(name: str, plane: str, entities: List[dict]) -> dict:
    """A complete BTMSketch-151 (newSketch) feature-definition call."""
    return {
        "btType": BT_FEATURE_CALL,
        "feature": {
            "btType": BT_SKETCH,
            "featureType": "newSketch",
            "name": name,
            "parameters": [_plane_query(plane)],
            "entities": list(entities),
            "constraints": [],
        },
    }


def build_extrude_feature(name: str, sketch_feature_id: str, distance: float) -> dict:
    """A complete BTMFeature-134 featureType=extrude call.

    A negative CISP distance becomes a positive depth with oppositeDirection=true,
    which is how Onshape expresses a downward extrude (its depth is unsigned).
    """
    depth = abs(distance)
    params = [
        {"btType": BT_PARAM_ENUM, "parameterId": "bodyType",
         "enumName": "ExtendedToolBodyType", "value": "SOLID"},
        {"btType": BT_PARAM_ENUM, "parameterId": "operationType",
         "enumName": "NewBodyOperationType", "value": "NEW"},
        {"btType": BT_QUERY_LIST, "parameterId": "entities",
         "queries": [{"btType": BT_SKETCH_REGION_QUERY,
                      "featureId": sketch_feature_id}]},
        {"btType": BT_PARAM_ENUM, "parameterId": "endBound",
         "enumName": "BoundingType", "value": "BLIND"},
        {"btType": BT_PARAM_QUANTITY, "parameterId": "depth",
         "expression": "%g mm" % depth},
    ]
    if distance < 0:
        params.append({"btType": BT_PARAM_BOOL, "parameterId": "oppositeDirection",
                       "value": True})
    return {
        "btType": BT_FEATURE_CALL,
        "feature": {
            "btType": BT_FEATURE, "featureType": "extrude", "name": name,
            "parameters": params,
            "returnAfterSubfeatures": False, "suppressed": False,
        },
    }


def _body_query(parameter_id: str, feature_ids: List[str]) -> dict:
    return {
        "btType": BT_QUERY_LIST, "parameterId": parameter_id,
        "queries": [{
            "btType": BT_INDIVIDUAL_QUERY,
            "queryString": 'query=qCreatedBy(makeId("%s"), EntityType.BODY);' % fid,
        } for fid in feature_ids],
    }


def build_boolean_feature(name: str, kind: str, target_fid: str,
                          tool_fid: str) -> dict:
    """A complete BTMFeature-134 featureType=boolean call.

    UNION/INTERSECT operate on both bodies as ``tools``. SUBTRACT keeps ``targets``
    (the target body) and removes ``tools`` (the tool body); the query in each case
    is qCreatedBy(makeId(<featureId>), EntityType.BODY) of the operand feature.
    """
    op = BOOLEAN_OP[kind]
    params = [{"btType": BT_PARAM_ENUM, "parameterId": "operationType",
               "enumName": "BooleanOperationType", "value": op}]
    if op == "SUBTRACT":
        params.append(_body_query("targets", [target_fid]))
        params.append(_body_query("tools", [tool_fid]))
    else:
        params.append(_body_query("tools", [target_fid, tool_fid]))
    return {
        "btType": BT_FEATURE_CALL,
        "feature": {
            "btType": BT_FEATURE, "featureType": "boolean", "name": name,
            "parameters": params,
            "returnAfterSubfeatures": False, "suppressed": False,
        },
    }


# ---------------------------------------------------------------------------
# the backend
# ---------------------------------------------------------------------------
class OnshapeBackend:
    """A :class:`GeometryBackend` that actuates Onshape geometry over REST.

    Construct with credentials present (or a mock ``client`` injected) or it raises
    :class:`BackendUnavailable`. The scratch document is created lazily on the first
    applied op, so construction alone touches no network.
    """

    #: Exports available through the oracle. STL is synchronous; STEP is an
    #: asynchronous translation (POST .../translations then poll) and is declared
    #: but not wired in this build (see :meth:`export`).
    FORMATS = ("stl",)

    def __init__(self, credentials: Optional[OnshapeCredentials] = None,
                 client: Optional[OnshapeClient] = None,
                 scratch_name: Optional[str] = None,
                 keep_document: bool = False) -> None:
        self.creds = credentials or OnshapeCredentials()
        if client is None and not self.creds.present:
            raise BackendUnavailable(
                "onshape",
                "the onshape backend needs API credentials: set %s and %s "
                "(environment variables only -- never entered, printed or written). "
                "Absent them the backend skips; inject a mock client to test the "
                "op->feature mapping offline." % (ACCESS_ENV, SECRET_ENV),
                ["env: %s" % ACCESS_ENV, "env: %s" % SECRET_ENV])
        self.client: OnshapeClient = client or OnshapeFeatureClient(self.creds)
        self.scratch_name = scratch_name or ("harnesscad-scratch-%d"
                                             % int(time.time()))
        self.keep_document = bool(keep_document)
        self.reset()

    @staticmethod
    def available() -> bool:
        """Whether this backend can run live here (credentials present). Never
        raises, never prompts, never touches the secret's value."""
        return OnshapeCredentials().present

    # -- lifecycle ---------------------------------------------------------
    def reset(self) -> None:
        """Discard all local state. Does NOT create a document (that is lazy) and
        does NOT delete an existing scratch doc -- call :meth:`close` for that."""
        self.doc: Optional[DocumentRef] = None
        #: buffered sketches: sid -> {plane, entities:[{kind,params}], onshape_fid}
        self.sketches: Dict[str, dict] = {}
        #: local feature records mirroring the other backends' bookkeeping.
        self.features: List[dict] = []
        #: local fid -> the Onshape featureId the server minted (for body queries).
        self._onshape_fid: Dict[str, str] = {}
        self.solid_present = False
        self._oplog: List[Op] = []
        self._posted: List[dict] = []   # the feature JSON we POSTed, in order
        self._n = {"sk": 0, "e": 0, "f": 0}

    def _new_id(self, kind: str) -> str:
        self._n[kind] += 1
        return kind + str(self._n[kind]) if kind != "sk" else "sk" + str(self._n["sk"])

    def _ensure_document(self) -> Optional[ApplyResult]:
        """Create the scratch document on first use. Returns an error result if the
        REST call fails (block-and-correct), else None."""
        if self.doc is not None:
            return None
        try:
            self.doc = self.client.create_scratch_document(self.scratch_name)
        except OnshapeApiError as exc:
            return _err("api-error", "could not create scratch document: %s" % exc)
        return None

    # -- op dispatch -------------------------------------------------------
    def apply(self, op: Op) -> ApplyResult:
        if isinstance(op, SetParam):
            return self._set_param(op)
        result = self._dispatch(op)
        if result.ok:
            self._oplog.append(op)
        return result

    def _dispatch(self, op: Op) -> ApplyResult:
        tag = getattr(type(op), "OP", "")
        if isinstance(op, NewSketch):
            return self._new_sketch(op)
        if isinstance(op, AddRectangle):
            return self._add_rectangle(op)
        if isinstance(op, AddCircle):
            return self._add_circle(op)
        if isinstance(op, Extrude):
            return self._extrude(op)
        if isinstance(op, Boolean):
            return self._boolean(op)
        # Everything else is refused with its precise reason.
        reason = REFUSED_OPS.get(tag, "op not supported by the onshape backend")
        return _err("unsupported-op",
                    "onshape backend cannot build '%s': %s" % (tag, reason), tag)

    # -- sketch ops (buffered locally; flushed as one feature on first use) --
    def _new_sketch(self, op: NewSketch) -> ApplyResult:
        if str(op.plane) not in PLANE_TO_ONSHAPE:
            return _err("bad-value",
                        "unknown sketch plane '%s' (supported: %s)"
                        % (op.plane, ", ".join(sorted(set(PLANE_TO_ONSHAPE)))))
        sid = self._new_id("sk")
        self.sketches[sid] = {"plane": str(op.plane), "entities": [],
                              "onshape_fid": None}
        return ApplyResult(True, [sid])

    def _add_rectangle(self, op: AddRectangle) -> ApplyResult:
        if op.sketch not in self.sketches:
            return _err("bad-ref", "unknown sketch '%s'" % op.sketch, op.sketch)
        if op.w <= 0 or op.h <= 0:
            return _err("bad-value", "rectangle w and h must be > 0")
        if self.sketches[op.sketch]["onshape_fid"] is not None:
            return _err("locked-sketch",
                        "sketch '%s' was already flushed to Onshape and cannot take "
                        "more entities" % op.sketch, op.sketch)
        eid = self._new_id("e")
        self.sketches[op.sketch]["entities"].append(
            {"kind": "rectangle", "id": eid,
             "params": {"x": op.x, "y": op.y, "w": op.w, "h": op.h}})
        return ApplyResult(True, [eid])

    def _add_circle(self, op: AddCircle) -> ApplyResult:
        if op.sketch not in self.sketches:
            return _err("bad-ref", "unknown sketch '%s'" % op.sketch, op.sketch)
        if op.r <= 0:
            return _err("bad-value", "circle radius must be > 0 (got %g)" % op.r)
        if self.sketches[op.sketch]["onshape_fid"] is not None:
            return _err("locked-sketch",
                        "sketch '%s' was already flushed to Onshape and cannot take "
                        "more entities" % op.sketch, op.sketch)
        eid = self._new_id("e")
        self.sketches[op.sketch]["entities"].append(
            {"kind": "circle", "id": eid,
             "params": {"cx": op.cx, "cy": op.cy, "r": op.r}})
        return ApplyResult(True, [eid])

    def _flush_sketch(self, sid: str) -> Tuple[Optional[str], Optional[ApplyResult]]:
        """POST the buffered sketch as one newSketch feature; return its Onshape
        featureId. Idempotent -- a second call returns the cached id."""
        sk = self.sketches[sid]
        if sk["onshape_fid"] is not None:
            return sk["onshape_fid"], None
        if not sk["entities"]:
            return None, _err("empty-sketch",
                              "sketch '%s' has no profile to extrude" % sid, sid)
        entities: List[dict] = []
        for ent in sk["entities"]:
            p = ent["params"]
            if ent["kind"] == "rectangle":
                entities.extend(_rectangle_entities(ent["id"], p["x"], p["y"],
                                                    p["w"], p["h"]))
            elif ent["kind"] == "circle":
                entities.append(_circle_entity(ent["id"], p["cx"], p["cy"], p["r"]))
        feature = build_sketch_feature("Sketch %s" % sid[2:], sk["plane"], entities)
        fid, err = self._post_feature(feature)
        if err is not None:
            return None, err
        sk["onshape_fid"] = fid
        self.features.append({"type": "sketch", "id": sid, "onshape_fid": fid})
        return fid, None

    def _post_feature(self, feature: dict) -> Tuple[Optional[str], Optional[ApplyResult]]:
        """Ensure the document, POST one feature, return the minted featureId."""
        ensured = self._ensure_document()
        if ensured is not None:
            return None, ensured
        try:
            resp = self.client.add_feature(self.doc, feature)
        except OnshapeApiError as exc:
            return None, _err("api-error", "feature POST failed: %s" % exc)
        self._posted.append(feature)
        fid = ""
        if isinstance(resp, dict):
            fid = ((resp.get("feature") or {}).get("featureId")
                   or resp.get("featureId") or "")
        return fid, None

    # -- extrude -----------------------------------------------------------
    def _extrude(self, op: Extrude) -> ApplyResult:
        if op.sketch not in self.sketches:
            return _err("bad-ref", "unknown sketch '%s'" % op.sketch, op.sketch)
        if op.distance == 0:
            return _err("bad-value", "extrude distance must be non-zero")
        sketch_fid, err = self._flush_sketch(op.sketch)
        if err is not None:
            return err
        feature = build_extrude_feature("Extrude %d" % (self._n["f"] + 1),
                                        sketch_fid, float(op.distance))
        fid, err = self._post_feature(feature)
        if err is not None:
            return err
        local = self._new_id("f")
        self.features.append({"type": "extrude", "id": local, "sketch": op.sketch,
                              "onshape_fid": fid})
        self._onshape_fid[local] = fid
        self.solid_present = True
        return ApplyResult(True, [local])

    # -- boolean -----------------------------------------------------------
    def _solid_feature_ids(self) -> List[str]:
        return [f["id"] for f in self.features
                if f["type"] in ("extrude", "boolean")]

    def _boolean(self, op: Boolean) -> ApplyResult:
        if op.kind not in BOOLEAN_OP:
            return _err("bad-value", "unknown boolean kind '%s'" % op.kind)
        solids = self._solid_feature_ids()
        if len(solids) < 2:
            return _err("no-solid", "boolean requires two solids")
        target = op.target or solids[-2]
        tool = op.tool or solids[-1]
        if target not in self._onshape_fid:
            return _err("bad-ref", "unknown boolean target '%s'" % op.target,
                        op.target)
        if tool not in self._onshape_fid:
            return _err("bad-ref", "unknown boolean tool '%s'" % op.tool, op.tool)
        if target == tool:
            return _err("bad-ref", "boolean target and tool are the same body")
        feature = build_boolean_feature("Boolean %d" % (self._n["f"] + 1), op.kind,
                                        self._onshape_fid[target],
                                        self._onshape_fid[tool])
        fid, err = self._post_feature(feature)
        if err is not None:
            return err
        local = self._new_id("f")
        self.features.append({"type": "boolean", "id": local, "kind": op.kind,
                              "onshape_fid": fid})
        self._onshape_fid[local] = fid
        return ApplyResult(True, [local])

    # -- set_param (edit replay) -------------------------------------------
    def _set_param(self, op: SetParam) -> ApplyResult:
        """Block-and-correct replay onto a fresh backend, mirroring the other
        backends. Because a replay re-POSTs to a NEW scratch document, this is only
        attempted when a client can create documents; the edit itself is validated
        against the op log first, so a bad target/param is refused without touching
        Onshape or ``self``."""
        new_log, err = edit_oplog(self._oplog, op)
        if err is not None:
            return _err(*err)
        trial = type(self)(credentials=self.creds, client=self.client,
                           scratch_name=self.scratch_name + "-edit",
                           keep_document=self.keep_document)
        for logged in new_log:
            r = trial.apply(logged)
            if not r.ok:
                trial.close()
                return ApplyResult(False, [], r.diagnostics)
        # Adopt the replayed state; drop our now-superseded scratch document.
        self.close()
        self.__dict__.update(trial.__dict__)
        return ApplyResult(True, [])

    def regenerate(self) -> List[Diagnostic]:
        return []  # Onshape regenerates server-side on every feature POST.

    # -- queries (the ORACLE reads) ----------------------------------------
    def query(self, q: str) -> dict:
        """Read-only queries. The geometric ones (measure/metrics/bbox) go through
        the oracle -- the SYNCHRONOUS structured read of the committed workspace --
        so they reflect Onshape's own kernel, not a local guess."""
        if q == "summary":
            return {"sketch_count": len(self.sketches),
                    "feature_count": len(self.features),
                    "solid_present": self.solid_present,
                    "document": self._doc_dict()}
        if q == "mapping":
            return {"supported": list(SUPPORTED_OPS), "refused": dict(REFUSED_OPS),
                    "feature_json": dict(FEATURE_DOC),
                    "posted_features": len(self._posted)}
        if q == "document":
            return self._doc_dict()
        if q == "features":
            feats = self._live_features()
            return {"features": feats, "count": len(feats)}
        if self.doc is None:
            return {}
        if q in ("measure", "metrics"):
            try:
                mp = self.client.mass_properties(self.doc)
                bb = self.client.bounding_box(self.doc)
            except OnshapeApiError as exc:
                return {"oracle_error": str(exc)}
            out = mp.to_dict()
            out.update(bb.to_dict())
            return out
        if q == "bbox":
            try:
                return self.client.bounding_box(self.doc).to_dict()
            except OnshapeApiError as exc:
                return {"oracle_error": str(exc)}
        return {}

    def _doc_dict(self) -> dict:
        if self.doc is None:
            return {}
        return {"did": self.doc.did, "wid": self.doc.wid, "eid": self.doc.eid}

    def _live_features(self) -> List[dict]:
        if self.doc is None:
            return []
        try:
            return self.client.features(self.doc)
        except OnshapeApiError:
            return []

    def export(self, fmt: str):
        f = str(fmt).lower()
        if self.doc is None:
            raise OnshapeApiError(0, "export",
                                  "no scratch document; apply an op first")
        if f == "stl":
            payload = self.client.export_stl(self.doc)
            return payload.decode("utf-8", "replace") \
                if isinstance(payload, bytes) else payload
        raise ValueError("onshape backend exports 'stl' synchronously; 'step' is an "
                         "asynchronous translation (POST .../translations then poll) "
                         "and is not wired in this build")

    def state_digest(self) -> str:
        """A deterministic digest of the OP STREAM + the feature JSON we POSTed.

        Honest scope: this is NOT Onshape's content hash. Onshape exposes a
        workspace microversion id, which is a VERSION handle, not a content hash of
        the geometry (two histories can share a solid) -- so, exactly as the CUA
        environment refuses to present the microversion as a digest, this digest is
        computed locally from the deterministic op log and the feature payloads,
        which is stable across identical replays.
        """
        model = {
            "oplog": [canonical_json(o) for o in self._oplog],
            "posted": [json.dumps(f, sort_keys=True, separators=(",", ":"))
                       for f in self._posted],
            "solid_present": self.solid_present,
        }
        blob = json.dumps(model, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    # -- teardown ----------------------------------------------------------
    def close(self) -> None:
        """Delete the scratch document (ours alone) unless ``keep_document``."""
        if self.doc is not None and not self.keep_document:
            try:
                self.client.delete_document(self.doc.did)
            except OnshapeApiError:
                pass
        self.doc = None

    def __enter__(self) -> "OnshapeBackend":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
