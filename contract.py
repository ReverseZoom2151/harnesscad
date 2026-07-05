"""Contract — the machine-verifiable acceptance spec (the "Contractor" model).

Per HARNESS_BLUEPRINT.md sec.6, the highest-leverage idea is to *formalize a
brief into a machine-verifiable acceptance Contract* — required dimensions with
tolerances, volume/mass targets, feature/hole counts, "manifold & watertight",
"no self-intersections", plus optional named predicates — and then iterate with
self-validation until **all** contract checks pass.

This module owns two things and nothing else:

  * :class:`Contract` — a JSON-serialisable dataclass describing the acceptance
    spec. It can be authored by hand or *emitted by an LLM* as structured output
    (see :func:`contract_from_brief_schema` for the JSON schema to ask for).
  * :class:`ContractCheck` — a :class:`verify.Verifier` (``name='contract'``)
    that reads the backend's read-only queries (``'summary'``, ``'validity'``,
    ``'measure'``) and emits an ERROR :class:`verify.Diagnostic` for every unmet
    requirement. It **degrades gracefully**: if the backend does not answer a
    given query (e.g. the stub has no ``'measure'``), the dependent checks are
    skipped with an INFO diagnostic rather than failing.

``ContractCheck`` is a standalone verifier a caller adds explicitly (it is *not*
part of ``verify.default_verifiers()``), so nothing else in the harness changes.

Geometry is judged by numbers, never strings (blueprint sec.6): bounding box,
volume, mass, feature counts and topological validity — the queries the backends
already expose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from verify import Diagnostic, Severity, VerifyReport


# --------------------------------------------------------------------------- #
# Tolerances / sub-specs
# --------------------------------------------------------------------------- #
@dataclass
class Tolerance:
    """A target value with a symmetric +/- tolerance band."""

    target: float
    tol: float = 0.0

    def contains(self, value: float) -> bool:
        return abs(float(value) - self.target) <= self.tol + 1e-9

    def to_dict(self) -> dict:
        return {"target": self.target, "tol": self.tol}

    @classmethod
    def from_dict(cls, d: dict) -> "Tolerance":
        return cls(target=float(d["target"]), tol=float(d.get("tol", 0.0)))


@dataclass
class MassSpec:
    """A target mass with tolerance. Backends expose *volume*, so a density
    (mass per unit volume, in consistent units) converts volume -> mass."""

    target: float
    tol: float = 0.0
    density: float = 1.0

    def contains(self, volume: float) -> bool:
        mass = float(volume) * self.density
        return abs(mass - self.target) <= self.tol + 1e-9

    def mass_of(self, volume: float) -> float:
        return float(volume) * self.density

    def to_dict(self) -> dict:
        return {"target": self.target, "tol": self.tol, "density": self.density}

    @classmethod
    def from_dict(cls, d: dict) -> "MassSpec":
        return cls(target=float(d["target"]), tol=float(d.get("tol", 0.0)),
                   density=float(d.get("density", 1.0)))


# In-memory registry of named predicates. Predicates are *not* serialised into
# to_dict()/from_dict() (a Contract only carries the *names* of the predicates
# it requires); the runtime supplies the callables via this registry so the JSON
# form stays portable. A predicate is ``fn(backend, opdag) -> bool``.
_PREDICATE_REGISTRY: Dict[str, Callable] = {}


def register_predicate(name: str, fn: Callable) -> None:
    """Register a named predicate ``fn(backend, opdag) -> bool`` for contracts
    that require it by name."""
    _PREDICATE_REGISTRY[name] = fn


def get_predicate(name: str) -> Optional[Callable]:
    return _PREDICATE_REGISTRY.get(name)


# --------------------------------------------------------------------------- #
# The Contract
# --------------------------------------------------------------------------- #
@dataclass
class Contract:
    """A machine-verifiable acceptance spec for a part.

    Every field is optional: a contract asserts only what it names, and
    :class:`ContractCheck` skips (with INFO) any requirement the backend cannot
    answer. All fields are JSON-serialisable via :meth:`to_dict`/:meth:`from_dict`
    so a contract can be authored by hand or emitted by an LLM.
    """

    name: str = ""
    description: str = ""

    # Required bounding-box dimensions, keyed by axis ("x"/"y"/"z"), each a
    # Tolerance (target +/- tol). Checked against query('measure')['bbox'].
    bbox: Dict[str, Tolerance] = field(default_factory=dict)

    # Target volume / mass (from query('measure')['volume']).
    volume: Optional[Tolerance] = None
    mass: Optional[MassSpec] = None

    # Feature / hole counts (from query('summary')['feature_count']).
    min_features: Optional[int] = None      # feature_count >= min_features
    feature_count: Optional[int] = None     # feature_count == feature_count
    hole_count: Optional[int] = None        # no backend query yet -> INFO skip

    # Topology flags (from query('validity')).
    require_manifold: bool = False          # manifold AND watertight
    no_self_intersections: bool = False     # is_valid B-rep

    # Names of registered predicates that must return True (see register_predicate).
    predicates: List[str] = field(default_factory=list)

    # ---- serialisation ---------------------------------------------------- #
    def to_dict(self) -> dict:
        d: dict = {"name": self.name, "description": self.description}
        if self.bbox:
            d["bbox"] = {axis: t.to_dict() for axis, t in self.bbox.items()}
        if self.volume is not None:
            d["volume"] = self.volume.to_dict()
        if self.mass is not None:
            d["mass"] = self.mass.to_dict()
        if self.min_features is not None:
            d["min_features"] = self.min_features
        if self.feature_count is not None:
            d["feature_count"] = self.feature_count
        if self.hole_count is not None:
            d["hole_count"] = self.hole_count
        if self.require_manifold:
            d["require_manifold"] = True
        if self.no_self_intersections:
            d["no_self_intersections"] = True
        if self.predicates:
            d["predicates"] = list(self.predicates)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Contract":
        bbox = {
            axis: Tolerance.from_dict(t)
            for axis, t in (d.get("bbox") or {}).items()
        }
        return cls(
            name=d.get("name", ""),
            description=d.get("description", ""),
            bbox=bbox,
            volume=Tolerance.from_dict(d["volume"]) if d.get("volume") else None,
            mass=MassSpec.from_dict(d["mass"]) if d.get("mass") else None,
            min_features=d.get("min_features"),
            feature_count=d.get("feature_count"),
            hole_count=d.get("hole_count"),
            require_manifold=bool(d.get("require_manifold", False)),
            no_self_intersections=bool(d.get("no_self_intersections", False)),
            predicates=list(d.get("predicates", [])),
        )


# --------------------------------------------------------------------------- #
# The verifier
# --------------------------------------------------------------------------- #
_AXES = ("x", "y", "z")


class ContractCheck:
    """A :class:`verify.Verifier` that checks a backend against a Contract.

    ``check(backend, opdag)`` reads ``query('summary')``, ``query('validity')``
    and ``query('measure')`` and returns a :class:`verify.VerifyReport` whose
    ``ok`` is True iff every named requirement is met. Requirements whose backing
    query is unavailable are reported as INFO (skipped), never ERROR — so the
    same contract can be run against the dependency-free stub (no ``'measure'``/
    ``'validity'``) and the real CadQuery backend.
    """

    name = "contract"

    def __init__(self, contract: Contract) -> None:
        self.contract = contract

    def check(self, backend, opdag) -> VerifyReport:
        c = self.contract
        diags: List[Diagnostic] = []

        summary = _query(backend, "summary")
        validity = _query(backend, "validity")
        measure = _query(backend, "measure")

        self._check_dimensions(c, measure, diags)
        self._check_volume_mass(c, measure, diags)
        self._check_feature_counts(c, summary, diags)
        self._check_topology(c, validity, diags)
        self._check_predicates(c, backend, opdag, diags)

        return VerifyReport(diags)

    # -- dimensions --------------------------------------------------------- #
    def _check_dimensions(self, c: Contract, measure: Optional[dict],
                          diags: List[Diagnostic]) -> None:
        if not c.bbox:
            return
        if not measure:
            diags.append(_info("dim-skipped",
                               "bbox requirements skipped: backend has no "
                               "'measure' query"))
            return
        bbox = measure.get("bbox")
        if not bbox:
            diags.append(_info("dim-skipped",
                               "bbox requirements skipped: 'measure' returned "
                               "no bbox"))
            return
        dims = {axis: bbox[i] for i, axis in enumerate(_AXES) if i < len(bbox)}
        for axis, tol in c.bbox.items():
            actual = dims.get(axis)
            if actual is None:
                diags.append(_err("dim-missing",
                                  f"bbox axis '{axis}' not reported by backend",
                                  axis))
                continue
            if not tol.contains(actual):
                diags.append(_err(
                    "dim-out-of-tol",
                    f"bbox {axis}={actual:.4g} out of tolerance "
                    f"{tol.target:g} +/- {tol.tol:g}", axis))

    # -- volume / mass ------------------------------------------------------ #
    def _check_volume_mass(self, c: Contract, measure: Optional[dict],
                           diags: List[Diagnostic]) -> None:
        if c.volume is None and c.mass is None:
            return
        if not measure:
            diags.append(_info("measure-skipped",
                               "volume/mass requirements skipped: backend has "
                               "no 'measure' query"))
            return
        volume = measure.get("volume")
        if volume is None:
            diags.append(_info("measure-skipped",
                               "volume/mass requirements skipped: 'measure' "
                               "returned no volume"))
            return
        if c.volume is not None and not c.volume.contains(volume):
            diags.append(_err(
                "volume-out-of-tol",
                f"volume={float(volume):.6g} out of tolerance "
                f"{c.volume.target:g} +/- {c.volume.tol:g}"))
        if c.mass is not None and not c.mass.contains(volume):
            diags.append(_err(
                "mass-out-of-tol",
                f"mass={c.mass.mass_of(volume):.6g} "
                f"(volume {float(volume):.6g} x density {c.mass.density:g}) "
                f"out of tolerance {c.mass.target:g} +/- {c.mass.tol:g}"))

    # -- feature / hole counts --------------------------------------------- #
    def _check_feature_counts(self, c: Contract, summary: Optional[dict],
                              diags: List[Diagnostic]) -> None:
        wants_features = (c.min_features is not None
                          or c.feature_count is not None)
        if wants_features:
            if not summary:
                diags.append(_info("count-skipped",
                                   "feature-count requirements skipped: backend "
                                   "has no 'summary' query"))
            else:
                actual = summary.get("feature_count")
                if actual is None:
                    diags.append(_info("count-skipped",
                                       "feature-count skipped: 'summary' has no "
                                       "feature_count"))
                else:
                    if (c.min_features is not None
                            and actual < c.min_features):
                        diags.append(_err(
                            "too-few-features",
                            f"feature_count={actual} < required "
                            f"minimum {c.min_features}"))
                    if (c.feature_count is not None
                            and actual != c.feature_count):
                        diags.append(_err(
                            "wrong-feature-count",
                            f"feature_count={actual} != required "
                            f"{c.feature_count}"))
        if c.hole_count is not None:
            # No backend exposes a hole count yet -> degrade gracefully.
            diags.append(_info("hole-count-skipped",
                               "hole-count requirement skipped: no backend "
                               "query for hole count"))

    # -- topology ----------------------------------------------------------- #
    def _check_topology(self, c: Contract, validity: Optional[dict],
                        diags: List[Diagnostic]) -> None:
        if not (c.require_manifold or c.no_self_intersections):
            return
        if not validity:
            diags.append(_info("topology-skipped",
                               "manifold/self-intersection requirements skipped: "
                               "backend has no 'validity' query"))
            return
        if c.require_manifold:
            if not validity.get("manifold", False):
                diags.append(_err("not-manifold",
                                  "part is not manifold"))
            if not validity.get("watertight", False):
                diags.append(_err("not-watertight",
                                  "part is not watertight"))
        if c.no_self_intersections and not validity.get("is_valid", False):
            diags.append(_err("self-intersections",
                              "part has an invalid B-rep (possible "
                              "self-intersections)"))

    # -- predicates --------------------------------------------------------- #
    def _check_predicates(self, c: Contract, backend, opdag,
                          diags: List[Diagnostic]) -> None:
        for pname in c.predicates:
            fn = get_predicate(pname)
            if fn is None:
                diags.append(_info("predicate-skipped",
                                   f"predicate '{pname}' skipped: not registered",
                                   pname))
                continue
            try:
                passed = bool(fn(backend, opdag))
            except Exception as exc:  # noqa: BLE001 - a predicate must not crash
                diags.append(_err("predicate-error",
                                  f"predicate '{pname}' raised: {exc}", pname))
                continue
            if not passed:
                diags.append(_err("predicate-failed",
                                  f"predicate '{pname}' not satisfied", pname))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _query(backend, q: str) -> Optional[dict]:
    """Read a backend query, returning None when the backend does not answer it
    (backends return {} for unknown queries) so callers can INFO-skip."""
    try:
        result = backend.query(q)
    except Exception:  # noqa: BLE001 - an unsupported query must degrade, not crash
        return None
    return result or None


def _err(code: str, msg: str, where: Optional[str] = None) -> Diagnostic:
    return Diagnostic(Severity.ERROR, code, msg, where)


def _info(code: str, msg: str, where: Optional[str] = None) -> Diagnostic:
    return Diagnostic(Severity.INFO, code, msg, where)


# --------------------------------------------------------------------------- #
# JSON schema (for LLM structured output)
# --------------------------------------------------------------------------- #
def contract_from_brief_schema() -> dict:
    """Return the JSON schema for a :class:`Contract`.

    Hand this to a planner/LLM so it can emit a Contract as *structured output*
    from a natural-language brief. This function does **not** call an LLM — it
    only returns the schema the caller asks the model to fill.
    """
    tolerance = {
        "type": "object",
        "properties": {
            "target": {"type": "number",
                       "description": "nominal value"},
            "tol": {"type": "number", "minimum": 0, "default": 0.0,
                    "description": "symmetric +/- tolerance band"},
        },
        "required": ["target"],
        "additionalProperties": False,
    }
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "Contract",
        "description": "Machine-verifiable acceptance spec for a CAD part. "
                       "Assert only what the brief requires; omit the rest.",
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "short part name"},
            "description": {"type": "string",
                            "description": "human-readable intent"},
            "bbox": {
                "type": "object",
                "description": "required bounding-box dimensions per axis",
                "properties": {
                    "x": tolerance, "y": tolerance, "z": tolerance,
                },
                "additionalProperties": False,
            },
            "volume": {**tolerance,
                       "description": "target volume (+/- tol), model units^3"},
            "mass": {
                "type": "object",
                "description": "target mass; density converts backend volume "
                               "to mass (consistent units)",
                "properties": {
                    "target": {"type": "number"},
                    "tol": {"type": "number", "minimum": 0, "default": 0.0},
                    "density": {"type": "number", "exclusiveMinimum": 0,
                                "default": 1.0,
                                "description": "mass per unit volume"},
                },
                "required": ["target"],
                "additionalProperties": False,
            },
            "min_features": {"type": "integer", "minimum": 0,
                             "description": "feature_count must be >= this"},
            "feature_count": {"type": "integer", "minimum": 0,
                              "description": "feature_count must equal this"},
            "hole_count": {"type": "integer", "minimum": 0,
                           "description": "required number of holes"},
            "require_manifold": {
                "type": "boolean", "default": False,
                "description": "part must be manifold AND watertight"},
            "no_self_intersections": {
                "type": "boolean", "default": False,
                "description": "part must have a valid B-rep (no "
                               "self-intersections)"},
            "predicates": {
                "type": "array",
                "items": {"type": "string"},
                "description": "names of registered predicates that must hold",
            },
        },
        "required": ["name"],
        "additionalProperties": False,
    }
