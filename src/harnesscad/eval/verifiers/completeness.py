"""Model-completeness gate — a standalone verifier for *intrinsic metadata
coverage*.

The blueprint's plural verifier (sec.21) asks not only "is the geometry valid?"
and "does it match the brief?" but also "is the model *complete enough to hand
off*?". A B-rep can be watertight and satisfy every countable ask in the prompt
and still be unmanufacturable-on-paper because it carries no material, its holes
have no tolerance/thread callout, its critical dimensions have no tolerance, and
the part itself has no name or units. Those are gaps in the model's *own*
metadata, independent of any brief.

This is deliberately **distinct from** :class:`checks_requirements.RequirementsCheck`:

  * ``RequirementsCheck`` is *prompt-conformance* — it compares the built model to
    a typed :class:`spec.formalize.RequirementSet` extracted from the brief ("the
    brief said 4 holes; are there 4?").
  * ``CompletenessCheck`` is *intrinsic metadata coverage* — it asserts the model
    carries the metadata a released part must have, regardless of what any brief
    asked for ("every body must name a material; every hole must carry a
    tolerance or thread spec").

A gap in *required* metadata is a hard ERROR ``missing-metadata`` (one per gap),
because an incomplete model must not silently pass the gate. A category the
backend simply cannot report (e.g. no per-dimension GD&T model) is an INFO skip
(``completeness-unmeasurable``), never an ERROR — so the same checklist runs
against the dependency-free stub and a metadata-aware kernel alike.

What is inspectable today: ``query('summary')`` / ``query('metrics')`` (merged
into one part-metadata view — metrics wins on conflict) plus the op stream (Hole
ops). A metadata-aware backend adds ``material`` / ``name`` / ``units`` part
keys, a ``bodies`` list (each with an optional ``material``), a ``holes`` list
(each with ``tolerance`` / ``thread``), and a ``critical_dimensions`` /
``dimensions`` list (each with a ``tolerance``); the checklist reads those the
moment they appear.

Standalone by design, exactly like :class:`checks_dfm.DFMCheck`: NOT wired into
:func:`verify.default_verifiers` (that would be a circular import, and this is an
opt-in gate). A caller adds it explicitly via :func:`with_completeness`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from harnesscad.eval.verifiers.verify import Diagnostic, Severity, VerifyReport


# --------------------------------------------------------------------------- #
# Rules (configurable required-metadata fields)
# --------------------------------------------------------------------------- #
@dataclass
class CompletenessRules:
    """Which intrinsic-metadata fields a released model MUST carry.

    Every flag toggles one required-field family. A ``True`` flag means "a gap in
    this metadata is an ERROR"; ``False`` disables the family entirely. Defaults
    require the full set (a part must name itself, its units, its material, and
    carry a tolerance/thread callout on every hole and every critical dimension).
    """

    require_part_name: bool = True          # the part must carry a name
    require_units: bool = True              # the part must declare its units
    require_material: bool = True           # every body must name a material
    require_hole_spec: bool = True          # every hole needs a tolerance/thread spec
    require_dimension_tolerance: bool = True  # every critical dim needs a tolerance

    def to_dict(self) -> dict:
        return {
            "require_part_name": self.require_part_name,
            "require_units": self.require_units,
            "require_material": self.require_material,
            "require_hole_spec": self.require_hole_spec,
            "require_dimension_tolerance": self.require_dimension_tolerance,
        }

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "CompletenessRules":
        d = d or {}
        defaults = cls()
        return cls(
            require_part_name=bool(d.get("require_part_name", defaults.require_part_name)),
            require_units=bool(d.get("require_units", defaults.require_units)),
            require_material=bool(d.get("require_material", defaults.require_material)),
            require_hole_spec=bool(d.get("require_hole_spec", defaults.require_hole_spec)),
            require_dimension_tolerance=bool(d.get(
                "require_dimension_tolerance", defaults.require_dimension_tolerance)),
        )


# Accepted key aliases in the merged part-metadata view.
_NAME_KEYS = ("name", "part_name")
_UNITS_KEYS = ("units", "unit")
_MATERIAL_KEYS = ("material",)
_HOLE_SPEC_KEYS = ("tolerance", "thread", "thread_spec")
_DIM_LIST_KEYS = ("critical_dimensions", "dimensions")

#: The keys whose PRESENCE (not truthiness) proves the backend has a metadata
#: surface to be incomplete about. A backend that reports none of them is not
#: an incomplete model; it is a model whose completeness is unknowable.
_METADATA_SURFACE_KEYS = (
    _NAME_KEYS + _UNITS_KEYS + _MATERIAL_KEYS + _DIM_LIST_KEYS
    + ("bodies", "holes", "metadata")
)


def _carries_metadata(meta: dict) -> bool:
    return any(k in meta for k in _METADATA_SURFACE_KEYS)


# --------------------------------------------------------------------------- #
# The verifier
# --------------------------------------------------------------------------- #
class CompletenessCheck:
    """A :class:`verify.Verifier` (``name='completeness'``) asserting intrinsic
    metadata coverage.

    ``check(backend, opdag)`` merges ``query('summary')`` and ``query('metrics')``
    into one part-metadata view (metrics wins), reads Hole ops off ``opdag``, and
    returns a :class:`verify.VerifyReport`:

      * ERROR ``missing-metadata``       — a required field is absent (one per gap;
        ``where`` names the gap, e.g. ``part.name``, ``body[b1].material``,
        ``hole[0]``, ``dimension[length]``).
      * INFO  ``completeness-unmeasurable`` — a required family cannot be
        evaluated (no body/solid to attach a material to, or no dimension list to
        read tolerances from); a real hook point, never an ERROR.
      * INFO  ``completeness-skipped``   — the backend/op-DAG exposed nothing to
        inspect at all.
    """

    name = "completeness"

    def __init__(self, checklist: Optional[CompletenessRules] = None) -> None:
        self.checklist = checklist or CompletenessRules()

    def check(self, backend, opdag=None) -> VerifyReport:
        summary = _query(backend, "summary") or {}
        metrics = _query(backend, "metrics") or {}
        meta = dict(summary)
        meta.update(metrics)  # a real metrics query overrides summary on conflict
        ops = _iter_ops(opdag)

        if not meta and not ops:
            return VerifyReport([_info(
                "completeness-skipped",
                "completeness checks skipped: backend exposed no "
                "'summary'/'metrics' and the op-DAG carried no ops to inspect.")])

        # A backend that exposes NO metadata surface at all cannot report a
        # name, units, a material or a hole callout for ANY part, correct or
        # broken. Against such a backend this checklist fired on 100% of inputs
        # -- 16 of 16 parts in the fleet audit, good and bad alike -- which makes
        # its precision equal to the corpus base rate BY CONSTRUCTION. A rule
        # that fires on every input carries zero information: it cannot separate
        # a correct part from a broken one, ever, and it inflated the fleet's
        # headline recall into an artifact.
        #
        # The rule already knew how to say this -- `completeness-unmeasurable`,
        # used for the GD&T dimension list -- and simply failed to apply it to
        # the other three families. So: no metadata surface -> UNMEASURABLE
        # (INFO), not a gap (ERROR). Nothing is weakened. Against a
        # metadata-aware backend, which is the only kind that can answer the
        # question, every ERROR fires exactly as before.
        if not _carries_metadata(meta):
            return VerifyReport([_info(
                "completeness-unmeasurable",
                "metadata coverage not evaluated: the backend exposes no "
                "metadata surface at all (no name/units/material/bodies/holes "
                "keys in 'summary' or 'metrics'), and the CISP op vocabulary "
                "cannot express those fields, so nothing here can distinguish a "
                "complete model from an incomplete one. Release readiness is a "
                "PDM-record question, not a build-gate question.",
                where="metadata")])

        diags: List[Diagnostic] = []
        self._check_part_fields(meta, diags)
        self._check_material(meta, diags)
        self._check_holes(meta, ops, diags)
        self._check_dimensions(meta, diags)
        return VerifyReport(diags)

    # -- part-level identity (name / units) --------------------------------- #
    def _check_part_fields(self, meta: dict, diags: List[Diagnostic]) -> None:
        r = self.checklist
        if r.require_part_name and not _first(meta, _NAME_KEYS):
            diags.append(_err(
                "missing-metadata",
                "part carries no name; a released model must be named.",
                where="part.name"))
        if r.require_units and not _first(meta, _UNITS_KEYS):
            diags.append(_err(
                "missing-metadata",
                "part declares no units; a released model must state its units "
                "(e.g. 'mm').",
                where="part.units"))

    # -- material per body -------------------------------------------------- #
    def _check_material(self, meta: dict, diags: List[Diagnostic]) -> None:
        r = self.checklist
        if not r.require_material:
            return
        part_material = _first(meta, _MATERIAL_KEYS)
        bodies = meta.get("bodies")
        if isinstance(bodies, list) and bodies:
            for i, b in enumerate(bodies):
                rec = b if isinstance(b, dict) else {}
                bid = str(rec.get("id", rec.get("name", f"body{i}")))
                material = _first(rec, _MATERIAL_KEYS) or part_material
                if not material:
                    diags.append(_err(
                        "missing-metadata",
                        f"body '{bid}' has no material assigned.",
                        where=f"body[{bid}].material"))
            return
        # No enumerable body list: fall back to the implicit single body.
        if meta.get("solid_present"):
            if not part_material:
                diags.append(_err(
                    "missing-metadata",
                    "the model's solid body has no material assigned.",
                    where="body.material"))
        else:
            diags.append(_info(
                "completeness-unmeasurable",
                "material coverage not evaluated: backend reports no bodies and "
                "no solid present to attach a material to.",
                where="material"))

    # -- hole tolerance / thread callout ------------------------------------ #
    def _check_holes(self, meta: dict, ops, diags: List[Diagnostic]) -> None:
        r = self.checklist
        if not r.require_hole_spec:
            return
        records: List[tuple] = []
        holes = meta.get("holes")
        if isinstance(holes, list) and holes:
            for i, h in enumerate(holes):
                records.append((f"hole[{i}]", h if isinstance(h, dict) else {}))
        else:
            # Derive from the op stream: a Hole op carries no tolerance/thread,
            # so an ops-only hole is genuinely a metadata gap.
            for idx, op in enumerate(ops):
                if type(op).__name__ == "Hole":
                    records.append((f"hole#{idx}", {
                        "diameter": getattr(op, "diameter", None),
                        "kind": getattr(op, "kind", None),
                    }))
        if not records:
            return  # no holes -> no hole-spec gaps
        for where, h in records:
            if not _first(h, _HOLE_SPEC_KEYS):
                diags.append(_err(
                    "missing-metadata",
                    f"hole '{where}' carries no tolerance or thread spec.",
                    where=where))

    # -- critical-dimension tolerance --------------------------------------- #
    def _check_dimensions(self, meta: dict, diags: List[Diagnostic]) -> None:
        r = self.checklist
        if not r.require_dimension_tolerance:
            return
        dims = None
        for key in _DIM_LIST_KEYS:
            v = meta.get(key)
            if isinstance(v, list):
                dims = v
                break
        if dims is None:
            diags.append(_info(
                "completeness-unmeasurable",
                "critical-dimension tolerance coverage not evaluated: backend "
                "reports no dimension list (needs a metadata / GD&T model).",
                where="critical-dimensions"))
            return
        for i, d in enumerate(dims):
            rec = d if isinstance(d, dict) else {}
            label = str(rec.get("label", rec.get("name", f"dim{i}")))
            if rec.get("tolerance") is None:
                diags.append(_err(
                    "missing-metadata",
                    f"critical dimension '{label}' carries no tolerance.",
                    where=f"dimension[{label}]"))


# --------------------------------------------------------------------------- #
# Wiring helper
# --------------------------------------------------------------------------- #
def with_completeness(verifiers,
                      checklist: Optional[CompletenessRules] = None) -> List:
    """Return a new verifier list with a :class:`CompletenessCheck` appended.

    Mirrors :func:`checks_dfm.with_dfm`::

        from harnesscad.eval.verifiers.verify import default_verifiers
        from harnesscad.eval.verifiers.completeness import with_completeness
        verifiers = with_completeness(default_verifiers())
    """
    return list(verifiers) + [CompletenessCheck(checklist)]


# --------------------------------------------------------------------------- #
# Helpers (mirror checks_dfm's graceful-degradation conventions)
# --------------------------------------------------------------------------- #
def _first(d: dict, keys) -> Optional[object]:
    """First truthy value among ``keys`` in mapping ``d`` (None if none)."""
    if not isinstance(d, dict):
        return None
    for k in keys:
        v = d.get(k)
        if v:
            return v
    return None


def _iter_ops(opdag) -> list:
    """Best-effort extraction of the op list from whatever ``opdag`` is
    (an OpDAG with ``ops()``, a plain list/tuple, or None). Never raises."""
    if opdag is None:
        return []
    ops_attr = getattr(opdag, "ops", None)
    if callable(ops_attr):
        try:
            return list(ops_attr())
        except Exception:  # noqa: BLE001 - degrade, never crash the verifier
            return []
    if isinstance(opdag, (list, tuple)):
        return list(opdag)
    return []


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
