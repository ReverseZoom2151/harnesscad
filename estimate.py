"""Quantitative measurement + cost/BOM layer for HarnessCAD.

This module turns the *geometry* a backend produces into *numbers a buyer cares
about*: mass, stock size, material cost and a rough machining cost, then rolls
those up into a bill-of-materials (BOM) over an assembly. It is the money-facing
sibling of :mod:`contract` / :mod:`checks_dfm` — where those verify a part is
buildable, this one says how heavy and how expensive it is.

Mass properties come from the backend's read-only queries, and — exactly like
:class:`contract.ContractCheck` and :class:`checks_dfm.DFMCheck` — this layer
**degrades gracefully**:

  * ``query('metrics')`` -> ``{volume, mass, surface_area, bbox, center_of_mass}``
    is preferred (the richest source the real kernel is growing).
  * ``query('measure')`` -> ``{volume, bbox}`` is the fallback.
  * when neither answers (the dependency-free stub), the estimate is marked
    ``measured=False`` and the unknowable fields stay ``None`` rather than
    crashing or inventing numbers.

Units convention (matches the rest of the harness): lengths are millimetres,
so volumes are mm^3 and bounding boxes are mm. Material densities are g/cm^3
and costs are per-kg, so all conversions funnel through mm^3 -> cm^3 (/1000)
and g -> kg (/1000).

Everything here is stdlib-only, deterministic, and free of any wall clock.
"""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from verify import Diagnostic, Severity, VerifyReport


# --------------------------------------------------------------------------- #
# Material table
# --------------------------------------------------------------------------- #
@dataclass
class Material:
    """Physical + economic properties of one stock material.

    ``density`` is g/cm^3, ``cost_per_kg`` is currency/kg (raw stock), and
    ``machining_cost_per_cm3`` is the rough cost to *remove* one cm^3 of this
    material (harder metals cost more to cut). All three are advisory defaults a
    caller can override per shop / per process.
    """

    density: float                       # g/cm^3
    cost_per_kg: float                   # currency / kg of raw stock
    machining_cost_per_cm3: float = 0.5  # currency / cm^3 of material removed

    def to_dict(self) -> dict:
        return {
            "density": self.density,
            "cost_per_kg": self.cost_per_kg,
            "machining_cost_per_cm3": self.machining_cost_per_cm3,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Material":
        defaults = cls(density=1.0, cost_per_kg=0.0)
        return cls(
            density=float(d["density"]),
            cost_per_kg=float(d.get("cost_per_kg", defaults.cost_per_kg)),
            machining_cost_per_cm3=float(
                d.get("machining_cost_per_cm3", defaults.machining_cost_per_cm3)),
        )


# Sane, conservative defaults (g/cm^3, currency/kg, currency/cm^3-removed).
_DEFAULT_MATERIALS: Dict[str, Material] = {
    "aluminium": Material(2.70, 4.0, 0.8),
    "steel": Material(7.85, 1.2, 1.5),
    "stainless": Material(8.00, 5.0, 3.0),
    "titanium": Material(4.43, 35.0, 6.0),
    "brass": Material(8.50, 8.0, 1.2),
    "abs": Material(1.05, 3.5, 0.2),
    "pla": Material(1.24, 25.0, 0.15),
    "nylon": Material(1.14, 6.0, 0.3),
}

# Common spelling aliases so 'aluminum' / 'ss' / 'al' resolve.
_ALIASES = {
    "aluminum": "aluminium",
    "al": "aluminium",
    "alu": "aluminium",
    "ss": "stainless",
    "stainless steel": "stainless",
    "ti": "titanium",
    "mild steel": "steel",
    "carbon steel": "steel",
}


class MaterialTable:
    """A registry of :class:`Material` keyed by lowercased name.

    ``MaterialTable()`` ships with sane defaults (aluminium, steel, stainless,
    titanium, brass, ABS, PLA, nylon). Unknown names resolve through a small
    alias map, then fall back to ``default`` (aluminium) so an estimate never
    hard-crashes on a typo — the fallback is surfaced via :meth:`resolve`.
    """

    def __init__(self, materials: Optional[Dict[str, Material]] = None,
                 default: str = "aluminium") -> None:
        src = materials if materials is not None else _DEFAULT_MATERIALS
        self._m: Dict[str, Material] = {k.lower(): v for k, v in src.items()}
        self.default = default.lower()

    # -- lookup ------------------------------------------------------------- #
    def __contains__(self, name: str) -> bool:
        return self._normalise(name) in self._m

    def names(self) -> List[str]:
        return sorted(self._m)

    def _normalise(self, name: Optional[str]) -> str:
        key = (name or self.default).strip().lower()
        if key in self._m:
            return key
        if key in _ALIASES and _ALIASES[key] in self._m:
            return _ALIASES[key]
        return key

    def resolve(self, name: Optional[str]) -> Tuple[str, Material]:
        """Return ``(canonical_name, Material)`` for ``name``.

        Falls back to the table default when the name is unknown, so callers
        get a usable material *and* the canonical name actually used.
        """
        key = self._normalise(name)
        if key in self._m:
            return key, self._m[key]
        return self.default, self._m[self.default]

    def get(self, name: Optional[str]) -> Material:
        return self.resolve(name)[1]

    def add(self, name: str, material: Material) -> None:
        self._m[name.lower()] = material

    # -- serialisation ------------------------------------------------------ #
    def to_dict(self) -> dict:
        return {name: m.to_dict() for name, m in sorted(self._m.items())}

    @classmethod
    def from_dict(cls, d: dict, default: str = "aluminium") -> "MaterialTable":
        materials = {name: Material.from_dict(md) for name, md in d.items()}
        return cls(materials=materials, default=default)


# --------------------------------------------------------------------------- #
# Metrics resolution (graceful degradation)
# --------------------------------------------------------------------------- #
_METRIC_KEYS = ("volume", "mass", "surface_area", "bbox", "center_of_mass")


def resolve_metrics(source: Any) -> Optional[Dict[str, Any]]:
    """Extract a mass-properties dict from many possible shapes.

    Preference order for a backend: ``query('metrics')`` (rich) then
    ``query('measure')`` (volume+bbox). Also accepts a raw metrics dict, an
    object carrying a ``.backend`` / ``.metrics`` attribute, or one exposing
    ``volume`` / ``bbox`` attributes directly. Returns ``None`` when nothing
    usable is present, so callers can mark the estimate unmeasured rather than
    crash (mirrors ``contract._query`` conventions).
    """
    if source is None:
        return None

    # A raw metrics dict.
    if isinstance(source, dict):
        return _clean_metrics(source) or None

    # A backend (or session/candidate proxying one).
    query = getattr(source, "query", None)
    if callable(query):
        for key in ("metrics", "measure"):
            try:
                result = query(key)
            except Exception:  # noqa: BLE001 - unknown query must degrade
                result = None
            cleaned = _clean_metrics(result) if result else None
            if cleaned:
                return cleaned
        # backend answered nothing usable
        return None

    # An object that wraps a backend or precomputed metrics.
    for attr in ("metrics", "backend"):
        inner = getattr(source, attr, None)
        if inner is not None and inner is not source:
            got = resolve_metrics(inner)
            if got:
                return got

    # An object exposing measurement attributes directly.
    attrs = {k: getattr(source, k, None) for k in _METRIC_KEYS}
    cleaned = _clean_metrics(attrs)
    return cleaned or None


def _clean_metrics(raw: Any) -> Dict[str, Any]:
    """Keep only recognised, non-empty metric keys from a mapping."""
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, Any] = {}
    for k in _METRIC_KEYS:
        if k not in raw or raw[k] is None:
            continue
        v = raw[k]
        if k in ("bbox", "center_of_mass"):
            if isinstance(v, (list, tuple)) and len(v) >= 1:
                out[k] = [float(x) for x in v]
        else:
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                continue
    return out


# --------------------------------------------------------------------------- #
# Per-part estimate
# --------------------------------------------------------------------------- #
MM3_PER_CM3 = 1000.0   # 1 cm^3 = 1000 mm^3
G_PER_KG = 1000.0


@dataclass
class PartEstimate:
    """Mass + cost estimate for a single part.

    ``measured`` is ``True`` only when real mass properties were available; when
    a backend answers no metrics query it is ``False`` and the money fields are
    ``None``. ``volume_estimated`` flags the weaker case where only a bbox was
    available and the solid volume was approximated by the bounding box.
    """

    material: str
    mass: Optional[float] = None            # grams
    volume: Optional[float] = None          # mm^3 (part solid)
    surface_area: Optional[float] = None    # mm^2
    bbox: Optional[List[float]] = None      # [x, y, z] mm
    stock_size: Optional[List[float]] = None  # [x, y, z] mm (bbox + allowance)
    material_cost: Optional[float] = None   # currency (raw stock block)
    rough_machining_cost: Optional[float] = None  # currency (removal)
    measured: bool = True
    volume_estimated: bool = False

    @property
    def mass_kg(self) -> Optional[float]:
        return None if self.mass is None else self.mass / G_PER_KG

    @property
    def total_cost(self) -> Optional[float]:
        parts = [c for c in (self.material_cost, self.rough_machining_cost)
                 if c is not None]
        return sum(parts) if parts else None

    def to_dict(self) -> dict:
        return {
            "material": self.material,
            "mass": self.mass,
            "mass_kg": self.mass_kg,
            "volume": self.volume,
            "surface_area": self.surface_area,
            "bbox": self.bbox,
            "stock_size": self.stock_size,
            "material_cost": self.material_cost,
            "rough_machining_cost": self.rough_machining_cost,
            "total_cost": self.total_cost,
            "measured": self.measured,
            "volume_estimated": self.volume_estimated,
        }


def estimate_part(
    backend_or_metrics: Any,
    material: str = "aluminium",
    *,
    table: Optional[MaterialTable] = None,
    machining_allowance: float = 2.0,
) -> PartEstimate:
    """Estimate mass + cost of one part from its mass properties x a material.

    ``backend_or_metrics`` may be a backend (queried for ``'metrics'`` then
    ``'measure'``), a raw metrics dict, or an already-built :class:`PartEstimate`
    (returned as-is with its material rebound if asked). ``machining_allowance``
    is the per-face stock margin (mm) added around the bounding box to size the
    raw block.

    Degrades gracefully:
      * full metrics -> mass from volume, stock block from bbox, material +
        machining cost.
      * only bbox (no volume) -> volume approximated by the bounding box
        (``volume_estimated=True``); machining cost is therefore ~0 (block ~=
        part) but the material/stock cost is still meaningful.
      * only volume (no bbox) -> mass + a stock-less material cost from the part
        mass; no machining estimate.
      * nothing measurable -> ``measured=False`` and money fields ``None``.
    """
    if isinstance(backend_or_metrics, PartEstimate):
        return backend_or_metrics

    tbl = table or MaterialTable()
    mat_name, mat = tbl.resolve(material)

    metrics = resolve_metrics(backend_or_metrics)
    if not metrics:
        return PartEstimate(material=mat_name, measured=False)

    bbox = metrics.get("bbox")
    surface_area = metrics.get("surface_area")
    volume = metrics.get("volume")
    volume_estimated = False

    # Approximate solid volume from the bbox when the kernel gave us no volume.
    if (volume is None or volume <= 0.0) and _usable_bbox(bbox):
        volume = float(bbox[0]) * float(bbox[1]) * float(bbox[2])
        volume_estimated = True

    if volume is None or volume <= 0.0:
        # Truly nothing to weigh (e.g. a degenerate/empty model).
        return PartEstimate(
            material=mat_name, bbox=_norm_bbox(bbox),
            surface_area=surface_area, measured=True)

    # Mass: prefer computing from volume x density so it tracks the material.
    volume_cm3 = volume / MM3_PER_CM3
    mass_g = volume_cm3 * mat.density

    est = PartEstimate(
        material=mat_name,
        mass=mass_g,
        volume=volume,
        surface_area=surface_area,
        bbox=_norm_bbox(bbox),
        volume_estimated=volume_estimated,
        measured=True,
    )

    if _usable_bbox(bbox):
        stock = [float(d) + 2.0 * machining_allowance for d in bbox[:3]]
        est.stock_size = stock
        stock_volume_mm3 = stock[0] * stock[1] * stock[2]
        stock_volume_cm3 = stock_volume_mm3 / MM3_PER_CM3
        stock_mass_kg = (stock_volume_cm3 * mat.density) / G_PER_KG
        est.material_cost = stock_mass_kg * mat.cost_per_kg
        removed_cm3 = max(stock_volume_cm3 - volume_cm3, 0.0)
        est.rough_machining_cost = removed_cm3 * mat.machining_cost_per_cm3
    else:
        # No bbox -> price the part mass as raw stock; no machining estimate.
        est.material_cost = (mass_g / G_PER_KG) * mat.cost_per_kg
        est.rough_machining_cost = None

    return est


def _usable_bbox(bbox: Any) -> bool:
    return (isinstance(bbox, (list, tuple)) and len(bbox) >= 3
            and all(isinstance(d, (int, float)) and d > 0.0 for d in bbox[:3]))


def _norm_bbox(bbox: Any) -> Optional[List[float]]:
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 3:
        return [float(d) for d in bbox[:3]]
    return None


# --------------------------------------------------------------------------- #
# Bill of materials
# --------------------------------------------------------------------------- #
@dataclass
class BOMLine:
    """One line of a BOM: a part, its quantity, and per-unit + extended cost."""

    part: str
    qty: int
    material: str
    unit_mass: Optional[float] = None       # grams, per unit
    unit_cost: Optional[float] = None        # currency, per unit
    estimate: Optional[PartEstimate] = None  # the underlying per-unit estimate

    @property
    def total_mass(self) -> Optional[float]:
        return None if self.unit_mass is None else self.unit_mass * self.qty

    @property
    def total_cost(self) -> Optional[float]:
        return None if self.unit_cost is None else self.unit_cost * self.qty

    def to_dict(self) -> dict:
        return {
            "part": self.part,
            "qty": self.qty,
            "material": self.material,
            "unit_mass": self.unit_mass,
            "unit_cost": self.unit_cost,
            "total_mass": self.total_mass,
            "total_cost": self.total_cost,
        }


@dataclass
class BOM:
    """A bill of materials: line items + rolled-up totals."""

    lines: List[BOMLine] = field(default_factory=list)

    @property
    def total_mass(self) -> float:
        return sum(l.total_mass for l in self.lines if l.total_mass is not None)

    @property
    def total_cost(self) -> float:
        return sum(l.total_cost for l in self.lines if l.total_cost is not None)

    @property
    def part_count(self) -> int:
        """Total number of physical parts (sum of quantities)."""
        return sum(l.qty for l in self.lines)

    @property
    def line_count(self) -> int:
        return len(self.lines)

    @property
    def fully_measured(self) -> bool:
        """True iff every line yielded a real cost (no unmeasured parts)."""
        return all(l.unit_cost is not None for l in self.lines)

    def to_dict(self) -> dict:
        return {
            "lines": [l.to_dict() for l in self.lines],
            "totals": {
                "mass": self.total_mass,
                "cost": self.total_cost,
                "part_count": self.part_count,
                "line_count": self.line_count,
                "fully_measured": self.fully_measured,
            },
        }

    # -- renders ------------------------------------------------------------ #
    def to_csv(self) -> str:
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(["part", "qty", "material", "unit_mass_g",
                    "unit_cost", "total_mass_g", "total_cost"])
        for l in self.lines:
            w.writerow([
                l.part, l.qty, l.material,
                _csv_num(l.unit_mass), _csv_num(l.unit_cost),
                _csv_num(l.total_mass), _csv_num(l.total_cost),
            ])
        w.writerow(["TOTAL", self.part_count, "",
                    "", "", _csv_num(self.total_mass), _csv_num(self.total_cost)])
        return buf.getvalue()

    def to_markdown(self) -> str:
        header = ("| Part | Qty | Material | Unit mass (g) | Unit cost | "
                  "Total mass (g) | Total cost |")
        sep = "| --- | ---: | --- | ---: | ---: | ---: | ---: |"
        rows = [header, sep]
        for l in self.lines:
            rows.append(
                f"| {l.part} | {l.qty} | {l.material} | "
                f"{_md_num(l.unit_mass)} | {_md_num(l.unit_cost)} | "
                f"{_md_num(l.total_mass)} | {_md_num(l.total_cost)} |")
        rows.append(
            f"| **Total** | **{self.part_count}** | | | | "
            f"**{_md_num(self.total_mass)}** | **{_md_num(self.total_cost)}** |")
        return "\n".join(rows)


def _csv_num(v: Optional[float]) -> str:
    return "" if v is None else f"{v:.6g}"


def _md_num(v: Optional[float]) -> str:
    return "-" if v is None else f"{v:.4g}"


class BOMEstimator:
    """Walk an assembly (or a single part) into a costed :class:`BOM`.

    ``estimate(backend)`` first tries ``query('assembly')``; when present it is
    expected to describe a list of part instances (see :meth:`_assembly_parts`
    for the shapes accepted). When absent — the common single-part case, and the
    only thing the current backends expose — the whole backend is costed as one
    line item of quantity 1. Either way the result is a BOM with per-line and
    rolled-up mass + cost totals.
    """

    def __init__(self, table: Optional[MaterialTable] = None,
                 default_material: str = "aluminium",
                 *, machining_allowance: float = 2.0) -> None:
        self.table = table or MaterialTable()
        self.default_material = default_material
        self.machining_allowance = machining_allowance

    def estimate(self, backend: Any) -> BOM:
        parts = self._assembly_parts(backend)
        if parts is None:
            # No assembly query -> single part.
            line = self._line_for(
                name="part", material=self.default_material,
                qty=1, metrics_source=backend)
            return BOM(lines=[line])

        lines: List[BOMLine] = []
        for i, part in enumerate(parts):
            name, material, qty, src = self._describe_part(part, i)
            lines.append(self._line_for(name, material, qty, src))
        return BOM(lines=lines)

    # -- assembly extraction ------------------------------------------------ #
    def _assembly_parts(self, backend: Any) -> Optional[List[Any]]:
        """Return a list of part descriptors from ``query('assembly')`` or None.

        Accepts an assembly that is a bare list, or a dict keyed by any of
        ``parts`` / ``components`` / ``instances`` / ``items``.
        """
        query = getattr(backend, "query", None)
        if not callable(query):
            return None
        try:
            asm = query("assembly")
        except Exception:  # noqa: BLE001
            return None
        if not asm:
            return None
        if isinstance(asm, list):
            return asm
        if isinstance(asm, dict):
            for key in ("parts", "components", "instances", "items"):
                seq = asm.get(key)
                if isinstance(seq, list):
                    return seq
        return None

    def _describe_part(self, part: Any,
                       index: int) -> Tuple[str, str, int, Any]:
        """Resolve ``(name, material, qty, metrics_source)`` from a descriptor."""
        if isinstance(part, dict):
            name = str(part.get("name") or part.get("id")
                       or part.get("part") or f"part{index + 1}")
            material = part.get("material") or self.default_material
            qty = _as_int(part.get("qty") or part.get("count")
                          or part.get("quantity"), default=1)
            # Where do this part's metrics live?
            src: Any = None
            for key in ("metrics", "measure", "backend"):
                if part.get(key) is not None:
                    src = part[key]
                    break
            if src is None:
                # The descriptor may itself carry volume/bbox keys.
                src = part
            return name, str(material), qty, src
        # A bare backend / metrics object.
        name = str(getattr(part, "name", f"part{index + 1}"))
        material = str(getattr(part, "material", self.default_material))
        qty = _as_int(getattr(part, "qty", None), default=1)
        return name, material, qty, part

    def _line_for(self, name: str, material: str, qty: int,
                  metrics_source: Any) -> BOMLine:
        est = estimate_part(
            metrics_source, material=material, table=self.table,
            machining_allowance=self.machining_allowance)
        return BOMLine(
            part=name, qty=qty, material=est.material,
            unit_mass=est.mass, unit_cost=est.total_cost, estimate=est)


def _as_int(v: Any, default: int = 1) -> int:
    try:
        n = int(v)
        return n if n > 0 else default
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- #
# Budget verifier (advisory)
# --------------------------------------------------------------------------- #
@dataclass
class BudgetSpec:
    """A cost/mass budget for a part or assembly.

    ``max_mass`` is in grams, ``max_cost`` in currency. Either may be ``None``
    (unchecked). ``material`` is the assumed stock material when the source
    carries none.
    """

    max_mass: Optional[float] = None
    max_cost: Optional[float] = None
    material: str = "aluminium"

    def to_dict(self) -> dict:
        return {"max_mass": self.max_mass, "max_cost": self.max_cost,
                "material": self.material}

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "BudgetSpec":
        d = d or {}
        return cls(
            max_mass=_opt_float(d.get("max_mass")),
            max_cost=_opt_float(d.get("max_cost")),
            material=str(d.get("material", "aluminium")))


class BudgetCheck:
    """A :class:`verify.Verifier` asserting a part stays within mass/cost budget.

    Advisory by design (like :class:`checks_dfm.DFMCheck`): a blown budget is a
    WARNING, never an ERROR, so it can annotate but never fail a
    :class:`verify.VerifyReport`. When the backend exposes no mass properties the
    check INFO-skips rather than erroring.
    """

    name = "budget"

    def __init__(self, spec: BudgetSpec,
                 table: Optional[MaterialTable] = None) -> None:
        self.spec = spec
        self.table = table or MaterialTable()

    def check(self, backend, opdag=None) -> VerifyReport:
        diags: List[Diagnostic] = []
        if self.spec.max_mass is None and self.spec.max_cost is None:
            diags.append(Diagnostic(
                Severity.INFO, "budget-empty",
                "budget check skipped: no max_mass or max_cost set"))
            return VerifyReport(diags)

        est = estimate_part(backend, material=self.spec.material,
                            table=self.table)
        if not est.measured or est.mass is None:
            diags.append(Diagnostic(
                Severity.INFO, "budget-skipped",
                "budget check skipped: backend exposes no mass properties "
                "('metrics'/'measure')"))
            return VerifyReport(diags)

        if self.spec.max_mass is not None:
            if est.mass > self.spec.max_mass:
                diags.append(Diagnostic(
                    Severity.WARNING, "over-mass-budget",
                    f"mass {est.mass:.4g} g exceeds budget "
                    f"{self.spec.max_mass:.4g} g"))
            else:
                diags.append(Diagnostic(
                    Severity.INFO, "mass-within-budget",
                    f"mass {est.mass:.4g} g within budget "
                    f"{self.spec.max_mass:.4g} g"))

        if self.spec.max_cost is not None:
            cost = est.total_cost
            if cost is None:
                diags.append(Diagnostic(
                    Severity.INFO, "cost-unmeasurable",
                    "cost budget skipped: no bbox/volume to price stock"))
            elif cost > self.spec.max_cost:
                diags.append(Diagnostic(
                    Severity.WARNING, "over-cost-budget",
                    f"cost {cost:.4g} exceeds budget "
                    f"{self.spec.max_cost:.4g}"))
            else:
                diags.append(Diagnostic(
                    Severity.INFO, "cost-within-budget",
                    f"cost {cost:.4g} within budget "
                    f"{self.spec.max_cost:.4g}"))
        return VerifyReport(diags)


def _opt_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
