"""Bill of materials extracted from a SCAD object tree.

Reimplementation of SolidPython's ``bom_part`` / ``bill_of_materials``
(``solid/utils.py``).  A design built from Python functions can annotate each
part-producing function with its description, unit price and any extra columns;
walking the emitted object tree then yields a costed parts list -- the design
and its BOM can never drift apart, because the BOM *is* the design tree.

Differences from SolidPython, all in the direction of determinism:

  * the extra column names are passed to :func:`bill_of_materials` instead of
    living in a module-level global that accumulates across calls;
  * the table is rendered here (fixed-width, left-justified, deterministic)
    rather than depending on the third-party ``prettytable``, which SolidPython
    silently falls back away from;
  * parts are listed in the order they are first met in a depth-first pre-order
    walk, so the same tree always produces byte-identical output.

Works on ``programs.solidpy_scad_emit.ScadNode`` trees via the generic
``add_trait`` / ``get_trait`` mechanism; :func:`bom_part` is a decorator for
functions that return one.

Pure stdlib, deterministic.
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Sequence

from harnesscad.domain.programs.emit.openscad_emit import ScadNode

__all__ = [
    "BOM_TRAIT",
    "bom_part",
    "bom_traits",
    "bom_rows",
    "bill_of_materials",
    "table_string",
]

BOM_TRAIT = "BOM"


def bom_part(description: str = "", per_unit_price: Optional[float] = None,
             currency: str = "US$", **extra: Any) -> Callable:
    """Decorate a ScadNode-returning function so its output carries a BOM entry."""
    def wrap(func: Callable[..., ScadNode]) -> Callable[..., ScadNode]:
        name = description if description else func.__name__

        @wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> ScadNode:
            node = func(*args, **kwargs)
            if not isinstance(node, ScadNode):
                raise TypeError("a bom_part function must return a ScadNode")
            entry: Dict[str, Any] = {
                "name": name,
                "currency": currency,
                "unit_price": per_unit_price,
            }
            entry.update(extra)
            node.add_trait(BOM_TRAIT, entry)
            return node

        return wrapped

    return wrap


def bom_traits(root: ScadNode) -> List[Dict[str, Any]]:
    """Every BOM trait in the tree, in depth-first pre-order."""
    found: List[Dict[str, Any]] = []
    trait = root.get_trait(BOM_TRAIT)
    if trait:
        found.append(trait)
    for child in root.children:
        found.extend(bom_traits(child))
    return found


def bom_rows(root: ScadNode) -> List[Dict[str, Any]]:
    """Aggregate the tree's BOM traits into one row per distinct part name."""
    rows: List[Dict[str, Any]] = []
    by_name: Dict[str, Dict[str, Any]] = {}
    for trait in bom_traits(root):
        name = trait["name"]
        row = by_name.get(name)
        if row is None:
            row = dict(trait)
            row["count"] = 1
            by_name[name] = row
            rows.append(row)
        else:
            row["count"] += 1
    return rows


def _currency_str(value: float, currency: str) -> str:
    return "%s %.2f" % (currency, value)


def table_string(field_names: Sequence[str], rows: Sequence[Sequence[Any]],
                 csv: bool = False) -> str:
    """Render a table: tab-separated when ``csv``, else fixed-width columns."""
    cells = [[("" if c is None else str(c)) for c in row] for row in rows]
    if csv:
        lines = ["\t".join(field_names)]
        lines.extend("\t".join(row) for row in cells)
        return "\n".join(lines) + "\n"

    widths = [len(str(f)) for f in field_names]
    for row in cells:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def render(row: Sequence[str]) -> str:
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(row)) + " |"

    rule = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    lines = [rule, render([str(f) for f in field_names]), rule]
    lines.extend(render(row) for row in cells)
    lines.append(rule)
    return "\n".join(lines) + "\n"


def bill_of_materials(root: ScadNode, headers: Sequence[str] = (),
                      csv: bool = False) -> str:
    """A costed parts table for the tree rooted at ``root``."""
    field_names = ["Description", "Count", "Unit Price", "Total Price"]
    field_names += list(headers)

    rows: List[List[Any]] = []
    totals: Dict[str, float] = {}
    order: List[str] = []

    for entry in bom_rows(root):
        count = entry["count"]
        price = entry.get("unit_price")
        currency = entry.get("currency", "")
        if price is None:
            unit_price = total_price = ""
        else:
            total = price * count
            if currency not in totals:
                totals[currency] = 0.0
                order.append(currency)
            totals[currency] += total
            unit_price = _currency_str(price, currency)
            total_price = _currency_str(total, currency)
        row: List[Any] = [entry["name"], count, unit_price, total_price]
        for key in headers:
            row.append(entry.get(key, ""))
        rows.append(row)

    for currency in order:
        blank = [""] * len(field_names)
        blank[0] = "Total Cost, %s" % currency
        blank[3] = _currency_str(totals[currency], currency)
        rows.append(blank)

    return table_string(field_names, rows, csv=csv)
