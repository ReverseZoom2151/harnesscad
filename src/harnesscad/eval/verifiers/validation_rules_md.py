"""VALIDATION.md -- a declarative per-skill geometry-check language (freecad-ai).

Source: ``resources/cad_repos/freecad-ai-master`` (``freecad_ai/extensions/
skill_validator.py`` + ``skills/*/VALIDATION.md`` + the ``create-validation``
skill). freecad-ai lets every agent *skill* ship a human-reviewable
``VALIDATION.md`` beside its ``SKILL.md``: a typed parameter block plus
per-body checks (bbox, volume-formula, solid count, through-hole count,
section area, ...) with absolute or percentage tolerances and ``#### when``
variant blocks. After the skill runs, the checks are executed against the
real document, so a skill that *claims* to build an enclosure cannot pass
with a plausible-but-wrong solid.

This was missed by the original freecad-ai mining pass (which took the tool
catalogue, the expression engine, and the relative-value resolver; see
``docs/corpus/repo-ideas.md`` section 46). It is exactly the harness's
doctrine -- an unverified skill recipe paired with a deterministic,
declarative acceptance contract -- so it is ported here on harness terms:

* :func:`parse_validation_md` -- the exact grammar the source parses
  (``## Parameters`` typed defaults, ``### <BodyLabel>`` targets, ``### ...``
  headings with spaces = document level, ``#### when p == "v"`` conditions,
  ``(tolerance X)`` absolute / ``(tolerance X%)`` relative suffixes).
* :func:`safe_arithmetic` -- the same ast-based evaluator (numbers, names,
  ``+ - * / // % **``, unary minus, ``pi``, ``abs``/``min``/``max``/
  ``round``/``sqrt``); no ``eval``.
* :class:`BodyState` / :class:`DocumentState` -- a neutral measured-state
  model replacing the live FreeCAD document, so the checks compose with the
  harness's own measurement layer instead of a running CAD host.
* :func:`run_checks` / :func:`validate` -- rule execution with the source's
  comparison semantics (bbox always absolute-tolerance per axis,
  ``bbox_position`` defaulting to 0.5 mm, volume honouring the declared
  tolerance type, ``has_holes`` counting through-all pockets, ...).

Checks whose measured input is absent are reported ``passed=False`` with an
explicit "not measured" message rather than silently skipped: soundness,
not completeness.

Stdlib only, deterministic, absolute imports. ``--selfcheck`` runs the real
enclosure VALIDATION.md grammar end to end against a synthetic document.
"""

from __future__ import annotations

import argparse
import ast
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "ParamDef",
    "ValidationRule",
    "CheckResult",
    "BodyState",
    "DocumentState",
    "safe_arithmetic",
    "parse_validation_md",
    "resolve_params",
    "run_checks",
    "validate",
    "compute_pass_rate",
    "main",
]


# ---------------------------------------------------------------------------
# Safe arithmetic (port of the source's _ArithmeticEvaluator; no eval)
# ---------------------------------------------------------------------------

_ALLOWED_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
}

_ALLOWED_FUNCS = {
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "sqrt": math.sqrt,
}


def safe_arithmetic(expression: str,
                    variables: Optional[Mapping[str, float]] = None) -> float:
    """Evaluate a numeric expression over named variables without ``eval``.

    Supports numbers, variable names, ``pi``, the binary operators
    ``+ - * / // % **``, unary ``+``/``-``, parentheses, and the whitelisted
    calls ``abs``/``min``/``max``/``round``/``sqrt``. Anything else raises
    ``ValueError``.
    """
    names: Dict[str, float] = {"pi": math.pi}
    for key, value in (variables or {}).items():
        names[str(key)] = float(value)

    def visit(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError(f"non-numeric constant: {node.value!r}")
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id not in names:
                raise ValueError(f"unknown variable: {node.id}")
            return names[node.id]
        if isinstance(node, ast.UnaryOp):
            operand = visit(node.operand)
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.UAdd):
                return operand
            raise ValueError("unsupported unary operator")
        if isinstance(node, ast.BinOp):
            op = _ALLOWED_BINOPS.get(type(node.op))
            if op is None:
                raise ValueError("unsupported binary operator")
            return float(op(visit(node.left), visit(node.right)))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
                raise ValueError("unsupported function call")
            if node.keywords:
                raise ValueError("keyword arguments are not supported")
            args = [visit(a) for a in node.args]
            return float(_ALLOWED_FUNCS[node.func.id](*args))
        raise ValueError(f"unsupported expression node: {type(node).__name__}")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid expression: {expression!r}") from exc
    return visit(tree)


# ---------------------------------------------------------------------------
# Grammar model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParamDef:
    """One ``name: type [= default]`` line from the ``## Parameters`` block."""
    name: str
    type: str  # "float" | "int" | "str" | "bool"
    default: Optional[Any] = None


@dataclass(frozen=True)
class ValidationRule:
    """One ``- check: value`` line, bound to its target body and condition."""
    target: str                      # body label, or "_document"
    check: str
    expected: str                    # raw expression text (evaluated later)
    tolerance: float = 0.0
    tolerance_type: str = "absolute"  # "absolute" | "relative" (percent)
    condition: Optional[str] = None   # e.g. 'lid_type == "screw"'


@dataclass(frozen=True)
class CheckResult:
    target: str
    check: str
    passed: bool
    expected: Any
    actual: Any
    message: str
    skipped: bool = False

    def to_dict(self) -> dict:
        return {
            "target": self.target, "check": self.check, "passed": self.passed,
            "expected": self.expected, "actual": self.actual,
            "message": self.message, "skipped": self.skipped,
        }


_PARAM_RE = re.compile(r"^(\w+)\s*:\s*(float|int|str|bool)(?:\s*=\s*(.+))?$")
_CHECK_RE = re.compile(r"^-\s+(\w+)\s*:\s*(.+)$")
_TOL_ABS_RE = re.compile(r"\(tolerance\s+([\d.]+)\)\s*$")
_TOL_PCT_RE = re.compile(r"\(tolerance\s+([\d.]+)%\)\s*$")
_CONDITION_RE = re.compile(r"^####\s+when\s+(.+)$")


def _coerce_default(value_str: str, type_str: str) -> Any:
    value_str = value_str.strip()
    if type_str == "float":
        return float(value_str)
    if type_str == "int":
        return int(value_str)
    if type_str == "bool":
        return value_str.lower() in ("true", "1", "yes")
    return value_str


def parse_validation_md(content: str) -> Tuple[Dict[str, ParamDef], List[ValidationRule]]:
    """Parse VALIDATION.md text into ``(param_defs, rules)``.

    Grammar (exactly the source's): ``## Parameters`` starts the typed
    parameter block; ``## Checks`` starts the rules; a ``### Heading`` with a
    space in it is document level (target ``_document``), a single word is a
    body label; ``#### when <cond>`` opens a conditional block that a new
    ``###`` heading resets; each rule line is ``- check: value`` with an
    optional trailing ``(tolerance X)`` or ``(tolerance X%)``.
    """
    param_defs: Dict[str, ParamDef] = {}
    rules: List[ValidationRule] = []
    if not content or not content.strip():
        return param_defs, rules

    section = ""
    current_target = "_document"
    current_condition: Optional[str] = None

    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            heading = stripped[3:].strip().lower()
            if "parameter" in heading:
                section = "parameters"
            elif "check" in heading:
                section = "checks"
                current_target = "_document"
                current_condition = None
            continue

        if section == "parameters":
            m = _PARAM_RE.match(stripped)
            if m:
                name, typ, default_str = m.groups()
                default = _coerce_default(default_str, typ) if default_str else None
                param_defs[name] = ParamDef(name=name, type=typ, default=default)
            continue

        if section == "checks":
            if stripped.startswith("### "):
                heading_text = stripped[4:].strip()
                current_target = "_document" if " " in heading_text else heading_text
                current_condition = None
                continue
            cond_m = _CONDITION_RE.match(stripped)
            if cond_m:
                current_condition = cond_m.group(1).strip()
                continue
            check_m = _CHECK_RE.match(stripped)
            if check_m:
                check_name = check_m.group(1)
                value_part = check_m.group(2).strip()
                tolerance = 0.0
                tolerance_type = "absolute"
                tol_pct = _TOL_PCT_RE.search(value_part)
                if tol_pct:
                    tolerance = float(tol_pct.group(1))
                    tolerance_type = "relative"
                    value_part = value_part[: tol_pct.start()].strip()
                else:
                    tol_abs = _TOL_ABS_RE.search(value_part)
                    if tol_abs:
                        tolerance = float(tol_abs.group(1))
                        value_part = value_part[: tol_abs.start()].strip()
                rules.append(ValidationRule(
                    target=current_target, check=check_name, expected=value_part,
                    tolerance=tolerance, tolerance_type=tolerance_type,
                    condition=current_condition))
    return param_defs, rules


def resolve_params(params: Mapping[str, Any],
                   param_defs: Mapping[str, ParamDef]) -> Dict[str, Any]:
    """Fill missing parameters from declared defaults (stated values win)."""
    resolved: Dict[str, Any] = dict(params)
    for name, pdef in param_defs.items():
        if name not in resolved and pdef.default is not None:
            resolved[name] = pdef.default
    return resolved


# ---------------------------------------------------------------------------
# Measured-state model (replaces the live FreeCAD document)
# ---------------------------------------------------------------------------

@dataclass
class BodyState:
    """Measured facts about one named body.

    ``None`` means "not measured": the corresponding check then FAILS with an
    explicit message instead of passing silently (soundness over optimism).
    ``sections`` maps ``(axis, offset)`` -- axis in ``"XYZ"``, offset rounded
    to 6 dp -- to a measured cross-section area, mirroring the source's
    ``Shape.slice`` probe.
    """
    label: str
    bbox: Optional[Tuple[float, float, float]] = None          # X/Y/Z lengths
    z_range: Optional[Tuple[float, float]] = None               # (zmin, zmax)
    volume: Optional[float] = None
    solid_count: Optional[int] = None
    is_valid_solid: Optional[bool] = None
    through_hole_count: Optional[int] = None
    feature_labels: Tuple[str, ...] = ()
    child_count: Optional[int] = None
    sections: Dict[Tuple[str, float], float] = field(default_factory=dict)

    def section_area(self, axis: str, offset: float) -> Optional[float]:
        return self.sections.get((axis.upper(), round(float(offset), 6)))


@dataclass
class DocumentState:
    """The whole measured document: named bodies plus the body count."""
    bodies: Dict[str, BodyState] = field(default_factory=dict)

    def add(self, body: BodyState) -> "DocumentState":
        self.bodies[body.label] = body
        return self

    @property
    def total_bodies(self) -> int:
        return len(self.bodies)

    def get(self, label: str) -> Optional[BodyState]:
        return self.bodies.get(label)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def _check_condition(condition: str, params: Mapping[str, Any]) -> bool:
    """``p == "v"`` / ``p != "v"`` string comparison; unknown forms match nothing."""
    for op in ("!=", "=="):
        if op in condition:
            var_name, _, expected_val = condition.partition(op)
            expected = expected_val.strip().strip('"').strip("'")
            actual = str(params.get(var_name.strip(), ""))
            return (actual == expected) if op == "==" else (actual != expected)
    return False


def _numeric_params(params: Mapping[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in params.items():
        try:
            out[key] = float(value)  # bools become 0.0/1.0, matching the source
        except (TypeError, ValueError):
            pass
    return out


def _compare(actual: float, expected: float, tolerance: float,
             tolerance_type: str) -> bool:
    if tolerance_type == "relative":
        if expected == 0:
            return actual == 0
        return abs(actual - expected) / abs(expected) * 100.0 <= tolerance
    return abs(actual - expected) <= tolerance


def _fail(rule: ValidationRule, actual: Any, message: str) -> CheckResult:
    return CheckResult(target=rule.target, check=rule.check, passed=False,
                       expected=rule.expected, actual=actual, message=message)


def _unmeasured(rule: ValidationRule, what: str) -> CheckResult:
    return _fail(rule, None, f"{what} not measured for '{rule.target}'")


def _run_single(doc: DocumentState, num_params: Dict[str, float],
                rule: ValidationRule) -> CheckResult:
    check = rule.check

    if check == "total_bodies":
        expected = int(safe_arithmetic(rule.expected, num_params))
        actual = doc.total_bodies
        return CheckResult(rule.target, check, actual == expected, expected,
                           actual, f"Expected {expected} bodies, found {actual}")

    body = doc.get(rule.target)

    if check == "exists":
        passed = body is not None
        return CheckResult(rule.target, check, passed, True, passed,
                           f"Body '{rule.target}' {'found' if passed else 'not found'}")

    if body is None:
        return _fail(rule, None, f"Body '{rule.target}' not found")

    if check == "bbox":
        parts = [p.strip() for p in rule.expected.split(",")]
        if len(parts) != 3:
            return _fail(rule, None, "bbox requires 3 comma-separated values")
        if body.bbox is None:
            return _unmeasured(rule, "bbox")
        expected_dims = [safe_arithmetic(p, num_params) for p in parts]
        actual_dims = list(body.bbox)
        passed = all(_compare(a, e, rule.tolerance, "absolute")
                     for a, e in zip(actual_dims, expected_dims))
        return CheckResult(rule.target, check, passed, expected_dims, actual_dims,
                           f"BBox expected {expected_dims}, got {actual_dims}")

    if check == "bbox_position":
        parts = [p.strip() for p in rule.expected.split(",")]
        if len(parts) != 2:
            return _fail(rule, None, "bbox_position requires 2 values: zmin, zmax")
        if body.z_range is None:
            return _unmeasured(rule, "z_range")
        expected_zmin = safe_arithmetic(parts[0], num_params)
        expected_zmax = safe_arithmetic(parts[1], num_params)
        tol = rule.tolerance or 0.5
        zmin, zmax = body.z_range
        passed = (_compare(zmin, expected_zmin, tol, "absolute")
                  and _compare(zmax, expected_zmax, tol, "absolute"))
        return CheckResult(
            rule.target, check, passed,
            f"Z[{expected_zmin:.1f}, {expected_zmax:.1f}]",
            f"Z[{zmin:.1f}, {zmax:.1f}]",
            f"Position expected Z[{expected_zmin:.1f}, {expected_zmax:.1f}], "
            f"got Z[{zmin:.1f}, {zmax:.1f}]")

    if check == "volume":
        if body.volume is None:
            return _unmeasured(rule, "volume")
        expected = safe_arithmetic(rule.expected, num_params)
        passed = _compare(body.volume, expected, rule.tolerance, rule.tolerance_type)
        return CheckResult(rule.target, check, passed, expected, body.volume,
                           f"Volume expected {expected}, got {body.volume}")

    if check == "section_area":
        parts = [p.strip() for p in rule.expected.split(",")]
        if len(parts) != 3:
            return _fail(rule, None, "section_area requires 3 values: axis, offset, area")
        axis = parts[0].upper()
        if axis not in ("X", "Y", "Z"):
            return _fail(rule, None,
                         f"section_area axis must be X, Y, or Z, got '{parts[0]}'")
        offset = safe_arithmetic(parts[1], num_params)
        expected_area = safe_arithmetic(parts[2], num_params)
        actual_area = body.section_area(axis, offset)
        if actual_area is None:
            return _unmeasured(rule, f"section at {axis}={offset:g}")
        passed = _compare(actual_area, expected_area, rule.tolerance,
                          rule.tolerance_type)
        return CheckResult(rule.target, check, passed,
                           f"{expected_area:.1f}", f"{actual_area:.1f}",
                           f"Section area at {axis}={offset:.1f}: expected "
                           f"{expected_area:.1f}, got {actual_area:.1f}")

    if check == "solid_count":
        if body.solid_count is None:
            return _unmeasured(rule, "solid_count")
        expected = int(safe_arithmetic(rule.expected, num_params))
        passed = body.solid_count == expected
        return CheckResult(rule.target, check, passed, expected, body.solid_count,
                           f"Solid count expected {expected}, got {body.solid_count}")

    if check == "valid_solid":
        if body.is_valid_solid is None or body.solid_count is None:
            return _unmeasured(rule, "solid validity")
        passed = bool(body.is_valid_solid) and body.solid_count >= 1
        return CheckResult(rule.target, check, passed, True, passed,
                           f"Valid solid: isValid={body.is_valid_solid}, "
                           f"solids={body.solid_count}")

    if check == "has_holes":
        if body.through_hole_count is None:
            return _unmeasured(rule, "through-hole count")
        expected = int(safe_arithmetic(rule.expected, num_params))
        passed = body.through_hole_count == expected
        return CheckResult(rule.target, check, passed, expected,
                           body.through_hole_count,
                           f"Through-all holes expected {expected}, "
                           f"got {body.through_hole_count}")

    if check == "has_feature":
        wanted = rule.expected.strip().strip('"').strip("'")
        found = wanted in body.feature_labels
        return CheckResult(rule.target, check, found, wanted, found,
                           f"Feature '{wanted}' "
                           f"{'found' if found else 'not found'} in {rule.target}")

    if check == "min_children":
        if body.child_count is None:
            return _unmeasured(rule, "child count")
        expected = int(safe_arithmetic(rule.expected, num_params))
        passed = body.child_count >= expected
        return CheckResult(rule.target, check, passed, expected, body.child_count,
                           f"Min children expected {expected}, got {body.child_count}")

    return _fail(rule, None, f"Unknown check type: {check}")


def run_checks(doc: DocumentState, params: Mapping[str, Any],
               rules: Sequence[ValidationRule]) -> List[CheckResult]:
    """Execute rules against a measured document state."""
    results: List[CheckResult] = []
    num_params = _numeric_params(params)
    for rule in rules:
        if rule.condition is not None and not _check_condition(rule.condition, params):
            results.append(CheckResult(
                target=rule.target, check=rule.check, passed=True,
                expected=rule.expected, actual="skipped (condition not met)",
                message=f"Skipped: condition '{rule.condition}' not met",
                skipped=True))
            continue
        try:
            results.append(_run_single(doc, num_params, rule))
        except Exception as exc:  # expression errors and the like
            results.append(_fail(rule, None, f"Error: {exc}"))
    return results


def validate(doc: DocumentState, params: Mapping[str, Any],
             validation_content: str) -> List[CheckResult]:
    """Parse, resolve defaults, run. The one-call top-level API."""
    param_defs, rules = parse_validation_md(validation_content)
    return run_checks(doc, resolve_params(params, param_defs), rules)


def compute_pass_rate(results: Sequence[CheckResult]) -> float:
    if not results:
        return 1.0
    return sum(1 for r in results if r.passed) / len(results)


# ---------------------------------------------------------------------------
# Selfcheck
# ---------------------------------------------------------------------------

_ENCLOSURE_RULES = """\
# Validation Rules

## Parameters
L: float
W: float
H: float
T: float = 2
lid_type: str = screw

## Checks

### Body count
- total_bodies: 2

### EnclosureBase
- exists: true
- bbox: L, W, H (tolerance 0.5)
- bbox_position: 0, H (tolerance 0.5)
- solid_count: 1
- valid_solid: true

#### when lid_type == "screw"
- volume: L*W*H - (L-2*T)*(W-2*T)*(H-T) (tolerance 5%)
- min_children: 4

#### when lid_type == "press-fit"
- volume: L*W*H - (L-2*T)*(W-2*T)*(H-T) (tolerance 5%)

### EnclosureLid
- exists: true

#### when lid_type == "screw"
- has_holes: 4
"""


def _selfcheck() -> int:
    failures: List[str] = []

    def check(cond: bool, message: str) -> None:
        if not cond:
            failures.append(message)

    # Grammar.
    param_defs, rules = parse_validation_md(_ENCLOSURE_RULES)
    check(set(param_defs) == {"L", "W", "H", "T", "lid_type"}, "param names")
    check(param_defs["T"].default == 2.0, "float default")
    check(param_defs["lid_type"].default == "screw", "str default")
    check(rules[0].target == "_document" and rules[0].check == "total_bodies",
          "document-level heading routes to _document")
    vol_rules = [r for r in rules if r.check == "volume"]
    check(len(vol_rules) == 2 and vol_rules[0].tolerance_type == "relative"
          and vol_rules[0].tolerance == 5.0, "percent tolerance parsed")
    check(vol_rules[0].condition == 'lid_type == "screw"', "when condition bound")
    bbox_rule = next(r for r in rules if r.check == "bbox")
    check(bbox_rule.tolerance == 0.5 and bbox_rule.tolerance_type == "absolute",
          "absolute tolerance parsed")

    # Expression evaluator.
    check(abs(safe_arithmetic("2*pi") - 2 * math.pi) < 1e-12, "pi constant")
    check(safe_arithmetic("2**3") == 8.0, "power operator")
    check(safe_arithmetic("L - 2*T", {"L": 10, "T": 2}) == 6.0, "variables")
    try:
        safe_arithmetic("__import__('os')")
        check(False, "dunder call must be rejected")
    except ValueError:
        pass

    # A correct screw enclosure passes.
    L, W, H, T = 60.0, 40.0, 30.0, 2.0
    base_volume = L * W * H - (L - 2 * T) * (W - 2 * T) * (H - T)
    doc = DocumentState()
    doc.add(BodyState(label="EnclosureBase", bbox=(L, W, H), z_range=(0.0, H),
                      volume=base_volume, solid_count=1, is_valid_solid=True,
                      child_count=5))
    doc.add(BodyState(label="EnclosureLid", bbox=(L, W, T),
                      z_range=(H, H + T), volume=L * W * T, solid_count=1,
                      is_valid_solid=True, through_hole_count=4))
    results = validate(doc, {"L": L, "W": W, "H": H}, _ENCLOSURE_RULES)
    check(all(r.passed for r in results), "correct enclosure passes: "
          + "; ".join(r.message for r in results if not r.passed))
    check(any(r.skipped for r in results), "press-fit variant skipped")
    check(compute_pass_rate(results) == 1.0, "pass rate 1.0")

    # A wall-eating shell fails the volume formula.
    bad = DocumentState()
    bad.add(BodyState(label="EnclosureBase", bbox=(L, W, H), z_range=(0.0, H),
                      volume=L * W * H, solid_count=1, is_valid_solid=True,
                      child_count=5))
    bad.add(BodyState(label="EnclosureLid", bbox=(L, W, T),
                      z_range=(H, H + T), volume=L * W * T, solid_count=1,
                      is_valid_solid=True, through_hole_count=4))
    bad_results = validate(bad, {"L": L, "W": W, "H": H}, _ENCLOSURE_RULES)
    check(any(r.check == "volume" and not r.passed for r in bad_results),
          "solid block fails the shell volume formula")

    # Missing measurements fail loudly, not silently.
    sparse = DocumentState()
    sparse.add(BodyState(label="EnclosureBase"))
    sparse.add(BodyState(label="EnclosureLid"))
    sparse_results = validate(sparse, {"L": L, "W": W, "H": H}, _ENCLOSURE_RULES)
    check(any("not measured" in r.message for r in sparse_results),
          "unmeasured facts are explicit failures")

    if failures:
        for f in failures:
            print(f"selfcheck FAIL: {f}")
        return 1
    print("validation_rules_md selfcheck: OK")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="VALIDATION.md declarative geometry-check language (freecad-ai)")
    parser.add_argument("--selfcheck", action="store_true",
                        help="run the built-in end-to-end checks")
    args = parser.parse_args(argv)
    if args.selfcheck:
        return _selfcheck()
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
