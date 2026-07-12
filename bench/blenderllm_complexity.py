"""CADBench task-complexity metrics for BlenderLLM (Du et al., 2024).

The BlenderLLM paper characterises how hard a modelling task is with three
deterministic statistics over the generated ``bpy`` script (README, "task
difficulty"):

* **Unit number** -- how many geometric units (primitive-adder calls) the model
  builds, a proxy for geometric complexity.
* **Parameter density** -- how many numeric parameters those units carry per
  unit, a proxy for parameter intricacy.
* **Entropy** -- the Shannon entropy (bits) of the distribution of primitive
  *types*, a proxy for spatial diversity: a scene of ten identical cubes has
  zero entropy, a scene mixing cubes, spheres and cones has more.

All three are computed from the AST of the script via
:mod:`programs.blenderllm_bpy_script` -- nothing is executed in Blender.
"""

from __future__ import annotations

import ast
import math
from collections import Counter
from dataclasses import dataclass

from programs.blenderllm_bpy_script import (
    PRIMITIVE_ADDERS,
    TRANSFORM_OPS,
    check_syntax,
    extract_calls,
)


def _count_numeric_literals(node: ast.AST) -> int:
    """Number of int/float constant leaves inside an AST subtree (excl. bool)."""
    count = 0
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, (int, float)):
            if not isinstance(child.value, bool):
                count += 1
    return count


@dataclass(frozen=True)
class Complexity:
    unit_number: int
    parameter_density: float
    entropy: float
    type_distribution: dict[str, int]


def unit_number(script: str) -> int:
    """Count of geometry-creating primitive-adder calls."""
    return sum(1 for call in extract_calls(script) if call.is_primitive)


def type_distribution(script: str) -> dict[str, int]:
    """Multiplicity of each primitive type used in the script."""
    counts = Counter(
        call.op for call in extract_calls(script) if call.is_primitive
    )
    return dict(counts)


def shannon_entropy(distribution: dict[str, int]) -> float:
    """Shannon entropy (base 2, in bits) of a count distribution.

    A single type -- or no types -- yields ``0.0``; the maximum is ``log2(k)``
    for ``k`` equally-frequent types.
    """
    total = sum(distribution.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in distribution.values():
        if count <= 0:
            continue
        p = count / total
        entropy -= p * math.log2(p)
    # Clamp tiny negative round-off (e.g. -0.0) to 0.0.
    return entropy if entropy > 0.0 else 0.0


def parameter_density(script: str) -> float:
    """Numeric parameters per geometric unit across primitive/transform calls.

    Returns ``0.0`` when the script has no geometric units (no division by a
    zero unit count).
    """
    tree = ast.parse(script)
    units = unit_number(script)
    if units == 0:
        return 0.0
    numeric = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        op = func.attr
        # Count parameters only on the calls that actually shape geometry.
        if op in PRIMITIVE_ADDERS or op in TRANSFORM_OPS:
            for arg in node.args:
                numeric += _count_numeric_literals(arg)
            for kw in node.keywords:
                numeric += _count_numeric_literals(kw.value)
    return numeric / units


def complexity(script: str) -> Complexity:
    """All three CADBench complexity metrics for ``script``.

    Raises ``SyntaxError`` if the script does not parse; guard with
    :func:`programs.blenderllm_bpy_script.check_syntax` for untrusted input.
    """
    dist = type_distribution(script)
    return Complexity(
        unit_number=sum(dist.values()),
        parameter_density=parameter_density(script),
        entropy=shannon_entropy(dist),
        type_distribution=dist,
    )


def safe_complexity(script: str) -> Complexity | None:
    """Complexity of ``script``, or ``None`` if it fails the syntax gate."""
    if not check_syntax(script).ok:
        return None
    return complexity(script)
