"""TRIZ contradiction matrix as a deterministic ideation scaffold (Lee et al.,
2026, "Enhancing Creativity in 3D Generative Design via a TRIZ-Inspired
Text-to-CAD Framework").

The paper's Stage-2 (design enhancement) embeds TRIZ analysis directly into the
LLM prompt: given a *technical contradiction* -- improving one engineering
parameter degrades another -- the classical TRIZ contradiction matrix recommends
a small set of *inventive principles* to try. The worked example (Section 4.2)
resolves Strength (improving feature #14) vs. Weight of moving object (worsening
feature #1), for which the matrix recommends Segmentation (#1), Anti-weight
(#8), Dynamics (#15) and Composite Materials (#40).

This module is the *deterministic* core of that scaffold: the canonical 40
inventive principles, the 39 engineering parameters, a contradiction-matrix
lookup, and a prompt-structure builder (Role/Task/Requirements/Context). It
makes NO model call and does no generation -- it only turns a named
contradiction into the structured guidance block the paper injects.

The recommendation table is *seeded*, not exhaustive: it carries the cells this
work documents plus a handful of classical cells. Unknown cells return ``()``;
callers should treat an empty recommendation as "no seeded guidance" rather than
"no principles exist".
"""

from __future__ import annotations

# The 40 TRIZ inventive principles (canonical numbering).
INVENTIVE_PRINCIPLES = {
    1: "Segmentation",
    2: "Taking out (extraction)",
    3: "Local quality",
    4: "Asymmetry",
    5: "Merging",
    6: "Universality",
    7: "Nested doll",
    8: "Anti-weight",
    9: "Preliminary anti-action",
    10: "Preliminary action",
    11: "Beforehand cushioning",
    12: "Equipotentiality",
    13: "The other way round",
    14: "Spheroidality (curvature)",
    15: "Dynamics",
    16: "Partial or excessive actions",
    17: "Another dimension",
    18: "Mechanical vibration",
    19: "Periodic action",
    20: "Continuity of useful action",
    21: "Skipping (rushing through)",
    22: "Blessing in disguise",
    23: "Feedback",
    24: "Intermediary",
    25: "Self-service",
    26: "Copying",
    27: "Cheap short-living objects",
    28: "Mechanics substitution",
    29: "Pneumatics and hydraulics",
    30: "Flexible shells and thin films",
    31: "Porous materials",
    32: "Color changes",
    33: "Homogeneity",
    34: "Discarding and recovering",
    35: "Parameter changes",
    36: "Phase transitions",
    37: "Thermal expansion",
    38: "Strong oxidants (boosted interactions)",
    39: "Inert atmosphere",
    40: "Composite materials",
}

# The 39 TRIZ engineering parameters (canonical numbering).
ENGINEERING_PARAMETERS = {
    1: "Weight of moving object",
    2: "Weight of stationary object",
    3: "Length of moving object",
    4: "Length of stationary object",
    5: "Area of moving object",
    6: "Area of stationary object",
    7: "Volume of moving object",
    8: "Volume of stationary object",
    9: "Speed",
    10: "Force",
    11: "Stress or pressure",
    12: "Shape",
    13: "Stability of the object's composition",
    14: "Strength",
    15: "Duration of action of moving object",
    16: "Duration of action by stationary object",
    17: "Temperature",
    18: "Illumination intensity",
    19: "Use of energy by moving object",
    20: "Use of energy by stationary object",
    21: "Power",
    22: "Loss of energy",
    23: "Loss of substance",
    24: "Loss of information",
    25: "Loss of time",
    26: "Quantity of substance",
    27: "Reliability",
    28: "Measurement accuracy",
    29: "Manufacturing precision",
    30: "Object-affected harmful factors",
    31: "Object-generated harmful factors",
    32: "Ease of manufacture",
    33: "Ease of operation",
    34: "Ease of repair",
    35: "Adaptability or versatility",
    36: "Device complexity",
    37: "Difficulty of detecting and measuring",
    38: "Extent of automation",
    39: "Productivity",
}

# Seeded contradiction-matrix cells: (improving, worsening) -> principle ids.
# The (14, 1) cell is the one this paper documents (Section 4.2). The remainder
# are classical, widely-tabulated cells retained as a small starter set.
_MATRIX = {
    (14, 1): (1, 8, 15, 40),   # Strength vs Weight of moving object (paper)
    (14, 2): (40, 26, 27, 1),  # Strength vs Weight of stationary object
    (1, 14): (28, 27, 18, 40),  # Weight of moving object vs Strength
    (1, 27): (3, 11, 1, 27),    # Weight of moving object vs Reliability
    (13, 14): (17, 9, 15),      # Stability vs Strength
    (11, 14): (10, 15, 36, 28),  # Stress/pressure vs Strength
    (12, 14): (30, 14, 10, 40),  # Shape vs Strength
    (35, 14): (35, 3, 32, 6),    # Adaptability vs Strength
}


def principle_name(principle_id: int) -> str:
    """Name of an inventive principle by id (1..40)."""
    if principle_id not in INVENTIVE_PRINCIPLES:
        raise ValueError(f"unknown inventive principle: {principle_id}")
    return INVENTIVE_PRINCIPLES[principle_id]


def parameter_name(parameter_id: int) -> str:
    """Name of an engineering parameter by id (1..39)."""
    if parameter_id not in ENGINEERING_PARAMETERS:
        raise ValueError(f"unknown engineering parameter: {parameter_id}")
    return ENGINEERING_PARAMETERS[parameter_id]


def recommend_principles(improving: int, worsening: int):
    """Inventive-principle ids for an (improving, worsening) contradiction.

    Returns a tuple of principle ids for a seeded cell, or ``()`` when the cell
    is not seeded. Raises for unknown parameter ids or a self-contradiction
    (a parameter cannot both improve and worsen).
    """
    parameter_name(improving)
    parameter_name(worsening)
    if improving == worsening:
        raise ValueError("improving and worsening features must differ")
    return _MATRIX.get((improving, worsening), ())


def recommend_named(improving: int, worsening: int):
    """Like :func:`recommend_principles` but as (id, name) pairs."""
    return tuple((pid, principle_name(pid))
                 for pid in recommend_principles(improving, worsening))


def enhancement_context(baseline_reference: str, improving: int,
                        worsening: int):
    """Build the paper's Stage-2 structured prompt scaffold (deterministic).

    Mirrors Figure 3(b): a Role/Task/Requirements/Context block that names the
    baseline, the identified contradiction and the recommended principles. This
    is *scaffold text* only -- it performs no generation and calls no model.

    Returns a mapping with keys: role, task, requirements, context. The context
    holds the resolved contradiction and the (id, name) principle
    recommendations.
    """
    principles = recommend_named(improving, worsening)
    return {
        "role": "You are a CAD design expert applying TRIZ inventive principles.",
        "task": ("Produce parametrically-editable, geometrically-valid CAD code "
                 "that resolves the identified technical contradiction while "
                 "preserving the baseline's parametric structure."),
        "requirements": (
            "Preserve variable-based dimensions and modular structure; keep the "
            "model a single valid solid; apply only the recommended principles.",
        ),
        "context": {
            "baseline_reference": baseline_reference,
            "improving_feature": (improving, parameter_name(improving)),
            "worsening_feature": (worsening, parameter_name(worsening)),
            "recommended_principles": principles,
        },
    }
