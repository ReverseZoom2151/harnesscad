"""Key-decision hierarchical design templates (dependent-parameter propagation).

From Séquin, *Interactive Procedural Computer-Aided Design*, Section 3.1
(OPASYN op-amp compiler). Rather than search over "all possible collections of
circuit elements", decades of design experience distil a circuit into "a few
generic stages with well defined functions". Each op-amp design class is
characterised by ``(number of stages, input stage type, output stage type)``,
and within a class "there are only 5 to 8 key decisions to be made ... and these
decisions then define most directly the other parameters". This "reduces this
potentially very large design task to a search space of only 5 to 8 dimensions".

This module implements that deterministic idea generically:

* :func:`classify` -- name a design class from its structural triple;
* :class:`KeyParameterTemplate` -- declares the handful of independent *key*
  parameters plus ordered *derivation rules* that compute every remaining
  parameter from them; :meth:`realize` expands a key-parameter assignment into
  the full parameter set, and :meth:`dimensionality` / :meth:`reduction_ratio`
  quantify the collapse of the search space.

Pure stdlib, deterministic (derivations must be pure functions).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Sequence, Tuple

# A derivation computes one parameter value from the currently-known values.
Derivation = Callable[[Mapping[str, float]], float]


def classify(num_stages: int, input_type: str, output_type: str) -> str:
    """Canonical name for an op-amp-style design class.

    The paper groups the "vast majority of all op-amp designs" by this triple.
    """
    if num_stages < 1:
        raise ValueError("num_stages must be >= 1")
    it = input_type.strip().lower()
    ot = output_type.strip().lower()
    valid = {"differential", "single_sided", "push_pull"}
    if it not in valid or ot not in valid:
        raise ValueError(f"stage types must be one of {sorted(valid)}")
    return f"{num_stages}stage_{it}_in_{ot}_out"


@dataclass
class KeyParameterTemplate:
    """A design class parameterised by a few key decisions plus derivations.

    ``key_params`` are the independent decisions the designer / optimiser sets
    (5-8 in the paper). ``derivations`` is an ordered list of
    ``(name, function)`` pairs; each function receives all values known so far
    (keys plus earlier derivations) and returns the next parameter, so later
    rules may depend on earlier ones.
    """

    design_class: str
    key_params: Tuple[str, ...]
    derivations: List[Tuple[str, Derivation]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.key_params) != len(set(self.key_params)):
            raise ValueError("duplicate key parameter names")
        known = set(self.key_params)
        for name, _ in self.derivations:
            if name in known:
                raise ValueError(f"derived parameter '{name}' collides with an existing name")
            known.add(name)

    def dimensionality(self) -> int:
        """Search-space dimension = number of key decisions."""
        return len(self.key_params)

    def full_dimensionality(self) -> int:
        """Total number of parameters after all derivations."""
        return len(self.key_params) + len(self.derivations)

    def reduction_ratio(self) -> float:
        """Full parameter count divided by the key-decision count (>= 1)."""
        return self.full_dimensionality() / self.dimensionality()

    def realize(self, key_values: Mapping[str, float]) -> Dict[str, float]:
        """Expand a key-parameter assignment into the full parameter set.

        Raises ``ValueError`` if a key parameter is missing or an extra one is
        supplied.
        """
        missing = set(self.key_params) - set(key_values)
        if missing:
            raise ValueError(f"missing key parameters: {sorted(missing)}")
        extra = set(key_values) - set(self.key_params)
        if extra:
            raise ValueError(f"unexpected parameters: {sorted(extra)}")
        result: Dict[str, float] = {k: float(key_values[k]) for k in self.key_params}
        for name, fn in self.derivations:
            result[name] = float(fn(result))
        return result


def two_stage_opamp_template() -> KeyParameterTemplate:
    """A concrete illustrative template (a two-stage differential op-amp).

    The derivations are simple, deterministic engineering-style relations (not
    fitted / learned), included so the reduction can be exercised end-to-end:
    the 5 key decisions define ~4 further parameters.
    """
    keys = ("bias_current", "output_swing", "load_cap", "gain_target", "supply_v")

    def tail_current(v):
        return 2.0 * v["bias_current"]

    def input_gm(v):
        # transconductance proportional to sqrt of bias current (monotone rule)
        return (v["bias_current"]) ** 0.5

    def comp_cap(v):
        # compensation cap sized from load cap and gain
        return v["load_cap"] * (1.0 + 0.1 * v["gain_target"])

    def slew_rate(v):
        return v["tail_current"] / v["comp_cap"]

    return KeyParameterTemplate(
        design_class=classify(2, "differential", "single_sided"),
        key_params=keys,
        derivations=[
            ("tail_current", tail_current),
            ("input_gm", input_gm),
            ("comp_cap", comp_cap),
            ("slew_rate", slew_rate),
        ],
    )
