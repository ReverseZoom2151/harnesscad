"""THE HIDDEN-PREDICATE MGC SPLIT. A contract the generator can read in full is
a contract the generator can overfit in full.

WHAT IT DOES
------------
TDAD (Rehan, Fiverr Labs, arXiv 2603.08806) and the Kitchen Loop (Roy, arXiv
2603.25697) share one anti-gaming move: **hold part of the specification back**.
TDAD splits tests into a VISIBLE set the generator sees and a HIDDEN set kept
for evaluation only; Kitchen's "unbeatable tests" are the ones the author cannot
tune toward because they never see them. A model that can read every acceptance
criterion can satisfy the letter of each one without producing the intended
part -- the MGC's many-to-one residual made exploitable. Holding predicates back
forces GENERALISATION: the visible contract shapes the part, the hidden contract
scores whether that part actually generalises to criteria it was never shown.

This module partitions a **Measured Geometric Contract (MGC)** into a
``visible_contract`` (handed to the generator) and a ``hidden_contract`` (used
only to evaluate). The partition is a **deterministic hash of predicate keys**,
seeded by the part id -- no ``random``, no clock -- so the same contract always
splits the same way (reproducible evaluation) while different parts split
differently (a model cannot learn one fixed hidden set).

Only **MEASURED, bound** predicates are eligible to be hidden. Advisory
predicates (aesthetic / ergonomic -- "looks premium") have no gate to hold back,
and unbound predicates are the ``[NEEDS CLARIFICATION]`` markers that must stay
visible so the generator knows what was left unspecified. Neither is ever
hidden.

THE MINIMAL MGC INTERFACE THIS DEPENDS ON
-----------------------------------------
The MGC type lives in :mod:`harnesscad.domain.spec.contract` (authored in
parallel). This module depends on it ONLY through the following duck-typed
surface, imported lazily so importing this module never forces the contract
module to exist:

* ``contract.predicates`` -- a tuple of predicate objects.
* ``contract.<seed_key>``  -- the part-id attribute named by ``seed_key``
  (default ``"part_id"``); used only to seed the hash. Absent -> seed "".
* ``predicate.key``     -- a stable string identifier, hashed for the partition.
* ``predicate.kind``    -- MEASURED or ADVISORY (enum member or string; matched
  by name, case-insensitive).
* ``predicate.unbound`` -- bool; an unspecified quantity that must stay visible.
* ``predicate.hidden``  -- bool; the flag this split sets on the two views.

If :mod:`harnesscad.domain.spec.contract` is not importable at call time, a
clear :class:`ContractModuleUnavailable` (an ``ImportError``) is raised -- the
module import itself never fails on the missing dependency.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any, List, Tuple

__all__ = [
    "ContractModuleUnavailable",
    "is_eligible_to_hide",
    "split_contract",
]


class ContractModuleUnavailable(ImportError):
    """Raised when ``harnesscad.domain.spec.contract`` cannot be imported yet."""


def _require_contract_module() -> Any:
    """Lazily import the MGC module, or raise a clear ImportError-derived error.

    Kept out of module import so that importing ``contract_split`` never depends
    on the sibling module existing; the dependency is resolved only when a split
    is actually requested.
    """
    try:
        from harnesscad.domain.spec import contract as contract_module
    except ImportError as exc:
        raise ContractModuleUnavailable(
            "the Measured Geometric Contract module "
            "'harnesscad.domain.spec.contract' is not importable yet; "
            "split_contract needs it to resolve the MGC interface. "
            "Original import error: %s" % (exc,)
        ) from exc
    return contract_module


def _kind_name(predicate: Any) -> str:
    """Normalise a predicate ``.kind`` (enum member OR string) to an upper name."""
    kind = getattr(predicate, "kind", None)
    name = getattr(kind, "name", None)
    if name is None:
        name = str(kind)
    return name.strip().upper()


def is_eligible_to_hide(predicate: Any) -> bool:
    """A predicate may be hidden only if it is MEASURED and bound.

    Advisory predicates have no measured gate to hold back, and unbound
    predicates are the ``[NEEDS CLARIFICATION]`` markers the generator must see.
    Either disqualifies the predicate from the hidden set.
    """
    if bool(getattr(predicate, "unbound", False)):
        return False
    return _kind_name(predicate) == "MEASURED"


def _partition_rank(seed: str, key: str) -> float:
    """A deterministic value in [0, 1) from the part id and the predicate key.

    SHA-256 of ``seed:key`` -> integer -> normalised. No ``random``, no clock:
    the same (part, predicate) always ranks the same, and different parts rank
    the same predicate differently (so no fixed hidden set can be learnt).
    """
    digest = hashlib.sha256(("%s:%s" % (seed, key)).encode("utf-8")).digest()
    value = int.from_bytes(digest, "big")
    return value / float(1 << (len(digest) * 8))


def _with_hidden(predicate: Any, value: bool) -> Any:
    """Return a copy of ``predicate`` with ``.hidden`` set to ``value``.

    Uses ``dataclasses.replace`` for frozen dataclasses (building a new
    predicate tuple, as required), and falls back to a shallow copy + setattr
    for non-dataclass predicate objects.
    """
    if bool(getattr(predicate, "hidden", False)) == value:
        return predicate
    if dataclasses.is_dataclass(predicate) and not isinstance(predicate, type):
        return dataclasses.replace(predicate, hidden=value)
    import copy
    clone = copy.copy(predicate)
    try:
        object.__setattr__(clone, "hidden", value)
    except Exception as exc:  # pragma: no cover - exotic predicate types.
        raise TypeError(
            "cannot set .hidden on predicate of type %r" % type(predicate).__name__
        ) from exc
    return clone


def _with_predicates(contract: Any, predicates: Tuple[Any, ...]) -> Any:
    """Return a contract-like view of ``contract`` carrying new ``predicates``."""
    if dataclasses.is_dataclass(contract) and not isinstance(contract, type):
        return dataclasses.replace(contract, predicates=predicates)
    import copy
    clone = copy.copy(contract)
    try:
        object.__setattr__(clone, "predicates", predicates)
    except Exception as exc:  # pragma: no cover - exotic contract types.
        raise TypeError(
            "cannot set .predicates on contract of type %r"
            % type(contract).__name__
        ) from exc
    return clone


def split_contract(
    contract: Any,
    hidden_fraction: float = 0.4,
    seed_key: str = "part_id",
) -> Tuple[Any, Any]:
    """Deterministically split an MGC into (visible_contract, hidden_contract).

    Roughly ``hidden_fraction`` of the MEASURED, bound predicates are held out.
    Eligible predicates are ranked by a deterministic hash of their key (seeded
    by the part id read from ``contract.<seed_key>``) and the lowest-ranked
    ``round(hidden_fraction * n)`` become hidden -- reproducible, and different
    per part. Unbound and advisory predicates are always visible.

    Returns two contract-like views over the SAME predicate set:

    * ``visible_contract`` -- every predicate with ``.hidden = False`` (the
      hidden ones removed); this is what the generator sees.
    * ``hidden_contract``  -- only the held-out predicates, each with
      ``.hidden = True``; used solely to evaluate generalisation.

    Raises :class:`ContractModuleUnavailable` if the MGC module is not
    importable at call time; raises ``ValueError`` for an out-of-range
    ``hidden_fraction``.
    """
    if not 0.0 <= hidden_fraction <= 1.0:
        raise ValueError(
            "hidden_fraction must be in [0, 1], got %r" % (hidden_fraction,))

    # Enforce the documented dependency: the MGC interface must be resolvable.
    _require_contract_module()

    predicates: Tuple[Any, ...] = tuple(getattr(contract, "predicates", ()))
    seed = str(getattr(contract, seed_key, "") or "")

    eligible = [p for p in predicates if is_eligible_to_hide(p)]
    n_hidden = int(round(hidden_fraction * len(eligible)))
    # Deterministic order: rank by hash, tie-break by key for stability.
    ranked = sorted(eligible, key=lambda p: (_partition_rank(seed, str(getattr(p, "key", ""))),
                                             str(getattr(p, "key", ""))))
    hidden_ids = set(id(p) for p in ranked[:n_hidden])

    visible_predicates: List[Any] = []
    hidden_predicates: List[Any] = []
    for p in predicates:
        if id(p) in hidden_ids:
            hidden_predicates.append(_with_hidden(p, True))
        else:
            visible_predicates.append(_with_hidden(p, False))

    visible_contract = _with_predicates(contract, tuple(visible_predicates))
    hidden_contract = _with_predicates(contract, tuple(hidden_predicates))
    return visible_contract, hidden_contract
