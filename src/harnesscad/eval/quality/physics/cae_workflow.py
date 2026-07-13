"""Seven-stage deep CAD/CAE integration workflow schema.

Deterministic schema and dependency validation for the overall framework
(Section 3, Fig. 1) of:

    Yoo et al., "Integrating deep learning into CAD/CAE system: generative design
    and evaluation of 3D conceptual wheel", Struct. Multidisc. Optim. 64 (2021)
    2725-2747.

The paper structures the conceptual-design pipeline as seven sequential stages,
each consuming the artifacts produced by earlier stages:

    1. 2D generative design            -> 2D wheel designs
    2. Dimensionality reduction        -> latent encoder + latent vectors
    3. DOE in latent space             -> sampled latent vectors (LHSnorm)
    4. 3D CAD automation               -> 3D CAD models (.stp)
    5. CAE automation                  -> modal analysis labels (frequency, mass)
    6. Transfer learning               -> trained surrogate model
    7. Visualization and analysis      -> ranked candidate concepts

Stages 1-4 form the "design generation" group; stages 5-7 form the "design
evaluation" group.  This module encodes those stages, their produced/required
artifacts, and validates that a proposed execution order respects the
dependencies (a stage may run only once all its required artifacts exist).

The schema is declarative data; nothing here trains a model or runs a solver.
Deterministic and stdlib-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set, Tuple


@dataclass(frozen=True)
class Stage:
    """One workflow stage: what it requires and what it produces."""

    number: int
    name: str
    group: str
    requires: Tuple[str, ...]
    produces: Tuple[str, ...]


# The canonical seven-stage schema (Fig. 1).
STAGES: Tuple[Stage, ...] = (
    Stage(1, "2D generative design", "design generation",
          requires=("reference_images",), produces=("wheel_2d_designs",)),
    Stage(2, "dimensionality reduction", "design generation",
          requires=("wheel_2d_designs",), produces=("latent_encoder", "latent_vectors")),
    Stage(3, "DOE in latent space", "design generation",
          requires=("latent_encoder", "latent_vectors"), produces=("sampled_latent_vectors",)),
    Stage(4, "3D CAD automation", "design generation",
          requires=("sampled_latent_vectors",), produces=("cad_3d_models",)),
    Stage(5, "CAE automation", "design evaluation",
          requires=("cad_3d_models",), produces=("modal_labels",)),
    Stage(6, "transfer learning", "design evaluation",
          requires=("modal_labels", "latent_encoder"), produces=("surrogate_model",)),
    Stage(7, "visualization and analysis", "design evaluation",
          requires=("surrogate_model",), produces=("ranked_candidates",)),
)

# The two artifacts available before any stage runs.
INITIAL_ARTIFACTS: Tuple[str, ...] = ("reference_images",)

_BY_NUMBER: Dict[int, Stage] = {s.number: s for s in STAGES}


def get_stage(number: int) -> Stage:
    """Return the :class:`Stage` with the given 1-based number."""
    try:
        return _BY_NUMBER[number]
    except KeyError:
        raise ValueError("unknown stage number: {0!r}".format(number))


def stages_in_group(group: str) -> List[Stage]:
    """Return the stages belonging to ``group`` (in schema order)."""
    result = [s for s in STAGES if s.group == group]
    if not result:
        raise ValueError("unknown group: {0!r}".format(group))
    return result


def validate_order(order: Sequence[int], initial: Sequence[str] = INITIAL_ARTIFACTS) -> bool:
    """Return ``True`` if ``order`` is a valid dependency-respecting execution.

    Walks the stage numbers in ``order``, accumulating produced artifacts; a
    stage is runnable only when all of its ``requires`` are already available.
    Raises ``ValueError`` on an unknown or duplicated stage number, or if a
    stage runs before its dependencies are satisfied.
    """
    available: Set[str] = set(initial)
    seen: Set[int] = set()
    for number in order:
        stage = get_stage(number)
        if number in seen:
            raise ValueError("stage {0} listed more than once".format(number))
        seen.add(number)
        missing = [a for a in stage.requires if a not in available]
        if missing:
            raise ValueError(
                "stage {0} ({1}) missing artifacts: {2}".format(
                    number, stage.name, ", ".join(missing)
                )
            )
        available.update(stage.produces)
    return True


def canonical_order() -> List[int]:
    """Return the canonical stage order ``[1, 2, ..., 7]``."""
    return [s.number for s in STAGES]


def topological_order(initial: Sequence[str] = INITIAL_ARTIFACTS) -> List[int]:
    """Compute a valid execution order by greedy dependency resolution.

    Repeatedly schedules the lowest-numbered stage whose requirements are met.
    Deterministic.  Raises ``ValueError`` if the schema cannot be fully ordered
    (a dependency is never producible).
    """
    available: Set[str] = set(initial)
    remaining = list(STAGES)
    order: List[int] = []
    while remaining:
        runnable = [s for s in remaining if all(a in available for a in s.requires)]
        if not runnable:
            raise ValueError("unsatisfiable dependencies among remaining stages")
        nxt = min(runnable, key=lambda s: s.number)
        order.append(nxt.number)
        available.update(nxt.produces)
        remaining.remove(nxt)
    return order


def required_upstream(number: int) -> List[int]:
    """Return the stage numbers that must run before ``number`` (transitively).

    Resolved by tracing which stages produce each required artifact.
    """
    target = get_stage(number)
    producers: Dict[str, int] = {}
    for s in STAGES:
        for art in s.produces:
            producers[art] = s.number

    needed: Set[int] = set()
    frontier: List[str] = list(target.requires)
    while frontier:
        art = frontier.pop()
        producer = producers.get(art)
        if producer is None or producer in needed:
            continue
        needed.add(producer)
        frontier.extend(get_stage(producer).requires)
    return sorted(needed)
