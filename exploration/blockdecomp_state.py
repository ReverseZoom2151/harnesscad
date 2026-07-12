"""MDP state and fixed-size local observation for RL block decomposition.

From *Reinforcement Learning for Block Decomposition of CAD Models* (DiPrete
et al., AAAI-2022). Because the environment "is dynamically changing as we make
cuts", a naive global observation varies in size and cannot feed a fixed-input
network; the authors therefore treat each part independently and build a
**fixed-size local observation at a chosen model vertex** (Sec. "Local
Observation"). The local observation features are:

  * vectors to the two neighbouring vertices,
  * the type of interior angle formed (acute / right / obtuse / reentrant),
  * the vector to the centroid of the shape being processed,
  * the aspect ratio of the shape being processed.

This yields the paper's 9-dimensional network input (Sec. "Network
Architecture"): 2 + 2 for the neighbour vectors, 2 for the centroid vector, 1
for the angle, 2 for the aspect ratio. This module builds that observation
deterministically and models the **decomposition MDP state**: the paper splits
the model, "set[s] aside quadrilateral parts, and put[s] the remaining parts in
a processing queue" (Sec. "Training Phase"); an episode ends when every part is
a quad. The *learned* policy that consumes the observation is external and NOT
modelled here.

Pure stdlib; deterministic (no randomness unless a seed is supplied).
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Sequence, Tuple

from geometry.blockdecomp_domain import Corner, Shape, Vec2, classify_angle
from geometry.blockdecomp_cut import CutAction, split_step

_EPS = 1e-9


@dataclass(frozen=True)
class LocalObservation:
    """Fixed-size local observation at a model vertex (paper's 9-D input)."""

    vertex: Vec2
    to_prev: Vec2  # vector from vertex to previous neighbour
    to_next: Vec2  # vector from vertex to next neighbour
    to_centroid: Vec2  # vector from vertex to shape centroid
    angle: float  # interior corner angle in degrees
    angle_type: str  # "acute" / "right" / "obtuse" / "reentrant"
    aspect_ratio: float

    def to_vector(self) -> Tuple[float, ...]:
        """The 9-D feature vector fed to the actor/critic networks."""
        return (
            self.to_prev[0],
            self.to_prev[1],
            self.to_next[0],
            self.to_next[1],
            self.to_centroid[0],
            self.to_centroid[1],
            self.angle,
            self.aspect_ratio,
            self.aspect_ratio,  # 2 components for aspect ratio (paper's input=9)
        )


def _angle_type(corner: Corner) -> str:
    if corner.corner_type == "reentrant":
        return "reentrant"
    v1 = (corner.prev[0] - corner.pos[0], corner.prev[1] - corner.pos[1])
    v2 = (corner.nxt[0] - corner.pos[0], corner.nxt[1] - corner.pos[1])
    return classify_angle(v1, v2)


def observe(shape: Shape, corner: Corner) -> LocalObservation:
    """Build the local observation at ``corner`` of ``shape``."""
    cx, cy = shape.centroid()
    px, py = corner.pos
    return LocalObservation(
        vertex=corner.pos,
        to_prev=(corner.prev[0] - px, corner.prev[1] - py),
        to_next=(corner.nxt[0] - px, corner.nxt[1] - py),
        to_centroid=(cx - px, cy - py),
        angle=corner.interior_angle,
        angle_type=_angle_type(corner),
        aspect_ratio=shape.aspect_ratio(),
    )


def observe_all(shape: Shape) -> List[LocalObservation]:
    """Local observations at every model vertex of ``shape``."""
    return [observe(shape, c) for c in shape.corners()]


@dataclass
class DecompositionState:
    """The MDP state: the queue of non-quad parts and finished quad blocks.

    Mirrors the paper's per-part treatment: quadrilateral parts are set aside as
    finished blocks and the remaining parts wait in a processing queue. The
    state is terminal when the queue is empty (fully decomposed into blocks).
    """

    queue: Deque[Shape] = field(default_factory=deque)
    blocks: List[Shape] = field(default_factory=list)
    steps: int = 0

    @staticmethod
    def initial(domain: Shape) -> "DecompositionState":
        st = DecompositionState()
        if domain.is_quad():
            st.blocks.append(domain)
        else:
            st.queue.append(domain)
        return st

    @property
    def is_terminal(self) -> bool:
        return len(self.queue) == 0

    def current(self) -> Optional[Shape]:
        """The part currently being processed (front of the queue)."""
        return self.queue[0] if self.queue else None

    def legal_actions(self) -> List[CutAction]:
        from geometry.blockdecomp_cut import cut_candidates

        cur = self.current()
        return cut_candidates(cur) if cur is not None else []

    def apply(self, action: CutAction) -> "DecompositionState":
        """Apply a cut to the current part, returning a new state.

        Effective cuts enqueue non-quad parts and bank quad parts. An
        ineffective cut (no subdivision) leaves the part in the queue but
        advances the step counter (the paper penalises such actions).
        """
        cur = self.current()
        if cur is None:
            raise ValueError("no part to process (terminal state)")
        res = split_step(cur, action)
        new_queue = deque(self.queue)
        new_queue.popleft()
        new_blocks = list(self.blocks)
        if not res.is_effective:
            # Cut did not affect the model: keep the part for another attempt.
            new_queue.append(cur)
        else:
            for q in res.quads:
                new_blocks.append(q)
            for nq in res.non_quads:
                new_queue.append(nq)
        return DecompositionState(
            queue=new_queue, blocks=new_blocks, steps=self.steps + 1
        )

    def all_blocks(self) -> List[Shape]:
        """All finished blocks (only complete once terminal)."""
        return list(self.blocks)


def select_vertex_index(state: DecompositionState, weights: Sequence[float],
                        rng: Optional[random.Random] = None) -> int:
    """Pick a model-vertex index of the current part.

    Deterministic (highest weight, the paper's deployment behaviour) when
    ``rng`` is None; otherwise samples proportional to ``weights`` -- the
    stochastic value-network selection used during training. Ties broken by
    lowest index for determinism.
    """
    if not weights:
        raise ValueError("no weights")
    if rng is None:
        best = 0
        for k in range(1, len(weights)):
            if weights[k] > weights[best] + _EPS:
                best = k
        return best
    total = sum(max(0.0, w) for w in weights)
    if total < _EPS:
        return 0
    r = rng.random() * total
    acc = 0.0
    for k, w in enumerate(weights):
        acc += max(0.0, w)
        if r <= acc:
            return k
    return len(weights) - 1
