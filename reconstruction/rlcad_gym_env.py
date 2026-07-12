"""RLCAD training-gym environment mechanics (deterministic core).

RLCAD frames B-Rep -> command-sequence reconstruction as a Markov Decision
Process and provides a **CAD training gym** (Sec. 3.2, Sec. 4): the policy emits
an action, the gym applies it to the *current* geometry to produce the next
state, and a reward is computed from the geometric difference between the current
and target geometry. The learned policy network (UV-Net + Actor-Critic, PPO) is
external; the **gym environment itself is deterministic**, and that is what this
module implements.

To stay stdlib-only and testable without a geometry kernel, geometry is modelled
as **voxel occupancy** -- a solid is a frozenset of integer voxel coordinates.
Each action carries the voxel *primitive* the kernel would produce from parsing
its face(s); the gym combines it with the current body through the action's
Boolean operation (newbody / union / intersection / subtraction, Sec. 4). This
reproduces the gym mechanics the paper relies on:

* ``reset() -> obs`` starts from an empty body;
* ``step(action) -> (obs, reward, done, info)`` applies one command;
* validity is checked before applying, and RLCAD's **mark-and-revert** mechanism
  (Sec. 6.2, "stable mark-and-revert") lets a candidate be trialled and rolled
  back with steady memory;
* the reward is a **geometric-agreement** composite: IoU (global volumetric
  alignment, Sec. 5.5) blended with an MMD-style symmetric-difference term and a
  Normal-Consistency-style boundary-overlap term. The learned Neural Reward (NR)
  term is external and omitted.

No ground truth leaks into ``obs`` beyond the scalar reward signal. Deterministic;
any tie-breaking is index-ordered, no wall clock.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Hashable, List, Optional, Tuple

from reconstruction.rlcad_command_spec import (
    INTERSECTION, NEWBODY, SUBTRACTION, UNION,
)

Voxel = Tuple[int, ...]
Body = FrozenSet[Voxel]


# --- Boolean combination on voxel bodies (Sec. 4) --------------------------
def boolean_apply(current: Body, primitive: Body, op: str) -> Body:
    """Combine the current body with a primitive under a Boolean op.

    * ``newbody``      -- start a fresh body (union with current if non-empty,
      i.e. an added disjoint body);
    * ``union``        -- ``current | primitive``;
    * ``intersection`` -- ``current & primitive``;
    * ``subtraction``  -- ``current - primitive``.
    """
    cur = frozenset(current)
    prim = frozenset(primitive)
    if op == NEWBODY:
        return cur | prim
    if op == UNION:
        return cur | prim
    if op == INTERSECTION:
        return cur & prim
    if op == SUBTRACTION:
        return cur - prim
    raise ValueError(f"unknown boolean op: {op!r}")


# --- geometric-agreement metrics (Sec. 5.5 / 6.1) --------------------------
def iou(a: Body, b: Body) -> float:
    """Volumetric Intersection-over-Union; 1.0 when both bodies are empty."""
    sa, sb = frozenset(a), frozenset(b)
    union = sa | sb
    if not union:
        return 1.0
    return len(sa & sb) / len(union)


def _surface_voxels(body: Body) -> FrozenSet[Voxel]:
    """Boundary voxels: those with a 6-neighbour not in the body."""
    s = frozenset(body)
    if not s:
        return s
    dim = len(next(iter(s)))
    out = set()
    for v in s:
        for axis in range(dim):
            for step in (-1, 1):
                nb = tuple(v[i] + (step if i == axis else 0) for i in range(dim))
                if nb not in s:
                    out.add(v)
                    break
            else:
                continue
            break
    return frozenset(out)


def normal_consistency(a: Body, b: Body) -> float:
    """Boundary-overlap proxy for Normal Consistency (Sec. 5.5).

    IoU of the two bodies' surface-voxel sets -- rewards matching boundaries,
    the deterministic surrogate for RLCAD's surface-normal alignment term.
    """
    return iou(_surface_voxels(a), _surface_voxels(b))


def mmd_term(a: Body, b: Body) -> float:
    """MMD-style closeness in ``[0, 1]`` from the symmetric difference (Sec. 5.5).

    ``1 - |A xor B| / |A union B|`` (== IoU here), reported separately so it can
    be weighted independently in the composite. Negative-distance-as-reward:
    larger is better. Empty/empty -> 1.0.
    """
    sa, sb = frozenset(a), frozenset(b)
    union = sa | sb
    if not union:
        return 1.0
    sym = (sa - sb) | (sb - sa)
    return 1.0 - len(sym) / len(union)


# Composite reward weights (paper Sec. 5.5: alpha=0.3 IoU, beta=0.2 MMD,
# gamma=0.2 NC, delta=0.3 NR). NR is a learned/external term -> omitted here,
# and its weight is redistributed proportionally across the geometric terms.
_W_IOU, _W_MMD, _W_NC = 0.3, 0.2, 0.2
_W_SUM = _W_IOU + _W_MMD + _W_NC


def composite_reward(current: Body, target: Body,
                     normalized: bool = True) -> float:
    """Weighted geometric-agreement reward ``R = a*IoU + b*MMD + c*NC``.

    With ``normalized`` the weights are rescaled to sum to 1 (NR omitted), giving
    a reward in ``[0, 1]`` where 1.0 is exact geometric agreement.
    """
    r = (_W_IOU * iou(current, target)
         + _W_MMD * mmd_term(current, target)
         + _W_NC * normal_consistency(current, target))
    return r / _W_SUM if normalized else r


# --- action bundle: a command paired with its kernel-parsed primitive ------
@dataclass(frozen=True)
class GymAction:
    """A gym action: a command-spec action tuple + the voxel primitive it builds.

    ``key`` names the action in the gym's discrete action space. ``valid`` lets a
    caller pre-mark an action infeasible (e.g. a revolve whose profile crosses
    the axis, per :mod:`geometry.rlcad_revolve`), so the gym skips it.
    """

    key: Hashable
    op: str
    primitive: Body
    valid: bool = True


# --- the gym ---------------------------------------------------------------
@dataclass
class RevolveGymEnv:
    """A deterministic RLCAD-style gym over voxel geometry.

    Parameters
    ----------
    target : Body
        The target solid the policy must reconstruct (never exposed in obs).
    actions : list[GymAction]
        The discrete action space (extrude/revolve commands + primitives).
    max_steps : int, optional
        Episode horizon; defaults to ``2 * len(actions)`` (>=1).
    """

    target: Body
    actions: List[GymAction]
    max_steps: Optional[int] = None

    _by_key: Dict[Hashable, GymAction] = field(default_factory=dict, init=False)
    _state: Body = field(default=frozenset(), init=False)
    _steps: int = field(default=0, init=False)
    _mark: Optional[Body] = field(default=None, init=False)

    def __post_init__(self):
        self.target = frozenset(self.target)
        self._by_key = {a.key: a for a in self.actions}
        if len(self._by_key) != len(self.actions):
            raise ValueError("duplicate action keys in action space")
        if self.max_steps is None:
            self.max_steps = max(1, 2 * len(self.actions))

    # -- core Gym API -------------------------------------------------------
    def reset(self) -> Dict:
        """Start a fresh episode from an empty body; returns the observation."""
        self._state = frozenset()
        self._steps = 0
        self._mark = None
        return self._obs()

    def state(self) -> Body:
        """The current geometric state ``G`` (the built body)."""
        return self._state

    def action_keys(self) -> List[Hashable]:
        return [a.key for a in self.actions]

    def valid_action_keys(self) -> List[Hashable]:
        """Keys of actions currently applicable (flagged valid, non-empty primitive)."""
        return [a.key for a in self.actions if self.is_valid(a.key)]

    def is_valid(self, key: Hashable) -> bool:
        """Whether an action may be applied to the current state."""
        a = self._by_key.get(key)
        if a is None or not a.valid:
            return False
        # Intersection/subtraction with an empty primitive are degenerate no-ops.
        if not a.primitive and a.op in (INTERSECTION, SUBTRACTION):
            return False
        return True

    def step(self, key: Hashable) -> Tuple[Dict, float, bool, Dict]:
        """Apply one action; returns ``(obs, reward, done, info)``.

        An invalid action does not modify the state and yields a penalty; a valid
        one applies its Boolean op and rewards the resulting geometric agreement.
        """
        self._steps += 1
        info: Dict = {"applied": False, "valid": False, "key": key}

        if not self.is_valid(key):
            info["reason"] = "invalid_action"
            reward = -1.0
            done = self._steps >= self.max_steps
            return self._obs(), reward, done, info

        a = self._by_key[key]
        prev = self._state
        self._state = boolean_apply(self._state, a.primitive, a.op)
        reward = composite_reward(self._state, self.target)
        info.update(applied=True, valid=True,
                    iou=iou(self._state, self.target),
                    delta_iou=iou(self._state, self.target) - iou(prev, self.target))
        done = self._state == self.target or self._steps >= self.max_steps
        info["solved"] = self._state == self.target
        return self._obs(), reward, done, info

    # -- mark-and-revert (Sec. 6.2) ----------------------------------------
    def mark(self) -> None:
        """Record the current state so a trial action can be rolled back."""
        self._mark = self._state

    def revert(self) -> None:
        """Restore the state saved by the last :meth:`mark`."""
        if self._mark is None:
            raise RuntimeError("revert() called without a prior mark()")
        self._state = self._mark
        self._mark = None

    def trial(self, key: Hashable) -> Body:
        """Return the state that applying ``key`` *would* produce, without committing.

        Used by validity enumeration: "each candidate operation is executed
        individually within the gym ... to validate" (Sec. 5.3).
        """
        if not self.is_valid(key):
            return self._state
        a = self._by_key[key]
        return boolean_apply(self._state, a.primitive, a.op)

    # -- observation (no ground truth beyond scalar reward) ----------------
    def _obs(self) -> Dict:
        return {
            "n_voxels": len(self._state),
            "surface_voxels": len(_surface_voxels(self._state)),
            "steps": self._steps,
            "max_steps": self.max_steps,
            "n_valid_actions": len(self.valid_action_keys()),
        }
