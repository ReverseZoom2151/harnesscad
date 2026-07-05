"""CADGymEnv — the Gym interface from HARNESS_BLUEPRINT.md sec.5.

``reset() -> obs``, ``step(action) -> (obs, reward, done, info)``, ``state()``,
``render()``, ``close()``, wrapping a :class:`loop.HarnessSession` + a
:class:`backends.base.GeometryBackend`. This is the book's ``FileEditEnv`` with
"pytest passes" swapped for "geometry verifier passes":

  - **Action space** = the CISP ops (a step applies one or more ops through the
    session's applyOps -> regen -> verify -> checkpoint spine).
  - **Hybrid observation** = a geometry/B-rep summary (JSON: feature tree,
    sketch DOF, validity, digest) **+ a render hook** (lazy import of
    :mod:`render`; a placeholder note when no kernel/solid is present). Image
    *bytes* stay out of the compact obs (only per-view availability), so the obs
    never blows the context window.
  - **Reward** comes from the verifier (pass = positive), reusing
    :func:`mcp.tools.reward_from_apply`.
  - **No ground truth ever leaks into the observation** — the env holds no
    target/answer; obs is built purely from the current model state.

One interface serves serving-time agents *and* future RL training, and makes
trajectory logging trivial. Stdlib only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from backends.stub import StubBackend
from cisp.ops import Op, parse_op
from loop import HarnessSession
from surfaces.mcp.tools import ToolCatalog, reward_from_apply


class CADGymEnv:
    """A Gym-style wrapper over a HarnessSession + geometry backend."""

    def __init__(self, backend=None, verifiers=None,
                 max_steps: Optional[int] = None) -> None:
        # A backend *instance* is reused across resets (its .reset() is called by
        # the session), so no ground truth or stale geometry survives a reset.
        self._backend = backend if backend is not None else StubBackend()
        self._verifiers = verifiers
        self.max_steps = max_steps
        self._catalog = ToolCatalog()
        self.session: Optional[HarnessSession] = None
        self._steps = 0
        self._last_reward = 0.0
        self.reset()

    # --- Gym API ----------------------------------------------------------
    def reset(self) -> Dict[str, Any]:
        """Clear all state and return the initial observation."""
        self.session = HarnessSession(self._backend, verifiers=self._verifiers)
        self._steps = 0
        self._last_reward = 0.0
        return self._observe()

    def step(self, action) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Apply an action (op / dict / list / (name, args)) and return
        ``(obs, reward, done, info)``.
        """
        ops = self._coerce_action(action)
        result = self.session.apply_ops(ops)
        reward = reward_from_apply(result)
        self._last_reward = reward
        self._steps += 1

        done = False
        if self.max_steps is not None and self._steps >= self.max_steps:
            done = True

        info: Dict[str, Any] = {
            "ok": result.ok,
            "applied": result.applied,
            "digest": result.digest,
            "rejected": result.rejected,
            "reward": reward,             # tool-result carries a reward field (sec.5)
            "diagnostics": [d.to_dict() for d in result.diagnostics],
            "step": self._steps,
            "n_ops": len(ops),
        }
        return self._observe(), reward, done, info

    def state(self) -> Dict[str, Any]:
        """Full structured model state (feature tree + validity + measurements)."""
        return {
            "tree": self._catalog.read_resource("cad://model/tree", self.session),
            "validity": self._catalog.read_resource("cad://model/validity", self.session),
            "measurements": self._catalog.read_resource(
                "cad://model/measurements", self.session),
            "digest": self.session.digest(),
            "step": self._steps,
        }

    def render(self, views=None, fmt: str = "svg") -> Dict[str, Any]:
        """Render the current model to multi-view images (the vision half of the
        hybrid observation). Lazily imports :mod:`render`; returns a note-bearing
        placeholder when no kernel/solid is available (never raises)."""
        try:
            import surfaces.render as _render_mod
        except Exception as exc:  # noqa: BLE001
            return {"images": {}, "note": f"render module unavailable ({exc})",
                    "fmt": fmt, "any_rendered": False}
        kwargs: Dict[str, Any] = {"fmt": fmt}
        if views:
            kwargs["views"] = views
        result = _render_mod.render(self._backend, **kwargs)
        return {"images": result.images, "note": result.note, "fmt": result.fmt,
                "any_rendered": result.any_rendered}

    def close(self) -> None:
        """Release the session (the backend instance is left resettable)."""
        self.session = None

    # --- action space / catalog ------------------------------------------
    @property
    def catalog(self) -> ToolCatalog:
        """The MCP tool catalog describing the action space."""
        return self._catalog

    def action_space(self) -> List[str]:
        """Names of the op tools an agent may emit as actions."""
        return [t.name for t in self._catalog.op_tools()]

    # --- internals --------------------------------------------------------
    def _coerce_action(self, action) -> List[Op]:
        if action is None:
            return []
        if isinstance(action, Op):
            return [action]
        if isinstance(action, dict):
            return [parse_op(action)]
        if isinstance(action, tuple) and len(action) == 2 and isinstance(action[0], str):
            name, args = action
            return [parse_op({"op": name, **(args or {})})]
        if isinstance(action, (list, tuple)):
            ops: List[Op] = []
            for a in action:
                ops.extend(self._coerce_action(a))
            return ops
        raise TypeError(f"unsupported action type: {type(action).__name__}")

    def _observe(self) -> Dict[str, Any]:
        """Compact hybrid observation. Built purely from current model state — it
        carries NO ground-truth target/answer, only what the agent has produced.
        Render bytes are excluded (only per-view availability), keeping obs small.
        """
        backend = self.session.backend
        validity = self._catalog.read_resource("cad://model/validity", self.session)
        render_meta = self._render_meta()
        return {
            "feature_tree": {
                "summary": backend.query("summary"),
                "ops": [op.to_dict() for op in self.session.opdag.ops()],
            },
            "sketch_dof": backend.query("sketch_dof"),
            "validity": validity,
            "digest": self.session.digest(),
            "step": self._steps,
            "render": render_meta,
        }

    def _render_meta(self) -> Dict[str, Any]:
        """Availability-only render metadata for the compact obs (no bytes)."""
        try:
            import surfaces.render as _render_mod
        except Exception:  # noqa: BLE001
            return {"available": False, "note": "render module unavailable"}
        try:
            result = _render_mod.render(self._backend)
        except Exception as exc:  # noqa: BLE001 - obs must never crash
            return {"available": False, "note": f"render error ({exc})"}
        return {
            "available": result.any_rendered,
            "views": {k: (v is not None) for k, v in result.images.items()},
            "note": result.note,
        }
