"""Reference-policy episode runner: proof the agent-environment loop closes.

This package drives the real :class:`~harnesscad.io.surfaces.mcp.gym.CADGymEnv`
end to end with a DETERMINISTIC, SCRIPTED reference policy (no model, no API
call) and records the first real trajectories, eval table, and populated
leaderboard row. See :mod:`harnesscad.eval.run.reference_loop`.
"""

from __future__ import annotations

__all__ = ["reference_loop"]
