"""trajectory_corpus — the grounding corpus pattern, extended to whole TRAJECTORIES.

:mod:`harnesscad.eval.grounding.corpus` self-labels CLICKS: it owns the B-rep and the
camera, projects an entity to a pixel, and lets the app's own picker adjudicate whether
a click there selects it -- 942 verified ``(screenshot, description) -> (x, y)`` pairs a
minute, correct by construction, because the label is an input and not a prediction.

This module lifts that SAME method from a click to a whole op stream. Instead of
projecting one entity and adjudicating one click, it takes a known-correct CISP op
stream, COMPILES an expert
:class:`~harnesscad.agents.cua.verified_trajectory.VerifiedTrajectory` from it
(:mod:`harnesscad.agents.cua.trajectory_compiler`), and lets the exact geometric gate
adjudicate every step and the finished part. The output is supervised GUI-trajectory
data -- ``(observation, action, oracle-verdict)`` per step -- at ``p = 1.0``, the data
Fara-7B trains a verifier model to approximate and we compute for free.

Composition, not new machinery
------------------------------
* the parts come from ``corpus.sample_parts`` / ``corpus._families`` -- the same seeded
  ``ParametricSampler`` streams the click corpus uses, so a trajectory corpus and a
  click corpus drawn from the same seed describe the SAME solids.
* the trajectories come from ``trajectory_compiler.compile_expert_trajectory`` -- the
  scripted-kernel oracle labels each step and the gate labels the whole part.
* the per-step reward (exact credit assignment) comes from
  ``step_reward_cua.TrajectoryReward.from_trajectory`` -- composed, not re-derived.

It is a COMPILER, not a filter, for exactly the reason ``corpus`` is: at cold start an
agent policy's success rate on a multi-step CAD task is a fraction of a percent, and you
cannot filter what you never sample. Every trajectory here is correct by construction.

.. warning:: REWARD-HACKING GUARD.

   The scripted kernel / GUI interpreter that LABELS these trajectories is THE ORACLE.
   It must never be in a training environment's action space: a policy that can reach a
   Python console has one optimal move -- paste the script, terminate, collect full
   reward, learn nothing. This generator is data-generation and evaluation; keep the
   oracle on this side of the wall. Same warning as
   :mod:`harnesscad.agents.cua.trajectory_compiler`, ``grounding.corpus`` and
   ``io.cua.viewport``.

Import-safe and deterministic. The scripted backend is spun up only when
:func:`generate` (or :meth:`TrajectoryCorpusGenerator.run`) is called; nothing runs at
import, and no model is ever involved.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from harnesscad.agents.cua.trajectory_compiler import ExpertTrajectoryCompiler
from harnesscad.agents.cua.verified_trajectory import (
    VERIFIED,
    VerifiedTrajectory,
)
from harnesscad.agents.cua.step_reward_cua import TrajectoryReward
from harnesscad.eval.grounding import corpus as click_corpus

__all__ = [
    "TrajectoryCorpusStats",
    "TrajectoryCorpusGenerator",
    "compile_corpus",
    "generate",
    "write_trajectory_corpus",
    "read_trajectories",
    "main",
]


@dataclass
class TrajectoryCorpusStats:
    """The numbers that keep a trajectory corpus honest.

    ``trajectories`` / ``fully_verified`` count whole streams; ``steps`` /
    ``verified_steps`` count individual ``(observation, action, verdict)`` triples.
    ``by_op`` tallies verified/total per op tag -- the per-op label yield, the
    trajectory analogue of the click corpus's per-kind discard rate.
    """

    samples: int = 0
    trajectories: int = 0
    fully_verified: int = 0
    steps: int = 0
    verified_steps: int = 0
    skipped: int = 0
    elapsed: float = 0.0
    by_op: Dict[str, List[int]] = field(default_factory=dict)   # op -> [verified, total]

    def record(self, traj: VerifiedTrajectory) -> None:
        self.trajectories += 1
        if traj.is_fully_verified():
            self.fully_verified += 1
        for step in traj.steps:
            self.steps += 1
            verb = str(step.action.get("verb", "?")) if isinstance(step.action, dict) else "?"
            slot = self.by_op.setdefault(verb, [0, 0])
            slot[1] += 1
            if step.verdict.label == VERIFIED:
                slot[0] += 1
                self.verified_steps += 1

    @property
    def verified_step_rate(self) -> float:
        return 0.0 if not self.steps else self.verified_steps / float(self.steps)

    @property
    def trajectories_per_minute(self) -> float:
        if self.elapsed <= 0.0:
            return 0.0
        return self.trajectories * 60.0 / self.elapsed

    def to_dict(self) -> dict:
        return {
            "samples": self.samples,
            "trajectories": self.trajectories,
            "fully_verified": self.fully_verified,
            "steps": self.steps,
            "verified_steps": self.verified_steps,
            "verified_step_rate": round(self.verified_step_rate, 4),
            "skipped": self.skipped,
            "elapsed_s": round(self.elapsed, 2),
            "trajectories_per_minute": round(self.trajectories_per_minute, 1),
            "by_op": {k: {"verified": v[0], "total": v[1]}
                      for k, v in sorted(self.by_op.items())},
        }


class TrajectoryCorpusGenerator:
    """Compiles verified expert trajectories from seeded op streams.

    Holds one :class:`ExpertTrajectoryCompiler` (default: the scripted-kernel oracle
    path, which needs no live GUI) and sweeps the same seeded parts the click corpus
    uses. Each part's op stream is compiled into a p=1.0 VerifiedTrajectory; the gate
    labels the whole part and the per-step oracle labels every step.
    """

    def __init__(self, compiler: Optional[ExpertTrajectoryCompiler] = None,
                 with_rewards: bool = True) -> None:
        self.compiler = compiler or ExpertTrajectoryCompiler()
        self.with_rewards = with_rewards

    def compile_one(self, sample: str, brief: str,
                    ops: Sequence[Any]) -> Tuple[VerifiedTrajectory, Optional[TrajectoryReward]]:
        """Compile one part into a labelled trajectory (+ its per-step reward)."""
        traj = self.compiler.compile(brief, list(ops), trajectory_id=sample)
        reward = TrajectoryReward.from_trajectory(traj) if self.with_rewards else None
        return traj, reward

    def run(self, count: int = 4, seed: int = 0, progress: bool = False
            ) -> Tuple[List[VerifiedTrajectory], List[Optional[TrajectoryReward]],
                       TrajectoryCorpusStats]:
        """Compile ``count`` trajectories from seed ``seed``. Deterministic.

        Runs the scripted kernel through the compiler; a part whose stream will not
        compile is SKIPPED and counted, never silently dropped. Nothing here runs until
        this method is called.
        """
        stats = TrajectoryCorpusStats()
        trajectories: List[VerifiedTrajectory] = []
        rewards: List[Optional[TrajectoryReward]] = []
        t0 = time.perf_counter()
        for sample, brief, ops, _params in click_corpus.sample_parts(count, seed=seed):
            stats.samples += 1
            try:
                traj, reward = self.compile_one(sample, brief, ops)
            except Exception as exc:  # noqa: BLE001 - an uncompilable part is a skip
                stats.skipped += 1
                if progress:
                    print("  skip %s: %s: %s" % (sample, type(exc).__name__, exc))
                continue
            stats.record(traj)
            trajectories.append(traj)
            rewards.append(reward)
            if progress:
                nver = sum(1 for s in traj.steps if s.verdict.label == VERIFIED)
                print("  %s %3d/%-3d steps verified; final=%s"
                      % (sample, nver, len(traj.steps), traj.final_verdict.label))
        stats.elapsed = time.perf_counter() - t0
        return trajectories, rewards, stats


# --------------------------------------------------------------------------- #
# Persistence -- one JSON object per line, sorted keys, no wall clock in the row.
# --------------------------------------------------------------------------- #
def write_trajectory_corpus(outdir: str, trajectories: Sequence[VerifiedTrajectory],
                            stats: TrajectoryCorpusStats,
                            rewards: Optional[Sequence[Optional[TrajectoryReward]]] = None
                            ) -> str:
    """Write ``trajectories.jsonl`` + ``stats.json`` (+ per-step rewards when present).

    Each JSONL row is a VerifiedTrajectory; when a matching :class:`TrajectoryReward`
    is supplied it is embedded under ``"reward"`` so the credit-assignment signal
    travels with the trajectory. Deterministic: same seed, same bytes.
    """
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "trajectories.jsonl")
    rlist = list(rewards) if rewards is not None else [None] * len(trajectories)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for traj, reward in zip(trajectories, rlist):
            row = traj.to_dict()
            if reward is not None:
                row["reward"] = reward.to_dict()
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    with open(os.path.join(outdir, "stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats.to_dict(), fh, indent=2, sort_keys=True)
    return path


def read_trajectories(path: str) -> List[VerifiedTrajectory]:
    """Read a trajectory corpus back (dropping any embedded ``reward`` block)."""
    out: List[VerifiedTrajectory] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(VerifiedTrajectory.from_dict(json.loads(line)))
    return out


def compile_corpus(count: int = 4, seed: int = 0, with_rewards: bool = True,
                   progress: bool = False
                   ) -> Tuple[List[VerifiedTrajectory], List[Optional[TrajectoryReward]],
                              TrajectoryCorpusStats]:
    """Compile a corpus in memory (no disk). The programmatic entry point."""
    gen = TrajectoryCorpusGenerator(with_rewards=with_rewards)
    return gen.run(count=count, seed=seed, progress=progress)


def generate(outdir: str, count: int = 4, seed: int = 0, with_rewards: bool = True,
             progress: bool = False) -> TrajectoryCorpusStats:
    """Compile a corpus and write it. The one call the CLI needs."""
    trajectories, rewards, stats = compile_corpus(
        count=count, seed=seed, with_rewards=with_rewards, progress=progress)
    write_trajectory_corpus(outdir, trajectories, stats, rewards)
    return stats


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("outdir")
    ap.add_argument("--count", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-rewards", action="store_true",
                    help="skip per-step reward computation")
    args = ap.parse_args(list(argv) if argv is not None else None)
    stats = generate(args.outdir, count=args.count, seed=args.seed,
                     with_rewards=not args.no_rewards, progress=True)
    print(json.dumps(stats.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
