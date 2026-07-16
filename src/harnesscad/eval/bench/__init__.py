"""CADBench-Verified — the SWE-bench-for-CAD evaluation harness.

docs/blueprint.md sec.16: spec -> agent builds part -> programmatic geometric
checker. Metrics rank editability/validity above fidelity: sketch-editability,
program-execution rebuild rate, B-rep validity, dimension match; reported as
task-success-rate + trajectory efficiency, per-difficulty.

Public surface::

    from harnesscad.eval.bench import Task, load_tasks, run_task, run_suite, SuiteReport

Alongside the task suite, :mod:`.analytic_fea` carries the harness's
closed-form STRUCTURAL oracles (cantilever / fixed-fixed / modal / Euler
buckling). Those are ground truth of a different kind from everything else
here: not a rubric, not a reference solution, but arithmetic -- so they are
imported below as a real edge, and their cross-check against cad-cae-copilot's
stored numbers is part of their ``--selfcheck``.
"""

from __future__ import annotations

from harnesscad.eval.bench import analytic_fea
from harnesscad.eval.bench.analytic_fea import FeaOracle, oracles as fea_oracles
from harnesscad.eval.bench.data.task import DIFFICULTIES, Task, load_task, load_tasks
from harnesscad.eval.bench.protocols.metrics import (
    assembly_mate_accuracy, brep_validity, cad_sequence_f1, collision_rate,
    dimension_match, program_execution, program_execution_rate,
    sketch_editability, trajectory_efficiency,
)
from harnesscad.eval.bench.harness.runner import (
    DifficultyReport, SuiteReport, TaskResult,
    reference_solver, run_suite, run_task,
)

__all__ = [
    "DIFFICULTIES",
    "analytic_fea",
    "FeaOracle",
    "fea_oracles",
    "Task",
    "load_task",
    "load_tasks",
    "brep_validity",
    "dimension_match",
    "program_execution",
    "program_execution_rate",
    "cad_sequence_f1",
    "assembly_mate_accuracy",
    "collision_rate",
    "sketch_editability",
    "trajectory_efficiency",
    "DifficultyReport",
    "SuiteReport",
    "TaskResult",
    "reference_solver",
    "run_suite",
    "run_task",
]
