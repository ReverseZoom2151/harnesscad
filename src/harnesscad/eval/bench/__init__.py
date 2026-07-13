"""CADBench-Verified — the SWE-bench-for-CAD evaluation harness.

docs/blueprint.md sec.16: spec -> agent builds part -> programmatic geometric
checker. Metrics rank editability/validity above fidelity: sketch-editability,
program-execution rebuild rate, B-rep validity, dimension match; reported as
task-success-rate + trajectory efficiency, per-difficulty.

Public surface::

    from harnesscad.eval.bench import Task, load_tasks, run_task, run_suite, SuiteReport
"""

from __future__ import annotations

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
