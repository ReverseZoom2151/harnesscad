"""CADBench-Verified runner (HARNESS_BLUEPRINT.md sec.16).

Executes a Task through the HarnessSession spine (applyOps -> regen -> verify ->
checkpoint) and scores it with the blueprint metrics, then aggregates a suite
into a task-success-rate + per-difficulty + mean-trajectory-efficiency report.

Solver plug point
-----------------
``run_task`` accepts an optional ``solver: Callable[[Task], List[Op]]``. Today
the default is :func:`reference_solver` — it replays the task's own reference op
stream, so the harness is exercised end-to-end without a model in the loop. A
real NL->ops planner (LLM/agent) is a drop-in: give it the same signature
(receive the Task, return an op list) and pass it as ``solver``. Everything
downstream — the session, the metrics, the report — is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from harnesscad.core.cisp.ops import Op
from harnesscad.core.loop import HarnessSession
from harnesscad.eval.bench.data.task import DIFFICULTIES, Task
from harnesscad.eval.bench.protocols.metrics import (
    assembly_mate_accuracy, brep_validity, cad_sequence_f1, collision_rate,
    dimension_match, program_execution, program_execution_rate,
    sketch_editability, trajectory_efficiency,
)

# A solver maps a task's brief to a CISP op stream (the NL->ops planner slot).
Solver = Callable[[Task], List[Op]]
# A backend_factory returns a fresh GeometryBackend per task (isolated state).
BackendFactory = Callable[[], object]
# A session_factory wraps a backend into a HarnessSession (override for tracing).
SessionFactory = Callable[[object], HarnessSession]


def reference_solver(task: Task) -> List[Op]:
    """The default solver: replay the task's ground-truth reference ops."""
    return task.reference_ops()


@dataclass
class TaskResult:
    """The scored outcome of one task."""

    task_id: str
    difficulty: str
    success: bool
    program_execution: bool
    brep_validity: bool
    sketch_editability: float
    dimension_match: bool
    trajectory_efficiency: float
    applied: int
    emitted: int
    dimension_details: dict = field(default_factory=dict)
    # Optional metric families (None = not applicable to this task/backend):
    #   cad_sequence_f1        -> {"precision","recall","f1",...} when the task
    #                             carries reference_ops (DeepCAD entity/CAD F1).
    #   assembly_mate_accuracy -> {"mate_type_accuracy","residual_dof_error",...}
    #                             when the task carries reference_assembly AND the
    #                             backend exposes the assembly query.
    #   collision_rate         -> interference fraction; None for single-part.
    cad_sequence_f1: Optional[dict] = None
    assembly_mate_accuracy: Optional[dict] = None
    collision_rate: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "difficulty": self.difficulty,
            "success": self.success,
            "program_execution": self.program_execution,
            "brep_validity": self.brep_validity,
            "sketch_editability": self.sketch_editability,
            "dimension_match": self.dimension_match,
            "trajectory_efficiency": self.trajectory_efficiency,
            "applied": self.applied,
            "emitted": self.emitted,
            "dimension_details": self.dimension_details,
            "cad_sequence_f1": self.cad_sequence_f1,
            "assembly_mate_accuracy": self.assembly_mate_accuracy,
            "collision_rate": self.collision_rate,
        }


def run_task(task: Task,
             backend_factory: BackendFactory,
             session_factory: Optional[SessionFactory] = None,
             solver: Optional[Solver] = None) -> TaskResult:
    """Build ``task``'s part on a fresh backend and score it.

    Success (the SWE-bench-style pass) = the program rebuilt (program_execution)
    AND the B-rep is valid (brep_validity) AND the geometry matches the spec
    (dimension_match). sketch_editability and trajectory_efficiency are reported
    quality metrics, not pass/fail gates.
    """
    backend = backend_factory()
    session = (session_factory or HarnessSession)(backend)
    solve = solver or reference_solver

    ops = solve(task)
    result = session.apply_ops(ops)

    pe = program_execution(result)
    valid = brep_validity(backend)
    editability = sketch_editability(backend)
    dim_ok, dim_details = dimension_match(backend, task.acceptance)

    emitted = len(ops)
    efficiency = trajectory_efficiency(task.optimal_len(), emitted)
    success = pe and valid and dim_ok

    # Optional metric families — each None when its input isn't present.
    ref_seq = task.sequence_reference_ops()
    seq_f1 = cad_sequence_f1(ops, ref_seq) if ref_seq else None
    built_assembly = _query_assembly(backend)
    mate_acc = (assembly_mate_accuracy(built_assembly, task.ref_assembly)
                if task.ref_assembly else None)
    collisions = collision_rate([backend])

    return TaskResult(
        task_id=task.id,
        difficulty=task.difficulty,
        success=success,
        program_execution=pe,
        brep_validity=valid,
        sketch_editability=editability,
        dimension_match=dim_ok,
        trajectory_efficiency=efficiency,
        applied=result.applied,
        emitted=emitted,
        dimension_details=dim_details,
        cad_sequence_f1=seq_f1,
        assembly_mate_accuracy=mate_acc,
        collision_rate=collisions,
    )


def _query_assembly(backend) -> dict:
    """Read the optional ``assembly`` query family, tolerating backends that
    don't expose it (returns ``{}``)."""
    query = getattr(backend, "query", None)
    if not callable(query):
        return {}
    try:
        return query("assembly") or {}
    except Exception:  # noqa: BLE001 - unknown query key -> no assembly
        return {}


@dataclass
class DifficultyReport:
    n_tasks: int
    success_rate: float
    mean_trajectory_efficiency: float

    def to_dict(self) -> dict:
        return {
            "n_tasks": self.n_tasks,
            "success_rate": self.success_rate,
            "mean_trajectory_efficiency": self.mean_trajectory_efficiency,
        }


@dataclass
class SuiteReport:
    """Aggregate scores across a task suite (sec.16 reporting)."""

    n_tasks: int
    task_success_rate: float
    mean_trajectory_efficiency: float
    mean_sketch_editability: float
    per_difficulty: dict  # difficulty -> DifficultyReport
    results: List[TaskResult] = field(default_factory=list)
    # Added metric families (None when no task in the suite supplies the input):
    #   program_execution_rate     -> suite-wide op-stream rebuild success rate.
    #   mean_cad_sequence_f1       -> mean CAD F1 over tasks that carry ref ops.
    #   mean_assembly_mate_accuracy-> mean mate-type accuracy over assembly tasks.
    #   collision_rate             -> fraction of multi-part assemblies colliding.
    program_execution_rate: Optional[float] = None
    mean_cad_sequence_f1: Optional[float] = None
    mean_assembly_mate_accuracy: Optional[float] = None
    collision_rate: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "n_tasks": self.n_tasks,
            "task_success_rate": self.task_success_rate,
            "mean_trajectory_efficiency": self.mean_trajectory_efficiency,
            "mean_sketch_editability": self.mean_sketch_editability,
            "program_execution_rate": self.program_execution_rate,
            "mean_cad_sequence_f1": self.mean_cad_sequence_f1,
            "mean_assembly_mate_accuracy": self.mean_assembly_mate_accuracy,
            "collision_rate": self.collision_rate,
            "per_difficulty": {
                d: r.to_dict() for d, r in self.per_difficulty.items()},
            "results": [r.to_dict() for r in self.results],
        }


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_suite(tasks: List[Task],
              backend_factory: BackendFactory,
              session_factory: Optional[SessionFactory] = None,
              solver: Optional[Solver] = None) -> SuiteReport:
    """Run every task and aggregate a suite report with per-difficulty buckets."""
    results = [run_task(t, backend_factory, session_factory, solver) for t in tasks]

    per_difficulty: dict = {}
    for difficulty in DIFFICULTIES:
        bucket = [r for r in results if r.difficulty == difficulty]
        if not bucket:
            continue
        per_difficulty[difficulty] = DifficultyReport(
            n_tasks=len(bucket),
            success_rate=_mean([1.0 if r.success else 0.0 for r in bucket]),
            mean_trajectory_efficiency=_mean(
                [r.trajectory_efficiency for r in bucket]),
        )

    # Optional aggregates — averaged only over tasks that reported the metric,
    # and left None when the whole suite lacks it (e.g. the single-part stub
    # suite reports no assembly, so collision/mate stay None).
    f1s = [r.cad_sequence_f1["f1"] for r in results
           if r.cad_sequence_f1 is not None]
    mates = [r.assembly_mate_accuracy["mate_type_accuracy"] for r in results
             if r.assembly_mate_accuracy is not None]
    collisions = [r.collision_rate for r in results
                  if r.collision_rate is not None]

    return SuiteReport(
        n_tasks=len(results),
        task_success_rate=_mean([1.0 if r.success else 0.0 for r in results]),
        mean_trajectory_efficiency=_mean(
            [r.trajectory_efficiency for r in results]),
        mean_sketch_editability=_mean([r.sketch_editability for r in results]),
        program_execution_rate=program_execution_rate(results),
        mean_cad_sequence_f1=_mean(f1s) if f1s else None,
        mean_assembly_mate_accuracy=_mean(mates) if mates else None,
        collision_rate=_mean(collisions) if collisions else None,
        per_difficulty=per_difficulty,
        results=results,
    )
