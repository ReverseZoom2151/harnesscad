"""Deterministic orchestration for optional external simulation solvers.

This module deliberately owns no threads or processes.  Callers inject an
executor, clock and solver, which makes the state machine usable from a
synchronous CLI, an async worker, or a test without changing its semantics.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, MutableMapping, Optional, Protocol


class JobState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


TERMINAL_STATES = frozenset(
    {JobState.SUCCEEDED, JobState.FAILED, JobState.TIMED_OUT, JobState.CANCELLED}
)


@dataclass(frozen=True)
class SolverProvenance:
    """Identity needed to reproduce and safely cache a solver result."""

    name: str
    version: str
    backend: str = "external"
    executable_digest: str = ""
    configuration: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class SimulationJob:
    id: str
    cache_key: str
    payload: Mapping[str, Any]
    provenance: SolverProvenance
    timeout_s: float
    max_retries: int
    state: JobState = JobState.PENDING
    attempt: int = 0
    created_at: float = 0.0
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result: Any = None
    error: Optional[str] = None
    cache_hit: bool = False
    cancel_requested: bool = False

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES


class Solver(Protocol):
    """Duck-typed solver seam, compatible with FEASolver-style adapters."""

    def solve(self, payload: Mapping[str, Any]) -> Any: ...


@dataclass(frozen=True)
class FEASolverAdapter:
    """Adapt ``verifiers.simulation.FEASolver.solve(mesh, load_case)`` to a job.

    The dependency remains duck typed to avoid importing the verifier module.
    """

    solver: Any

    def solve(self, payload: Mapping[str, Any]) -> Any:
        return self.solver.solve(payload["mesh"], payload["load_case"])


Executor = Callable[[Callable[[], Any], float], Any]
Clock = Callable[[], float]


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"simulation payload is not JSON-serializable: {type(value).__name__}")


def content_key(payload: Mapping[str, Any], provenance: SolverProvenance) -> str:
    """Hash inputs and solver identity; equivalent inputs share a cache entry."""

    document = {"payload": _canonical(payload), "solver": _canonical(asdict(provenance))}
    raw = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def direct_executor(call: Callable[[], Any], timeout_s: float) -> Any:
    """Run inline. Production callers can inject a timeout-enforcing executor."""

    del timeout_s
    return call()


class SimulationJobs:
    """Small, deterministic job registry with retry, cancellation and caching."""

    def __init__(
        self,
        *,
        clock: Clock,
        executor: Executor = direct_executor,
        cache: Optional[MutableMapping[str, Any]] = None,
    ) -> None:
        self._clock = clock
        self._executor = executor
        self.cache: MutableMapping[str, Any] = cache if cache is not None else {}
        self.jobs: dict[str, SimulationJob] = {}
        self._sequence = 0

    def submit(
        self,
        payload: Mapping[str, Any],
        provenance: SolverProvenance,
        *,
        timeout_s: float = 300.0,
        max_retries: int = 0,
    ) -> SimulationJob:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        key = content_key(payload, provenance)
        self._sequence += 1
        job = SimulationJob(
            id=f"sim-{self._sequence:06d}",
            cache_key=key,
            payload=_canonical(payload),
            provenance=provenance,
            timeout_s=float(timeout_s),
            max_retries=int(max_retries),
            created_at=self._clock(),
        )
        if key in self.cache:
            job.state = JobState.SUCCEEDED
            job.result = self.cache[key]
            job.finished_at = self._clock()
            job.cache_hit = True
        self.jobs[job.id] = job
        return job

    def cancel(self, job_id: str) -> SimulationJob:
        job = self.jobs[job_id]
        if job.terminal:
            return job
        job.cancel_requested = True
        if job.state is JobState.PENDING:
            job.state = JobState.CANCELLED
            job.finished_at = self._clock()
        return job

    def run(self, job_id: str, solver: Solver) -> SimulationJob:
        """Run or resume a job. Exceptions are recorded, never leaked."""

        job = self.jobs[job_id]
        if job.terminal:
            return job
        if job.cancel_requested:
            job.state = JobState.CANCELLED
            job.finished_at = self._clock()
            return job

        while job.attempt <= job.max_retries:
            job.state = JobState.RUNNING
            job.attempt += 1
            job.started_at = self._clock()
            try:
                result = self._executor(lambda: solver.solve(job.payload), job.timeout_s)
            except TimeoutError as exc:
                job.error = str(exc) or "solver timed out"
                job.state = JobState.TIMED_OUT
            except Exception as exc:  # solver boundary: normalize failures
                job.error = f"{type(exc).__name__}: {exc}"
                job.state = JobState.FAILED
            else:
                if job.cancel_requested:
                    job.state = JobState.CANCELLED
                else:
                    job.state = JobState.SUCCEEDED
                    job.result = result
                    job.error = None
                    self.cache[job.cache_key] = result
                job.finished_at = self._clock()
                return job

            if job.attempt > job.max_retries:
                job.finished_at = self._clock()
                return job
            job.state = JobState.PENDING

        return job

    def run_now(
        self,
        payload: Mapping[str, Any],
        provenance: SolverProvenance,
        solver: Solver,
        **options: Any,
    ) -> SimulationJob:
        """Synchronous convenience API; async workers can use submit/run."""

        return self.run(self.submit(payload, provenance, **options).id, solver)
