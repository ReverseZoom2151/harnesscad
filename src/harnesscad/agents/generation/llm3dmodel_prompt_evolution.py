"""Error-driven prompt evolution loop (Kumar et al., "Generative AI for CAD
Automation: Leveraging Large Language Models for 3D Modelling", 2025, sec. 2).

The paper drives a stateless LLM (GPT-4) around a FreeCAD executor with an
*error-driven prompt evolution* controller.  The LLM itself and FreeCAD are
external; everything that surrounds them is a deterministic protocol the paper
specifies exactly and which this module re-implements with injectable
``generate_fn`` / ``execute_fn`` callables:

  * Structured initial prompt P_i (sec. 2.1): the paper's PromptTemplate that
    always injects the required scaffolding lines (imports, ``Part.show()``,
    ``FreeCAD.ActiveDocument.recompute()``, object validation).  Because LLM
    APIs are stateless, the *whole* context is rebuilt on every call.
  * Execution E = F(S) (Eq. 3): ``E == ""`` means success, any non-empty string
    is the error.  ``execute_fn(script) -> (stdout, stderr)``.
  * Terminal-log packaging (sec. 2.4): the refined prompt bundles the original
    request, the initial prompt, the *last script*, and the combined terminal
    log (stdout for flow context + stderr for the specific error).
  * Refined prompt P_r^{t+1} = f(P_i, E^{t}) (Eq. 4): error-derived constraints
    are accumulated so each retry is strictly more constrained than the last.
  * Iterative loop S^{t+1} = G(P_r^{t+1}, d, S^{t}, E^{t}) (Eq. 5) terminating
    when E == 0 or t >= T (Eq. 7), failing *gracefully* by logging on exhaustion.

Stdlib only, deterministic, no wall clock, no LLM.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

# The scaffolding lines the paper's PromptTemplate guarantees are present in
# every generated script (sec. 2.1 / "LLM with GPT-4 API and LangChain").
REQUIRED_SCAFFOLD: Tuple[str, ...] = (
    "import FreeCAD",
    "import Part",
    "Part.show(",
    "FreeCAD.ActiveDocument.recompute(",
)


def build_initial_prompt(description: str,
                         constraints: Optional[List[str]] = None) -> str:
    """Structured initial prompt P_i (sec. 2.1).

    Combines the user description ``d`` with the fixed FreeCAD scripting
    requirements and any caller constraints.  The output is deterministic and
    always lists the required scaffolding so the executor's checks can pass.
    """
    if not description or not description.strip():
        raise ValueError("description must be non-empty")
    lines = [
        "You are a FreeCAD Python scripting assistant.",
        "Generate a headless FreeCAD script for the design below.",
        "Design description:",
        f"  {description.strip()}",
        "Scripting requirements:",
    ]
    for req in REQUIRED_SCAFFOLD:
        lines.append(f"  - must include: {req}")
    for c in constraints or ():
        lines.append(f"  - constraint: {c}")
    return "\n".join(lines)


def combine_terminal_log(stdout: str, stderr: str) -> str:
    """Terminal log = stdout (flow context) + stderr (error), sec. 2.4."""
    parts = []
    if stdout and stdout.strip():
        parts.append("STDOUT:\n" + stdout.strip())
    if stderr and stderr.strip():
        parts.append("STDERR:\n" + stderr.strip())
    return "\n".join(parts)


def refine_prompt(initial_prompt: str, description: str, last_script: str,
                  terminal_log: str, accumulated: List[str]) -> str:
    """Refined prompt P_r^{t+1} = f(P_i, E^{t}) (Eq. 4).

    Rebuilds the full stateless context: the original request, the last script,
    the terminal log, and every constraint accumulated so far.  ``accumulated``
    is the running list of error-derived constraints (monotonically growing).
    """
    lines = [initial_prompt, "", "The previous script failed; fix it minimally.",
             "Original request:", f"  {description.strip()}",
             "Previous script:", last_script.rstrip(),
             "Terminal log:", terminal_log.rstrip()]
    if accumulated:
        lines.append("Accumulated constraints from earlier failures:")
        for c in accumulated:
            lines.append(f"  - {c}")
    return "\n".join(lines)


def constraint_from_error(terminal_log: str) -> str:
    """Derive one additional prompt constraint from a terminal log (Eq. 4 f).

    Deterministic keyword mapping so each retry is *more* constrained than the
    last, matching the paper's failure modes (unsupported API, null shape,
    overconstraint, syntax).
    """
    low = terminal_log.lower()
    if "has no attribute" in low or "unsupported" in low:
        return "do not call unsupported FreeCAD APIs; use documented Part methods only"
    if "null shape" in low:
        return "ensure every operation yields a non-null shape before the next step"
    if "overconstrain" in low or "over-constrain" in low:
        return "avoid redundant constraints that overconstrain the sketch"
    if "syntaxerror" in low or "indentation" in low:
        return "produce syntactically valid, correctly indented Python"
    if "boolean" in low or "degenerate" in low:
        return "verify boolean operands overlap and are valid solids"
    return "address the reported terminal error before regenerating"


@dataclass
class EvolutionStep:
    """One iteration record of the refinement loop."""
    iteration: int
    script: str
    stderr: str
    ok: bool


@dataclass
class EvolutionResult:
    """Outcome of the full error-driven prompt-evolution loop."""
    script: str
    converged: bool
    iterations: int
    steps: List[EvolutionStep] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    log: List[str] = field(default_factory=list)


def evolve(description: str,
           generate_fn: Callable[[str], str],
           execute_fn: Callable[[str], Tuple[str, str]],
           *,
           max_retries: int,
           initial_constraints: Optional[List[str]] = None) -> EvolutionResult:
    """Run the error-driven prompt-evolution loop (Eqs. 1-7).

    ``generate_fn(prompt) -> script`` stands in for G (the LLM); ``execute_fn(
    script) -> (stdout, stderr)`` stands in for F (headless FreeCAD).  Empty
    ``stderr`` == success (E = 0).  The loop runs the initial generation plus up
    to ``max_retries`` refinements (t >= T terminates, Eq. 7), failing
    gracefully by recording the last error.
    """
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")
    p_initial = build_initial_prompt(description, initial_constraints)
    accumulated: List[str] = list(initial_constraints or ())
    steps: List[EvolutionStep] = []
    log: List[str] = []

    script = generate_fn(p_initial)
    stdout, stderr = execute_fn(script)
    ok = not (stderr and stderr.strip())
    steps.append(EvolutionStep(0, script, stderr, ok))
    log.append("iter 0: " + ("success" if ok else "error"))

    t = 0
    while not ok and t < max_retries:
        t += 1
        term = combine_terminal_log(stdout, stderr)
        new_c = constraint_from_error(term)
        if new_c not in accumulated:
            accumulated.append(new_c)
        prompt = refine_prompt(p_initial, description, script, term, accumulated)
        script = generate_fn(prompt)
        stdout, stderr = execute_fn(script)
        ok = not (stderr and stderr.strip())
        steps.append(EvolutionStep(t, script, stderr, ok))
        log.append(f"iter {t}: " + ("success" if ok else "error"))

    if not ok:
        log.append(f"graceful failure after {t} refinements: {stderr.strip()}")
    return EvolutionResult(script=script, converged=ok, iterations=t,
                           steps=steps, constraints=accumulated, log=log)
