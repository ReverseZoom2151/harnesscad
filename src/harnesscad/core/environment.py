"""Environment — the interface an agent acts against, and what it HONESTLY offers.

Why this exists
---------------
Every backend in :mod:`harnesscad.io.backends` satisfies four contracts that the
harness leans on everywhere:

1. ``state_digest()`` is a **content digest** — replaying the same ops yields the
   same string (deterministic replay).
2. ``apply()`` **rejects without mutating** (block-and-correct).
3. ``query()`` is a **synchronous structured read** of the model.
4. ``export()`` hands back real geometry.

A *GUI* satisfies none of the first three. A CAD application has no content hash
of its document; a dialog that refuses a value has already moved focus, opened a
panel, maybe half-created a feature; and there is no synchronous structured read
of a running Qt app — only an accessibility tree and a screenshot.

The tempting move is to force the GUI in behind :class:`GeometryBackend` anyway
and let it *fabricate* a digest. That is exactly the class of silent lie the whole
project exists to eradicate. So the seam moves up one level:

    Environment          <- what an agent acts against, and DECLARES its limits
      +-- BackendEnvironment   (a GeometryBackend: all capabilities True)
      +-- FreeCADGuiEnvironment (a live GUI: digest/reject/read all False)

:class:`GeometryBackend` is not changed and not wrapped in place — the six
existing backends keep working byte-for-byte. :class:`BackendEnvironment` adapts
any of them into an Environment, and reads its capability declaration from the
backend's optional ``CAPABILITIES`` attribute (default: the full kernel contract).

Stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

from harnesscad.core.cisp.ops import Op, parse_op
from harnesscad.eval.verifiers.verify import Diagnostic


class CapabilityError(RuntimeError):
    """An Environment was asked for something it declared it cannot do.

    Raised INSTEAD of returning a plausible-looking wrong answer. A GUI
    environment asked for ``state_digest()`` raises this rather than hashing a
    screenshot and calling it a content digest.
    """

    def __init__(self, env: str, capability: str, detail: str = "") -> None:
        self.env = env
        self.capability = capability
        msg = "environment '%s' does not provide '%s'" % (env, capability)
        if detail:
            msg += ": " + detail
        super().__init__(msg)


@dataclass(frozen=True)
class Capabilities:
    """What an Environment can and cannot honestly do. Declared, never inferred.

    The three load-bearing flags are the three :class:`GeometryBackend` contracts
    a GUI cannot honour. Anything that consumes an Environment (the differential
    oracle, the measured-output gate, a replay check) must consult these BEFORE
    it relies on the corresponding call — and every method below raises
    :class:`CapabilityError` when its flag is False, so a caller that forgets
    gets an exception, not a fiction.
    """

    name: str = "environment"

    #: ``state_digest()`` is a content hash of the model, stable across identical
    #: replays. False for a GUI: a running application has no such hash.
    content_digest: bool = False

    #: An op the environment refuses leaves state EXACTLY as it was
    #: (block-and-correct). False for a GUI: a rejected dialog has already
    #: opened panels, moved focus, and may have created a half-feature.
    nonmutating_reject: bool = False

    #: ``observe()``/``query()`` return structured model state synchronously.
    #: False for a GUI: reads are asynchronous, out-of-band, and may lag the
    #: application's own recompute.
    synchronous_read: bool = False

    #: Replaying the same op stream produces the same geometry.
    deterministic_replay: bool = False

    #: The environment can hand back real geometry (a STEP/STL/... payload).
    export: bool = True

    #: Formats ``export()`` accepts.
    export_formats: Tuple[str, ...] = ()

    #: Ops this environment can actually execute. EMPTY TUPLE MEANS "all" (that
    #: is what a kernel backend declares); a GUI declares its real subset.
    supported_ops: Tuple[str, ...] = ()

    #: Ops it explicitly cannot execute, with the reason. Surfaced verbatim to
    #: the agent so a refusal is actionable rather than a bare "unsupported".
    unsupported_ops: Dict[str, str] = field(default_factory=dict)

    #: Actions are resolved to a named UI element before dispatch and can
    #: therefore be REFUSED before they happen (a11y grounding). Kernel backends
    #: have no UI, so this is False and meaningless for them.
    resolve_before_act: bool = False

    #: Free-text limits, for the report and for the agent's prompt.
    notes: Tuple[str, ...] = ()

    def supports(self, op_tag: str) -> bool:
        if op_tag in self.unsupported_ops:
            return False
        if not self.supported_ops:
            return True  # empty = all
        return op_tag in self.supported_ops

    def why_not(self, op_tag: str) -> str:
        if op_tag in self.unsupported_ops:
            return self.unsupported_ops[op_tag]
        if self.supported_ops and op_tag not in self.supported_ops:
            return "op '%s' is not in this environment's supported set" % op_tag
        return ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "content_digest": self.content_digest,
            "nonmutating_reject": self.nonmutating_reject,
            "synchronous_read": self.synchronous_read,
            "deterministic_replay": self.deterministic_replay,
            "export": self.export,
            "export_formats": list(self.export_formats),
            "supported_ops": list(self.supported_ops),
            "unsupported_ops": dict(self.unsupported_ops),
            "resolve_before_act": self.resolve_before_act,
            "notes": list(self.notes),
        }


#: What every kernel :class:`GeometryBackend` declares: the full contract.
KERNEL_CAPABILITIES = Capabilities(
    name="geometry-backend",
    content_digest=True,
    nonmutating_reject=True,
    synchronous_read=True,
    deterministic_replay=True,
    export=True,
    resolve_before_act=False,
    notes=("scripted kernel: the four GeometryBackend contracts hold",),
)


@dataclass
class Observation:
    """What the agent sees. Hybrid: structured state + (optionally) pixels.

    ``digest`` is ``None`` whenever the environment does not provide a content
    digest — it is NEVER a hash of something else standing in for one.
    """

    kind: str = "structured"          # structured | hybrid | pixel
    state: Dict[str, Any] = field(default_factory=dict)
    digest: Optional[str] = None
    images: Dict[str, Any] = field(default_factory=dict)   # metadata, not bytes
    step: int = 0
    notes: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {"kind": self.kind, "state": self.state, "digest": self.digest,
                "images": self.images, "step": self.step, "notes": list(self.notes)}


@dataclass
class StepResult:
    """The outcome of one action. ``verified`` is the whole point.

    ``verified`` means the environment READ BACK evidence that the action took
    effect (a value changed, a feature appeared, a digest moved). In a
    verifier-first harness an unverified action is not an action; a GUI driver
    that trusts a Windows API return value is lying to itself (SetValue on a Qt
    spinbox returns success and does nothing).
    """

    ok: bool = False
    verified: bool = False
    observation: Optional[Observation] = None
    reward: float = 0.0
    done: bool = False
    diagnostics: List[Diagnostic] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "verified": self.verified, "reward": self.reward,
                "done": self.done,
                "observation": None if self.observation is None
                else self.observation.to_dict(),
                "diagnostics": [d.to_dict() for d in self.diagnostics],
                "info": self.info}


@runtime_checkable
class Environment(Protocol):
    """The interface an agent acts against. OpenEnv-shaped, capability-declaring."""

    def capabilities(self) -> Capabilities:
        """What this environment can and cannot honestly do."""

    def reset(self) -> Observation:
        """Discard all state; return the initial observation."""

    def step(self, action) -> StepResult:
        """Take one action (an Op / dict / sequence of them)."""

    def observe(self) -> Observation:
        """The current observation, without acting."""

    def export(self, fmt: str):
        """Hand back real geometry in ``fmt``."""

    def close(self) -> None:
        """Release any resources (processes, windows, handles)."""


def coerce_ops(action) -> List[Op]:
    """Actions -> a list of Ops. Shared by every Environment implementation."""
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
        out: List[Op] = []
        for a in action:
            out.extend(coerce_ops(a))
        return out
    raise TypeError("unsupported action type: %s" % type(action).__name__)


class BackendEnvironment:
    """Any :class:`GeometryBackend` as an :class:`Environment`. Nothing is changed
    in the backend; this is a pure adapter, so the six existing backends keep
    working exactly as they did.

    The capability declaration is read from the backend's optional
    ``CAPABILITIES`` attribute, defaulting to :data:`KERNEL_CAPABILITIES` — the
    full contract, which is what a scripted kernel genuinely provides.
    """

    def __init__(self, backend, verifiers=None, max_steps: Optional[int] = None) -> None:
        from harnesscad.core.loop import HarnessSession  # local: avoid import cycle

        self._backend = backend
        self._verifiers = verifiers
        self._session_cls = HarnessSession
        self.max_steps = max_steps
        self.session = None
        self._steps = 0
        self.reset()

    # -- Environment -------------------------------------------------------
    def capabilities(self) -> Capabilities:
        declared = getattr(self._backend, "CAPABILITIES", None)
        if isinstance(declared, Capabilities):
            return declared
        formats = tuple(getattr(self._backend, "FORMATS", ()) or ())
        unsupported = dict(getattr(self._backend, "UNSUPPORTED", {}) or {})
        return replace(KERNEL_CAPABILITIES,
                       name=type(self._backend).__name__,
                       export_formats=formats,
                       unsupported_ops=unsupported)

    def reset(self) -> Observation:
        self.session = self._session_cls(self._backend, verifiers=self._verifiers)
        self._steps = 0
        return self.observe()

    def step(self, action) -> StepResult:
        ops = coerce_ops(action)
        before = self.session.digest()
        result = self.session.apply_ops(ops)
        self._steps += 1
        after = self.session.digest()
        done = self.max_steps is not None and self._steps >= self.max_steps
        return StepResult(
            ok=bool(result.ok),
            # A kernel backend's digest IS the read-back: it moved, so the ops
            # landed. An empty op list legitimately verifies nothing.
            verified=bool(result.ok) and (after != before or not ops),
            observation=self.observe(),
            reward=1.0 if result.ok else -1.0,
            done=done,
            diagnostics=list(result.diagnostics),
            info={"applied": result.applied, "rejected": result.rejected,
                  "digest": after, "step": self._steps},
        )

    def observe(self) -> Observation:
        # An observation NEVER raises, and it never substitutes something else for
        # a digest it does not have: a non-declaring environment reports None.
        digest = self.session.digest() if self.capabilities().content_digest else None
        return Observation(
            kind="structured",
            state={
                "summary": self._backend.query("summary"),
                "sketch_dof": self._backend.query("sketch_dof"),
                "ops": [op.to_dict() for op in self.session.opdag.ops()],
            },
            digest=digest,
            step=self._steps,
        )

    def export(self, fmt: str):
        return self._backend.export(fmt)

    def close(self) -> None:
        self.session = None

    # -- capability-gated ---------------------------------------------------
    def state_digest(self) -> str:
        caps = self.capabilities()
        if not caps.content_digest:
            raise CapabilityError(caps.name, "content_digest")
        return self.session.digest()

    def query(self, q: str) -> dict:
        caps = self.capabilities()
        if not caps.synchronous_read:
            raise CapabilityError(caps.name, "synchronous_read")
        return self._backend.query(q)

    @property
    def backend(self):
        return self._backend


def require(env: Environment, *capabilities: str) -> None:
    """Assert an Environment declares every named capability, or raise.

    The call a consumer makes BEFORE it relies on a contract. The differential
    oracle requires ``content_digest`` of its reference side and nothing of the
    side under test — which is precisely why the GUI can be tested against the
    kernel and not the other way round.
    """
    caps = env.capabilities()
    for cap in capabilities:
        if not getattr(caps, cap, False):
            raise CapabilityError(caps.name, cap)


def supported_subset(caps: Capabilities, ops: Sequence[Op]) -> Tuple[List[Op], List[str]]:
    """Split an op stream into (ops this env supports, reasons for the rest)."""
    ok: List[Op] = []
    reasons: List[str] = []
    for op in ops:
        tag = getattr(type(op), "OP", "")
        if caps.supports(tag):
            ok.append(op)
        else:
            reasons.append(caps.why_not(tag))
    return ok, reasons
