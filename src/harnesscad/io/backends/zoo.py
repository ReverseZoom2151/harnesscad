"""ZooBackend -- CISP ops driven through Zoo (KittyCAD): KCL + the engine API.

Zoo (zoo.dev, formerly KittyCAD) ships three things a HarnessCAD backend cares
about, and this module is the seam onto all three:

1.  **KCL** -- the KittyCAD Language, a code-CAD language. Lowering a CISP op
    stream to a ``.kcl`` program is fully *offline and deterministic* and needs
    no API key. That lowering lives in :mod:`harnesscad.io.formats.kcl`; this
    backend composes it, so ``ZooBackend.export("kcl")`` yields a program.
2.  **The geometry engine API** -- Zoo's server tessellates KCL / modeling
    commands into meshes and B-reps and exports them (STL/OBJ/STEP/glTF/PLY...).
    This is *live* and needs ``ZOO_API_TOKEN`` (or ``KITTYCAD_API_TOKEN``). The
    Python SDK is ``kittycad`` (``KittyCAD().ml`` / ``.file`` / modeling
    websocket). Absent a key -- or absent the SDK -- every live path SKIPs
    cleanly: :meth:`export` of a *mesh* format raises
    :class:`~harnesscad.io.backends.base.BackendUnavailable`, never a bogus part.
3.  **The text-to-CAD API** -- ``client.ml.create_text_to_cad(prompt=...)``: Zoo's
    own prompt->CAD model. That is a *competitor's oracle*, and the most
    interesting integration point: their output, measured by OUR gate. The
    comparator is designed in :class:`ZooTextToCadComparator` below (code +
    docstring) but never calls the live API here.

The security contract is absolute
---------------------------------
This module NEVER handles, prints, logs, writes, or accepts as an argument an API
token. The token is read from the environment *only*, and only inside
:func:`_read_token`, which returns a bool-ish "present?" to callers and hands the
raw value to nobody but the SDK client constructor. A token is a secret; the code
treats it as one.

Offline vs live, at a glance
----------------------------
======================  ==============================================
``apply`` / ``query``   offline (op-state model, reused from the stub)
``export("kcl")``       offline, deterministic (the KCL emitter)
``state_digest``        offline (hash of the op stream + the KCL text)
``export("stl"|...)``   LIVE -- needs a key; SKIPs (BackendUnavailable) without
``text_to_cad(...)``    LIVE -- needs a key; SKIPs without
======================  ==============================================

stdlib-only for everything offline; the ``kittycad`` SDK is import-guarded and
only ever touched on a live path.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from harnesscad.core.cisp.ops import Op, canonical_json
from harnesscad.eval.verifiers.verify import Diagnostic
from harnesscad.io.backends.base import ApplyResult, BackendUnavailable
from harnesscad.io.backends.stub import StubBackend
from harnesscad.io.formats import kcl as kcl_codec

__all__ = [
    "ZooBackend",
    "ZooTextToCadComparator",
    "TOKEN_ENV_VARS",
    "token_present",
    "MESH_EXPORT_FORMATS",
]

#: The environment variables a Zoo token may live in, in priority order. The
#: canonical HarnessCAD name first, then the SDK's own name.
TOKEN_ENV_VARS: Tuple[str, ...] = ("ZOO_API_TOKEN", "KITTYCAD_API_TOKEN")

#: Formats the live Zoo engine exports a KCL model to. (The engine converts to
#: many more on the file-conversion endpoint; these are the model-export ones.)
MESH_EXPORT_FORMATS: Tuple[str, ...] = ("stl", "obj", "ply", "gltf", "glb",
                                        "fbx", "step")


def _read_token() -> Optional[str]:
    """Return the token from the environment, or None. The ONLY reader of it.

    The raw value is never returned to general callers (see :func:`token_present`
    for the public, value-free probe) and is only ever passed on to the SDK
    client. It is never logged, written, or placed in a digest.
    """
    for name in TOKEN_ENV_VARS:
        value = os.environ.get(name)
        if value:
            return value
    return None


def token_present() -> bool:
    """True iff a Zoo token is set in the environment. Reveals nothing about it."""
    return _read_token() is not None


def _import_sdk():
    """Import the ``kittycad`` SDK, or None if it is not installed. Guarded."""
    try:
        import kittycad  # noqa: F401

        return kittycad
    except Exception:  # noqa: BLE001 - any import failure means 'unavailable'
        return None


class ZooBackend:
    """A GeometryBackend that emits KCL offline and reaches Zoo's engine live.

    The op-state model (references, DOF, block-and-correct, replay) is delegated
    to a composed :class:`StubBackend`, so this backend inherits exactly the same
    op admission the rest of the harness is tested against -- it adds the KCL
    emission and the (key-gated) engine bridge on top, and does not re-implement
    op validation.
    """

    #: Advertised so tooling can see this backend is a code-CAD (KCL) emitter.
    LANGUAGE = "kcl"

    def __init__(self, *, length_unit: str = "mm", kcl_version: str = "1.0",
                 name: str = "model") -> None:
        self.length_unit = length_unit
        self.kcl_version = kcl_version
        self.name = name
        self._state = StubBackend()

    # -- op state (delegated) --------------------------------------------
    def reset(self) -> None:
        self._state.reset()

    def apply(self, op: Op) -> ApplyResult:
        return self._state.apply(op)

    def regenerate(self) -> List[Diagnostic]:
        return self._state.regenerate()

    def query(self, q: str) -> dict:
        return self._state.query(q)

    @property
    def _oplog(self) -> List[Op]:
        """The successfully-applied op stream (what the KCL emitter reads)."""
        return self._state._oplog

    # -- KCL emission (offline, deterministic) ---------------------------
    def to_kcl(self) -> str:
        """The current model as a deterministic ``.kcl`` program. No key needed."""
        return kcl_codec.emit_kcl(
            self._oplog, name=self.name, length_unit=self.length_unit,
            kcl_version=self.kcl_version)

    # -- export ----------------------------------------------------------
    def export(self, fmt: str):
        """Export the model. ``kcl`` is offline; mesh/brep formats are LIVE.

        ``export("kcl")`` runs the deterministic emitter and needs no key.
        ``export("stl"|"obj"|"step"|...)`` needs the Zoo engine: without a token
        (or without the SDK) it raises :class:`BackendUnavailable`, so callers
        SKIP rather than receive a fabricated mesh.
        """
        f = str(fmt).lower()
        if f == "kcl":
            return self.to_kcl()
        if f in MESH_EXPORT_FORMATS:
            return self._export_via_engine(f)
        raise BackendUnavailable(
            "zoo", "the zoo backend exports 'kcl' offline, or %s via the live "
            "engine; %r is neither" % (", ".join(MESH_EXPORT_FORMATS), fmt))

    def _export_via_engine(self, fmt: str):
        """LIVE: tessellate the KCL on Zoo's engine and return the exported bytes.

        Gated on a token AND the SDK. This method builds the KCL offline and would
        hand it to the engine's KCL-execute/export endpoint; with no key it SKIPs.
        The live transport is intentionally the only part not exercised offline.
        """
        token = _read_token()
        if token is None:
            raise BackendUnavailable(
                "zoo",
                "the Zoo engine export needs an API token in %s; none is set, so "
                "this live path is skipped (KCL emission via export('kcl') is "
                "offline and always available)" % " or ".join(TOKEN_ENV_VARS),
                searched=list(TOKEN_ENV_VARS))
        sdk = _import_sdk()
        if sdk is None:
            raise BackendUnavailable(
                "zoo",
                "the 'kittycad' SDK is not installed, so the live Zoo engine "
                "cannot be reached; install it (`pip install kittycad`) to export "
                "meshes. KCL emission (export('kcl')) does not need it.",
                searched=["kittycad"])
        # --- live path (only reached with a real key + SDK) ---------------
        # Deliberately not executed by the offline test suite. The KCL program is
        # what we would submit; the SDK client reads the token from the env on its
        # own (KittyCAD()), so we never pass the secret ourselves.
        program = self.to_kcl()
        return self._submit_kcl_to_engine(sdk, program, fmt)  # pragma: no cover

    def _submit_kcl_to_engine(self, sdk, program: str, fmt: str):  # pragma: no cover
        """The one live call. Isolated so the offline suite never enters it.

        Uses the SDK's own environment-based auth (``KittyCAD()`` reads
        ``KITTYCAD_API_TOKEN``); this code passes no token. The exact endpoint is
        the engine's KCL-execute-and-export -- see the SDK's modeling/file API.
        """
        raise BackendUnavailable(
            "zoo",
            "live Zoo engine submission is intentionally not wired in this "
            "offline build; the KCL program is ready (export('kcl')). Wire "
            "kittycad's modeling/export endpoint here behind a real key.")

    # -- digest ----------------------------------------------------------
    def state_digest(self) -> str:
        """Content hash of the op stream AND the KCL it emits. Deterministic.

        The KCL text is in the digest because it is the artifact this backend is
        responsible for: two op streams that emit different programs must hash
        differently, and the same stream must always hash the same. The API token
        is NEVER part of the digest (it is a secret, and would also make the hash
        non-reproducible).
        """
        try:
            program = self.to_kcl()
        except Exception:  # noqa: BLE001 - an unemittable stream still has a digest
            program = "<unemittable>"
        blob = json.dumps({
            "backend": "zoo",
            "language": self.LANGUAGE,
            "length_unit": self.length_unit,
            "kcl_version": self.kcl_version,
            "oplog": [canonical_json(o) for o in self._oplog],
            "kcl": program,
        }, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Text-to-CAD comparator -- DESIGN ONLY (no live call here)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ComparatorPlan:
    """A resolved, inert plan for benchmarking Zoo's text-to-CAD via our gate.

    It records WHAT would run without running it: the prompt, the output format
    we would ask Zoo for, and whether the live path is available (a key + SDK are
    present). Pure data, safe to build offline.
    """

    prompt: str
    output_format: str
    live_available: bool
    reason: str


class ZooTextToCadComparator:
    """Benchmark Zoo's text-to-CAD output against the HarnessCAD gate + oracle.

    THE IDEA
    --------
    Zoo has its own prompt->CAD model (``client.ml.create_text_to_cad``). It is a
    direct comparator for HarnessCAD's own text-to-CAD. The valuable experiment is
    not "does Zoo's model run" but "how does a competitor's output score on OUR
    measuring stick" -- because our stick (the output gate + the differential
    oracle) is exactly what we trust for our own parts. Running a rival's output
    through it is an apples-to-apples benchmark.

    THE PIPELINE (each stage already exists in this repo)
    -----------------------------------------------------
    1.  **Submit the prompt to Zoo.** ``client.ml.create_text_to_cad(
        output_format=FileExportFormat.STEP, body=TextToCadCreateBody(
        prompt=...))`` returns a ``TextToCad`` async operation; poll it (the SDK's
        ``wait_for_async_operation``) until ``completed``/``failed``. On success
        the geometry bytes are in ``outputs`` (and, notably, the generated KCL is
        in ``.code`` -- a second, source-level comparison point). The
        request/response contract is already modelled, offline and key-free, in
        :mod:`harnesscad.io.adapters.zoo_api` (submit/poll descriptors, terminal
        status semantics, the ``source.{format}`` output key).
    2.  **Land the bytes as a neutral mesh.** Ask Zoo for a mesh format (STL/OBJ/
        PLY) and parse it with the existing codec
        (:mod:`harnesscad.io.formats.registry`), or ask for STEP and read it with
        the STEP codec. Either way the result is the same neutral geometry the
        rest of the harness measures.
    3.  **Run it through OUR gate.** Hand that geometry to
        :func:`harnesscad.io.gate.check`. The MEASURED family scores it with no
        intent needed: watertight, 2-manifold, consistently wound, positive
        non-degenerate volume, outward normals, plausible bbox, no self-
        intersection. A competitor's part that is open or self-intersecting fails
        our gate exactly as ours would -- that is the point.
    4.  **Differential oracle vs our own output.** For the same prompt, build the
        HarnessCAD part and measure BOTH with :func:`harnesscad.io.gate.measure`,
        then compare the invariants the gate treats as ground truth (volume within
        tolerance, bbox extents, genus/Euler characteristic). Agreement is
        evidence both solved the brief; a divergence localises which invariant one
        got wrong. The oracle is many-to-one (the gate says so in
        ``DOES_NOT_PROVE``), so this bounds, not proves, equivalence -- and the
        report says so.
    5.  **Cross-check with Zoo's OWN mass-properties endpoints.** Zoo exposes
        ``client.file.create_file_volume`` / ``create_file_mass`` /
        ``create_file_center_of_mass``. Feeding Zoo's own STEP back to those gives
        a volume computed by THEIR kernel; comparing it with the volume our gate
        measured off THEIR mesh is a kernel-vs-mesh consistency check on a single
        part -- a conversion/measurement oracle that needs no ground-truth model.

    WHY NOTHING RUNS HERE
    ---------------------
    Every step above is live (Zoo engine + ML model) and needs a key. This class
    only *plans* the run: :meth:`plan` resolves the prompt and availability and
    returns an inert :class:`ComparatorPlan`. The actual submission is left to a
    live harness that a human runs deliberately, with a key, outside the offline
    test suite. (And per the build's hard rule, no live Zoo API call and no LLM
    run happens here.)
    """

    #: STEP is the default benchmark format: a B-rep our STEP codec reads and
    #: Zoo's mass-properties endpoints also accept, so both oracles apply to it.
    DEFAULT_FORMAT = "step"

    def __init__(self, output_format: str = DEFAULT_FORMAT) -> None:
        self.output_format = str(output_format).lower()

    def plan(self, prompt: str) -> ComparatorPlan:
        """Resolve an inert plan for one prompt. Never calls the network."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        have_key = token_present()
        have_sdk = _import_sdk() is not None
        if have_key and have_sdk:
            reason = "a key and the kittycad SDK are present; a live run is possible"
        elif not have_key:
            reason = ("no Zoo token in the environment (%s): the live benchmark is "
                      "skipped" % " or ".join(TOKEN_ENV_VARS))
        else:
            reason = "the kittycad SDK is not installed: the live benchmark is skipped"
        return ComparatorPlan(
            prompt=prompt.strip(),
            output_format=self.output_format,
            live_available=have_key and have_sdk,
            reason=reason,
        )

    def score_geometry(self, model, *, source=None, path: Optional[str] = None):
        """Score ALREADY-FETCHED geometry with our gate. Offline, no network.

        This is the measuring half of the comparator and is fully offline: given
        geometry that a live run produced (a mesh or a STEP payload), it returns a
        :class:`harnesscad.io.gate.GateReport`. It performs no I/O of its own, so
        it is safe to unit-test with a synthetic mesh.
        """
        from harnesscad.io import gate

        return gate.check(model, path, source=source)
