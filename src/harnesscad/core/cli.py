"""harnesscad CLI — drive a HarnessSession from the command line.

    python cli.py apply <ops.json> [--backend stub|cadquery|frep]
        Load a JSON array of ops, run them through a session, and pretty-print
        the ApplyOpsResult (ok, applied, digest, diagnostics).

    python cli.py demo [--backend stub|cadquery|frep]
        Run a built-in sample (constrained plate -> extrude) and print the
        result plus a model summary.

    python cli.py build "<brief>" [--backend stub|cadquery|frep] [--model NAME]
                                   [--out part.step] [--trace run.jsonl]
        Turn a natural-language brief into verified geometry via the LLM planner
        and the correction loop, then optionally write the exported STEP.

    python cli.py formats [--kind mesh|brep|csg|drawing|image] [--mode read|write]
                          [--json]
        Print the honest I/O capability matrix: every adapted codec, its
        extensions, kind, and whether it can genuinely read / write / round-trip.

    python cli.py export <out.stl> [--ops ops.json] [--backend stub|cadquery|frep]
        Run an op stream (the built-in demo when --ops is omitted) and write the
        resulting model through the format registry; the extension picks the codec.
        `export part.png` renders a shaded solid (see `render` for the options).

    python cli.py render <out.png> [--ops ops.json] [--backend stub|cadquery|frep]
                         [--view iso|front|top|side|hero] [--shading flat|smooth]
                         [--no-edges] [--width N] [--height N] [--ssaa N]
                         [--projection orthographic|perspective]
        Run an op stream (the built-in demo when --ops is omitted) and rasterise
        the model to a PNG: a z-buffered shaded solid with its feature edges drawn
        over the top -- what a CAD viewport shows. Stdlib only; no PIL, no numpy.

    python cli.py ingest <tokens.json|mesh.obj> --family deepcad|skexgen|hnc|
                                                        vitruvion|mesh
        The INGEST leg: decode an existing model back into an editable CISP op
        stream, apply it to a HarnessSession and print the ops + digest + summary.
        `--family` is mandatory and is never inferred -- the quantiser families
        are mutually incompatible (see domain/reconstruction/ingest_pipeline.py).

    python cli.py reconstruct --list [--from point_cloud] [--to primitives]
    python cli.py reconstruct --kinds | --rivals | --unadapted | --show <route>
        Browse the reconstruction route registry
        (domain/reconstruction/registry.py): every runnable inverse-leg
        capability, keyed by (input kind -> output kind), so "what can turn a
        point cloud into CAD?" has a real answer. Rival encodings (the quantiser
        families, the three B-rep graph encodings, the three canonical sketch
        orderings) are listed by name and are never blended.

    python cli.py program --list
    python cli.py program --lang cadquery|openscad|... --validate FILE
    python cli.py program --lang openscad --review FILE [--reference GOLD]
    python cli.py program --lang cadquery --emit ops.json
    python cli.py program --operations | --unadapted
        The code-CAD program surface (domain/programs/registry.py):
        parse / validate / emit / review, DISPATCHED BY LANGUAGE. `--lang` is
        mandatory and is never guessed -- CadQuery, OpenSCAD, OpenECAD, typed
        CSG and the rest are different languages, not rival implementations, and
        one language's parser is never used on another's source.

    python cli.py edit --list | --rivals | --unadapted
    python cli.py edit --params [--ops ops.json]
    python cli.py edit --set 1:w=40 [--ops ops.json] [--json]
    python cli.py edit --strategy plan_verify|refine|geometry_beam --target goal.json
        The EDIT surface (domain/editing/registry.py): the harness could build and
        ingest, but not change what it built. `--set I:PARAM=VALUE` applies a real
        parametric edit (SetParam -> deterministic rebuild) and prints the diff;
        `--strategy` runs a named edit LOOP toward a target model. The three loops
        (CADMorph plan-verify, CADReasoner refine, CADReasoner beam) are RIVALS and
        are selected by name -- their results are never averaged.

    python cli.py search --list | --rivals | --unadapted | --space
    python cli.py search --strategy evolution --target goal.json [--seed N] [--json]
        The SEARCH surface (agents/exploration/registry.py): run a real design
        search over the session's shape parameters toward an objective. `evolution`
        (a GA over CAD programs) and `evolution_strategy` (a numeric mu,lambda ES)
        are different algorithms, as are the three samplers -- all exposed by name.

    python cli.py generate "<brief>" [--strategy dual_loop|verify_loop|...]
    python cli.py generate --list | --rivals | --retrieve "<query>"
        The GENERATION surface (agents/generation/registry.py): named strategies
        that drive a session to a solid, with a DETERMINISTIC STUB PLANNER by
        default (no LLM, no network). The three correction loops (CADSmith dual
        loop, CADCodeVerify, prompt evolution) are rivals; `--retrieve` runs the RAG
        layer over this repo's own capability index.

    python cli.py bench --list [--kind geometry|sequence|sketch|vision|...]
    python cli.py bench --suites | --rivals | --unadapted
    python cli.py bench --suite deepcad --input samples.json [--json]
        Discover the bench metric registry (harnesscad.eval.bench.registry) and run
        a named evaluation suite over a JSON file of pred/gold samples. Suites are
        explicit: rival metrics (e.g. the unit-sphere vs bounding-box Chamfer
        protocols) are never averaged together, and every reported number carries
        the name of the metric and the module that produced it.

    python cli.py report [--ops ops.json] [--brief "..."] [--extras extras.json]
    python cli.py report --list | --rivals | --unadapted [--kind geometry|sequence|...]
        The quality ANALYSIS surface (harnesscad.eval.quality.registry): run an op
        stream, then measure it -- complexity level, mass properties, canonical pose,
        intent graph, exposed parameters, anomaly score, traceability. Verifiers gate;
        these analyse. An analyser whose input the state does not carry is SKIPPED,
        never guessed, and rival analysers (three anomaly scorers, two reward
        functions) are reported side by side and never averaged.

    python cli.py dataset --pipeline text2cad --input records.json [--out data.jsonl]
    python cli.py dataset --list | --presets | --rivals | --unadapted
        The data engine (harnesscad.data.pipeline): generate/ingest -> annotate ->
        curate -> augment -> emit, over the dataengine/datagen modules. Pipelines are
        named and rival-free by construction: scale-invariant dedup and exact-token
        dedup disagree by design, so a pipeline may select only one of them.

    python cli.py selftest [--differential] [--golden] [--fleet] [--properties]
                          [--all] [--json] [--strict]
        SELF-evaluation (harnesscad.eval.selftest): the only surface here that
        points INWARD. Every other eval scores a MODEL; this one asks whether the
        HARNESS is correct. Four oracles, none of which needs a model or a label:
        DIFFERENTIAL (run one op stream on all six engines -- where they disagree,
        one of them is wrong, and no ground truth was needed to find out), GOLDEN
        (parts whose volume/bbox/genus are known in closed form, which is what
        adjudicates a disagreement), FLEET (precision/recall/F1 PER VERIFIER over a
        known-good and a known-bad corpus -- an ERROR on a washer is a false
        positive, and that metric is the one the fleet was never held to), and
        PROPERTIES (a shell must not grow the part; scaling a plan by k must scale
        its volume by k^3 -- metamorphic laws over a seeded random corpus).

    python cli.py gallery --list [--json]
    python cli.py gallery --build [--out assets/gallery] [--only NAME] [--no-compare]
        The RENDERED PARTS GALLERY (harnesscad.eval.gallery): sixteen distinct
        parts -- counterbored bracket, shelled enclosure + lid, involute spur gear,
        ISO-threaded bolt, gyroid TPMS lattice, SDF smooth blend, revolved pulley,
        swept duct, coil spring, patterned heatsink + flange, fillet vs chamfer,
        three-arc cam, spiral flexure -- each naming the capability module it
        exercises and the backends that can (and provably cannot) build it. Every
        PNG is decoded back and QC'd (variance, silhouette coverage, colour count)
        before it is shipped. `--build` also renders the bracket on all four
        kernels (compare-<backend>.png) so a kernel-free SDF mesh and a real B-rep
        are side by side.

    python cli.py capabilities --list [--tag X] [--layer Y] [--package Z]
    python cli.py capabilities --search TEXT
    python cli.py capabilities --show harnesscad.domain.geometry.sdf.primitives
    python cli.py capabilities --stats | --orphans | --rebuild
        Browse the static capability registry (harnesscad.registry): every product
        module, its tags, docstring summary and public symbols, indexed by AST
        without importing anything.

Exit code is non-zero when the resulting model is not ok, so the CLI composes
in scripts / CI (`cli.py apply plan.json && next-step`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from harnesscad.io.formats import registry as formats
from harnesscad.io.surfaces.server import BACKENDS, CISPServer

# The `--backend` choices are the server's BACKENDS, not a second hand-kept list:
# adding a backend in one place must not be able to leave the CLI behind.
# `blender`, `openscad` and `freecad` shell out to a real external kernel; when
# that binary is absent they degrade to the stub and say so, so the choice is
# always offerable. `cadquery` and `freecad` are the real B-rep kernels.
BACKEND_CHOICES = list(BACKENDS)

#: Loop strategies `harnesscad build --strategy` offers. Mirrored (and asserted
#: identical, tests/core/test_cli_strategy.py) from `core.pipeline.STRATEGIES`,
#: which is imported lazily so `apply`/`demo` never pay for the planner stack.
BUILD_STRATEGIES = ("refine", "best-of-n")


# The deterministic sample: sk1 = first sketch, e1 = first entity (rectangle).
DEMO_OPS: List[dict] = [
    {"op": "new_sketch", "plane": "XY"},
    {"op": "add_rectangle", "sketch": "sk1", "x": 0.0, "y": 0.0, "w": 20.0, "h": 10.0},
    {"op": "constrain", "kind": "distance", "a": "e1", "value": 20.0},
    {"op": "constrain", "kind": "distance", "a": "e1", "value": 10.0},
    {"op": "constrain", "kind": "distance", "a": "e1", "value": 20.0},
    {"op": "constrain", "kind": "distance", "a": "e1", "value": 10.0},
    {"op": "extrude", "sketch": "sk1", "distance": 5.0},
]


def _print_result(result: dict) -> None:
    print(f"ok:       {result['ok']}")
    print(f"applied:  {result['applied']}")
    print(f"digest:   {result['digest']}")
    diags = result.get("diagnostics") or []
    if diags:
        print("diagnostics:")
        for d in diags:
            where = f" @{d['where']}" if d.get("where") else ""
            print(f"  [{d['severity']}] {d['code']}: {d['message']}{where}")
    else:
        print("diagnostics: (none)")
    if result.get("rejected"):
        print(f"rejected:  {json.dumps(result['rejected'])}")


def _print_refusal(exc: "Exception") -> None:
    """Print an output-gate refusal. A refusal is the gate WORKING, so say so.

    Exit code 3 is reserved for it: 0 ok, 1 the model failed to build, 2 a usage
    or I/O error, 3 the harness built something and refused to ship it.
    """
    report = getattr(exc, "report", None)
    print("REFUSED:  the output gate rejected this artifact. No file was written.",
          file=sys.stderr)
    for f in getattr(exc, "failures", ()):
        print(f"  [{f.family}] {f.check}: {f.detail}", file=sys.stderr)
        if f.expected is not None:
            print(f"      measured {f.measured!r}, expected {f.expected!r}",
                  file=sys.stderr)
    if report is not None:
        m = report.measurement or {}
        if m.get("bbox"):
            print(f"  measured bbox={m['bbox']} volume={m.get('volume')}",
                  file=sys.stderr)
    print("  (pass --force to write it anyway; a .INVALID.json sidecar will name "
          "the failures)", file=sys.stderr)


def _run_ops(ops: List[dict], backend: str, verify: str = "core") -> dict:
    server = CISPServer(backend=backend, verify_level=verify)
    return server.applyOps(ops)


def cmd_apply(args: argparse.Namespace) -> int:
    try:
        with open(args.ops, "r", encoding="utf-8") as fh:
            ops = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: could not load ops from {args.ops!r}: {exc}", file=sys.stderr)
        return 2
    if not isinstance(ops, list):
        print(f"error: {args.ops!r} must contain a JSON array of ops", file=sys.stderr)
        return 2
    # `--verify full` runs the whole discovered verifier fleet, not just the core
    # checks. Without it the fleet was reachable only through the built-in demo,
    # which made the harness's main capability unusable on a real op stream.
    result = _run_ops(ops, args.backend, getattr(args, "verify", "core"))
    _print_result(result)
    return 0 if result["ok"] else 1


def cmd_demo(args: argparse.Namespace) -> int:
    # The demo runs the FULL verifier fleet (every check discovered by
    # harnesscad.eval.verifiers.registry), not just the three core verifiers --
    # it is the shop window for what the harness can catch. Fleet findings are
    # advisory: they are printed, but they do not fail the demo.
    level = "core" if getattr(args, "core_only", False) else "full"
    server = CISPServer(backend=args.backend, verify_level=level)
    result = server.applyOps([dict(op) for op in DEMO_OPS])
    _print_result(result)
    print("summary:", json.dumps(server.query("summary")["result"], sort_keys=True))
    return 0 if result["ok"] else 1


def cmd_build(args: argparse.Namespace) -> int:
    # Imported here so `apply`/`demo` keep working even if the pipeline module's
    # optional dependencies are missing.
    from harnesscad.core.pipeline import build, BuildError
    from harnesscad.core.trace import JsonlTracer

    tracer = JsonlTracer(args.trace) if args.trace else None
    try:
        result = build(
            args.brief,
            backend=args.backend,
            model=args.model,
            max_iters=args.max_iters,
            tracer=tracer,
            strategy=args.strategy,
            n=args.best_of,
        )
    except BuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    _print_result(result)
    print("summary:", json.dumps(result.get("summary") or {}, sort_keys=True))
    if result.get("backend_note"):
        print(f"note:     {result['backend_note']}")

    if args.out and result["ok"]:
        step = result.get("step")
        if step is None:
            print("error: build ok but no STEP was produced to write", file=sys.stderr)
            return 1
        # THE GATE. `build` producing a STEP is not the same thing as that STEP
        # being a valid part: the geometry is measured (and its declared intent
        # checked) against the session's backend before a byte is written.
        from harnesscad.io import gate

        try:
            report = gate.guard(step, args.out, source=result.get("session"),
                                force=getattr(args, "force", False))
        except gate.InvalidArtifact as exc:
            _print_refusal(exc)
            return 3
        try:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(step)
        except OSError as exc:
            print(f"error: could not write STEP to {args.out!r}: {exc}", file=sys.stderr)
            return 2
        if not report.ok:
            side = gate.write_sidecar(args.out, report)
            print(f"FORCED:   {args.out} is INVALID; see {side}")
        print(f"wrote:    {args.out}")

    return 0 if result["ok"] else 1


def cmd_formats(args: argparse.Namespace) -> int:
    """Print the honest I/O capability matrix (or its JSON)."""
    report = formats.format_report()
    if args.json:
        print(json.dumps(report, sort_keys=True, indent=2))
        return 0
    rows = report["formats"]
    if args.kind:
        rows = [r for r in rows if r["kind"] == args.kind]
    if args.mode:
        rows = [r for r in rows if r[args.mode]]
    if not rows:
        print("no formats match that filter")
        return 1
    print(formats.render_matrix(rows))
    counts = report["counts"]
    print()
    print(f"{counts['total']} formats: {counts['readable']} readable, "
          f"{counts['writable']} writable, {counts['round_trip']} round-trippable")
    for r in rows:
        if r["note"]:
            print(f"  {r['name']}: {r['note']}")
    return 0


def _ops_from(path: Optional[str]) -> List[dict]:
    """The op stream at `path`, or the built-in demo when it is None.

    Raises ValueError with a CLI-ready message when the file is unusable.
    """
    if not path:
        return [dict(op) for op in DEMO_OPS]
    try:
        with open(path, "r", encoding="utf-8") as fh:
            ops = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not load ops from {path!r}: {exc}") from exc
    if not isinstance(ops, list):
        raise ValueError(f"{path!r} must contain a JSON array of ops")
    return ops


def cmd_export(args: argparse.Namespace) -> int:
    """Run an op stream (or the demo), then write the model to <out>."""
    try:
        ops = _ops_from(args.ops)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    server = CISPServer(backend=args.backend)
    result = server.applyOps(ops)
    _print_result(result)
    if not result["ok"]:
        return 1
    from harnesscad.io import gate

    try:
        formats.write(server.session, args.out, force=args.force)
    except gate.InvalidArtifact as exc:
        _print_refusal(exc)
        return 3
    except formats.FormatError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.force and os.path.exists(gate.sidecar_path(args.out)):
        print(f"FORCED:   {args.out} is INVALID; "
              f"see {gate.sidecar_path(args.out)}")
    print(f"wrote:    {args.out}")
    return 0


def render_views() -> List[str]:
    """The camera views `render --view` accepts, taken from the renderer itself."""
    from harnesscad.io.render import VIEW_PRESETS

    return list(VIEW_PRESETS)


def cmd_render(args: argparse.Namespace) -> int:
    """Run an op stream (or the demo), then rasterise the model to a PNG."""
    # Imported here so the rasteriser is only touched by `render`/`export *.png`.
    from harnesscad.io import render as render_route

    try:
        ops = _ops_from(args.ops)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    server = CISPServer(backend=args.backend)
    result = server.applyOps(ops)
    _print_result(result)
    if not result["ok"]:
        return 1
    from harnesscad.io import gate

    try:
        render_route.render_session(
            server.session, args.out,
            view=args.view,
            width=args.width,
            height=args.height,
            shading=args.shading,
            edges=not args.no_edges,
            ssaa=args.ssaa,
            projection=args.projection,
            force=args.force,
        )
    except gate.InvalidArtifact as exc:
        _print_refusal(exc)
        return 3
    except (render_route.RenderError, formats.FormatError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.force and os.path.exists(gate.sidecar_path(args.out)):
        print(f"FORCED:   {args.out} is INVALID; "
              f"see {gate.sidecar_path(args.out)}")
    width, height = render_route.png_size(args.out)
    print(f"view:     {args.view} ({args.projection}, {args.shading}"
          f"{'' if args.no_edges else ', feature edges'})")
    print(f"wrote:    {args.out} ({width}x{height})")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    # Imported here so the reconstruction fleet is only touched by `ingest`.
    from harnesscad.domain.reconstruction.ingest_pipeline import IngestError, ingest_file

    try:
        result = ingest_file(args.source, family=args.family, backend=args.backend,
                             arc_policy=args.arc_policy)
    except IngestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: could not read {args.source!r}: {exc}", file=sys.stderr)
        return 2

    print(f"family:   {result['family']}")
    _print_result(result)
    print("ops:", json.dumps(result.get("ops") or [], sort_keys=True))
    print("summary:", json.dumps(result.get("summary") or {}, sort_keys=True))
    if result.get("note"):
        print(f"note:     {result['note']} (confidence {result.get('confidence')})")
    return 0 if result["ok"] else 1


def cmd_gallery(args: argparse.Namespace) -> int:
    # Imported here so the gallery (and the renderer it drives) is only touched
    # by `gallery`.
    from harnesscad.eval.gallery import render_gallery

    return render_gallery.run_cli(args)


def cmd_capabilities(args: argparse.Namespace) -> int:
    # Imported here so the registry (and its JSON index) is only touched when
    # the `capabilities` subcommand actually runs.
    from harnesscad import registry

    return registry.run(args)


def cmd_reconstruct(args: argparse.Namespace) -> int:
    # Imported here so the reconstruction fleet is only touched by `reconstruct`.
    from harnesscad.domain.reconstruction import registry as reconstruction_registry

    return reconstruction_registry.run_cli(args)


def cmd_program(args: argparse.Namespace) -> int:
    # Imported here so the programs tree is only touched by `program`.
    from harnesscad.domain.programs import registry as programs_registry

    return programs_registry.run_cli(args)


def cmd_spec(args: argparse.Namespace) -> int:
    # Imported here so the spec tree is only touched by `spec`.
    from harnesscad.domain.spec import registry as spec_registry

    return spec_registry.run_cli(args)


def cmd_procedural(args: argparse.Namespace) -> int:
    # Imported here so the procedural tree is only touched by `procedural`.
    from harnesscad.domain.procedural import registry as procedural_registry

    return procedural_registry.run_cli(args)


def cmd_catalog(args: argparse.Namespace) -> int:
    # Imported here so the catalogue (and its execution gate) is only touched by
    # `catalog`.
    from harnesscad.domain.library import registry as library_registry

    return library_registry.run_cli(args)


def cmd_fabricate(args: argparse.Namespace) -> int:
    # Imported here so the fabrication tree is only touched by `fabricate`.
    from harnesscad.domain.fabrication import registry as fabrication_registry

    return fabrication_registry.run_cli(args)


def cmd_govern(args: argparse.Namespace) -> int:
    # Imported here so the governance tree is only touched by `govern`.
    from harnesscad.governance import registry as governance_registry

    return governance_registry.run_cli(args)


def cmd_edit(args: argparse.Namespace) -> int:
    # Imported here so the editing tree is only touched by `edit`.
    from harnesscad.domain.editing import registry as editing_registry

    return editing_registry.run_cli(args)


def cmd_search(args: argparse.Namespace) -> int:
    # Imported here so the exploration tree is only touched by `search`.
    from harnesscad.agents.exploration import registry as exploration_registry

    return exploration_registry.run_cli(args)


def cmd_assembly(args: argparse.Namespace) -> int:
    # Imported here so the assembly tree is only touched by `assembly`.
    from harnesscad.domain.assembly import registry as assembly_registry

    return assembly_registry.run_cli(args)


def cmd_generate(args: argparse.Namespace) -> int:
    # Imported here so the generation tree is only touched by `generate`.
    from harnesscad.agents.generation import registry as generation_registry

    return generation_registry.run_cli(args)


def cmd_vision(args: argparse.Namespace) -> int:
    # Imported here so the vision tree is only touched by `vision`.
    from harnesscad.domain.vision import registry as vision_registry

    return vision_registry.run_cli(args)


def cmd_ecosystem(args: argparse.Namespace) -> int:
    # Imported here so the adapter/backend catalogues are only touched by `ecosystem`.
    from harnesscad.io.adapters import registry as ecosystem_registry

    return ecosystem_registry.run_cli(args)


def cmd_ui(args: argparse.Namespace) -> int:
    # Imported here so the interaction surfaces are only touched by `ui`.
    from harnesscad.io.surfaces import registry as surfaces_registry

    return surfaces_registry.run_cli(args)


def cmd_core(args: argparse.Namespace) -> int:
    # Imported here so the core guards are only touched by `core`.
    from harnesscad.core import registry as core_registry

    return core_registry.run_cli(args)


def cmd_agent(args: argparse.Namespace) -> int:
    # Imported here so the agent tree is only touched by `agent`.
    from harnesscad.agents import registry as agents_registry

    return agents_registry.run_cli(args)


def cmd_bench(args: argparse.Namespace) -> int:
    # Imported here so the metric registry (and the bench tree it adapts) is only
    # touched when the `bench` subcommand actually runs.
    from harnesscad.eval.bench import registry as bench_registry

    return bench_registry.run(args)


def cmd_report(args: argparse.Namespace) -> int:
    # Imported here so the quality fleet is only touched by `report`.
    from harnesscad.eval.quality import registry as quality_registry

    return quality_registry.run(args)


def cmd_dataset(args: argparse.Namespace) -> int:
    # Imported here so the data engine is only touched by `dataset`.
    from harnesscad.data import pipeline as data_pipeline

    return data_pipeline.run(args)


def cmd_selftest(args: argparse.Namespace) -> int:
    # Imported here so the self-evaluation oracles (which drive every installed
    # geometry engine) are only touched by `selftest`.
    from harnesscad.eval.selftest import registry as selftest_registry

    return selftest_registry.run(args)


def cmd_pressure(args: argparse.Namespace) -> int:
    # Imported here so the pressure harness (and litellm/ollama, which only it
    # needs) is only touched by `pressure`.
    from harnesscad.eval.pressure import cli as pressure_cli

    return pressure_cli.run(args)


def _brief_from(brief: str) -> str:
    """The brief text: the contents of `brief` when it names a readable file,
    else the literal string. A part brief is short prose, so a bare string is
    the common case; a path is accepted so a longer brief can live in a file."""
    try:
        if os.path.isfile(brief):
            with open(brief, "r", encoding="utf-8") as fh:
                return fh.read()
    except OSError:
        pass
    return brief


def _print_verdict(verdict) -> None:
    """Print a PddVerdict as text: the verdict line, every reason that pushed it
    below PASS, any [NEEDS CLARIFICATION] markers, and the honest residual that
    the synthesis doc requires on every report (a PASS is measured, not proven
    to match intent)."""
    print(f"verdict:  {verdict.verdict}")
    print(f"part:     {verdict.part_id or '?'}")
    print(f"digest:   {verdict.contract_digest}")
    if verdict.failures:
        print("failures:")
        for f in verdict.failures:
            print(f"  - {f}")
    reasons = [r for r in verdict.reasons if r not in verdict.failures]
    if reasons:
        print("reasons:")
        for r in reasons:
            print(f"  - {r}")
    if verdict.clarifications:
        print(f"[NEEDS CLARIFICATION]: {', '.join(verdict.clarifications)}")
    print()
    print(f"honest residual: {verdict.honest_residual}")


def cmd_pdd(args: argparse.Namespace) -> int:
    # Imported here so `apply`/`demo` never pay for the PDD package, and so the
    # CLI does not hard-depend on it: the pipeline's siblings are optional and
    # its import must not break the rest of the CLI.
    from harnesscad.agents.pdd.pipeline import PASS, run_pdd

    try:
        ops = _ops_from(args.ops)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    measurement = None
    if args.measurement:
        try:
            with open(args.measurement, "r", encoding="utf-8") as fh:
                measurement = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"error: could not load measurement from {args.measurement!r}: "
                  f"{exc}", file=sys.stderr)
            return 2
        if not isinstance(measurement, dict):
            print(f"error: {args.measurement!r} must contain a JSON object of "
                  f"contract-keyed measurements", file=sys.stderr)
            return 2

    # The Implement phase: apply the CISP op program through a session on the
    # chosen backend and hand the built session to the pipeline, which re-measures
    # it through io/gate.py. The pipeline hard-depends on no kernel; this executor
    # supplies one so the CLI can run a real part end to end.
    def executor(op_list):
        server = CISPServer(backend=args.backend)
        server.applyOps([dict(op) for op in op_list])
        return server.session

    verdict = run_pdd(
        _brief_from(args.brief),
        ops,
        executor,
        measurement=measurement,
        part_id=args.part_id,
    )

    if args.json:
        print(json.dumps(verdict.as_dict(), indent=2, sort_keys=True, default=str))
    else:
        _print_verdict(verdict)

    # Exit 0 only for a full PASS (certified). UNCERTIFIED and FAIL are both
    # not-certified, so a script/CI gate can key on the exit code.
    return 0 if verdict.verdict == PASS else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cli.py", description="harnesscad CISP CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_apply = sub.add_parser("apply", help="run a JSON array of ops")
    p_apply.add_argument("ops", help="path to a JSON array of ops")
    p_apply.add_argument("--backend", default="stub", choices=BACKEND_CHOICES)
    p_apply.add_argument(
        "--verify", default="core", choices=["core", "full"],
        help="'core' runs the three core checks; 'full' runs the whole discovered "
             "verifier fleet over the plan (advisory: reported, not fatal)",
    )
    p_apply.set_defaults(func=cmd_apply)

    p_demo = sub.add_parser("demo", help="run the built-in constrained-plate sample")
    p_demo.add_argument("--backend", default="stub", choices=BACKEND_CHOICES)
    p_demo.add_argument("--core-only", action="store_true", dest="core_only",
                        help="run only the core verifiers (skip the discovered fleet)")
    p_demo.set_defaults(func=cmd_demo)

    p_build = sub.add_parser(
        "build", help="build a part from a natural-language brief via the LLM planner")
    p_build.add_argument("brief", help="natural-language design brief")
    p_build.add_argument("--backend", default="cadquery", choices=BACKEND_CHOICES)
    p_build.add_argument("--model", default=None, help="model name for the default LLM client")
    p_build.add_argument("--out", default=None, help="write the exported STEP to this path")
    p_build.add_argument("--trace", default=None, help="write JSONL trace events to this path")
    p_build.add_argument("--max-iters", type=int, default=5, dest="max_iters",
                         help="max plan->apply->replan iterations (default 5)")
    p_build.add_argument(
        "--strategy", default="refine", choices=list(BUILD_STRATEGIES),
        help="'refine' feeds the (soundness-gated) diagnostics back and replans. "
             "'best-of-n' draws N independent plans, applies each in a fresh "
             "session and lets the deterministic verifier pick the winner -- no "
             "feedback channel, therefore no poisoning surface.")
    p_build.add_argument("--best-of", type=int, default=4, dest="best_of",
                         help="N for --strategy best-of-n (default 4)")
    p_build.add_argument(
        "--force", action="store_true",
        help="write the artifact even when the output gate refuses it. The "
             "file is written AND a <name>.INVALID.json sidecar naming every "
             "failed measurement is written beside it. For debugging only.")
    p_build.set_defaults(func=cmd_build)

    p_formats = sub.add_parser(
        "formats", help="list the I/O capability matrix (read/write/round-trip)")
    p_formats.add_argument("--kind", default=None,
                           choices=["mesh", "brep", "csg", "drawing", "image"])
    p_formats.add_argument("--mode", default=None, choices=["read", "write"])
    p_formats.add_argument("--json", action="store_true", help="emit the report as JSON")
    p_formats.set_defaults(func=cmd_formats)

    p_export = sub.add_parser(
        "export", help="run ops (or the demo) and write the model to <out>")
    p_export.add_argument("out", help="output path; the extension picks the format")
    p_export.add_argument("--ops", default=None,
                          help="path to a JSON array of ops (default: the built-in demo)")
    p_export.add_argument("--backend", default="stub", choices=BACKEND_CHOICES)
    p_export.add_argument(
        "--force", action="store_true",
        help="write the artifact even when the output gate refuses it. The "
             "file is written AND a <name>.INVALID.json sidecar naming every "
             "failed measurement is written beside it. For debugging only.")
    p_export.set_defaults(func=cmd_export)

    p_render = sub.add_parser(
        "render",
        help="run ops (or the demo) and rasterise the model to a shaded-solid PNG")
    p_render.add_argument("out", help="output PNG path")
    p_render.add_argument("--ops", default=None,
                          help="path to a JSON array of ops (default: the built-in demo)")
    p_render.add_argument("--backend", default="frep",
                          choices=BACKEND_CHOICES,
                          help="geometry backend (default: frep -- it meshes anything)")
    p_render.add_argument("--view", default="iso",
                          choices=sorted(render_views()),
                          help="named camera view (default: iso)")
    p_render.add_argument("--shading", default="smooth", choices=["flat", "smooth"])
    p_render.add_argument("--no-edges", action="store_true", dest="no_edges",
                          help="do not draw the feature edges over the solid")
    p_render.add_argument("--width", type=int, default=1200)
    p_render.add_argument("--height", type=int, default=900)
    p_render.add_argument("--ssaa", type=int, default=2,
                          help="supersampling factor 1..4 (default 2)")
    p_render.add_argument("--projection", default="orthographic",
                          choices=["orthographic", "perspective"])
    p_render.add_argument(
        "--force", action="store_true",
        help="write the artifact even when the output gate refuses it. The "
             "file is written AND a <name>.INVALID.json sidecar naming every "
             "failed measurement is written beside it. For debugging only.")
    p_render.set_defaults(func=cmd_render)

    p_ingest = sub.add_parser(
        "ingest",
        help="decode an existing model (CAD tokens or a mesh) into editable CISP ops")
    p_ingest.add_argument("source",
                          help="path to a tokens JSON document, or a .obj/.xyz mesh")
    p_ingest.add_argument(
        "--family", required=True,
        choices=["deepcad", "skexgen", "hnc", "vitruvion", "mesh"],
        help="the token family to decode with -- MANDATORY and never guessed: the "
             "quantisers are mutually incompatible ('mesh' takes the point-cloud path)")
    p_ingest.add_argument("--backend", default="stub", choices=BACKEND_CHOICES)
    p_ingest.add_argument("--arc-policy", default="chord", dest="arc_policy",
                          choices=["chord", "reject"],
                          help="CISP has no arc op: approximate arcs by their chord "
                               "(default) or refuse them")
    p_ingest.set_defaults(func=cmd_ingest)

    p_reconstruct = sub.add_parser(
        "reconstruct",
        help="reconstruction route registry (input kind -> output kind)")
    from harnesscad.domain.reconstruction import registry as _reconstruction_registry

    _reconstruction_registry.add_arguments(p_reconstruct)
    p_reconstruct.set_defaults(func=cmd_reconstruct)

    p_program = sub.add_parser(
        "program",
        help="code-CAD program surface: parse/validate/emit/review, by --lang")
    from harnesscad.domain.programs import registry as _programs_registry

    _programs_registry.add_arguments(p_program)
    p_program.set_defaults(func=cmd_program)

    p_spec = sub.add_parser(
        "spec",
        help="spec surface: a brief -> a checked spec -> constraints; "
             "EXPRESS/Part-21 validation; structured spec formats")
    from harnesscad.domain.spec import registry as _spec_registry

    _spec_registry.add_arguments(p_spec)
    p_spec.set_defaults(func=cmd_spec)

    p_procedural = sub.add_parser(
        "procedural",
        help="named procedural generators that emit CISP ops (--list/--rivals/--gen)")
    from harnesscad.domain.procedural import registry as _procedural_registry

    _procedural_registry.add_arguments(p_procedural)
    p_procedural.set_defaults(func=cmd_procedural)

    p_catalog = sub.add_parser(
        "catalog",
        help="parts catalogue + standards knowledge base "
             "(--parts/--find/--part/--thread/--heatsert/--aci)")
    from harnesscad.domain.library import registry as _library_registry

    _library_registry.add_arguments(p_catalog)
    p_catalog.set_defaults(func=cmd_catalog)

    p_fabricate = sub.add_parser(
        "fabricate",
        help="manufacturing surface: workflows, feasibility, readiness, "
             "flat-pack, bricks, export planning")
    from harnesscad.domain.fabrication import registry as _fabrication_registry

    _fabrication_registry.add_arguments(p_fabricate)
    p_fabricate.set_defaults(func=cmd_fabricate)

    p_govern = sub.add_parser(
        "govern",
        help="governance surface: security gates, research evidence, audit closure")
    from harnesscad.governance import registry as _governance_registry

    _governance_registry.add_arguments(p_govern)
    p_govern.set_defaults(func=cmd_govern)

    p_edit = sub.add_parser(
        "edit",
        help="edit surface: apply a parametric edit, diff it, or run an edit loop")
    from harnesscad.domain.editing import registry as _editing_registry

    _editing_registry.add_arguments(p_edit)
    p_edit.set_defaults(func=cmd_edit)

    p_assembly = sub.add_parser(
        "assembly",
        help="assembly checks over placed parts (AABB interference + fix vectors)")
    from harnesscad.domain.assembly import registry as _assembly_registry

    _assembly_registry.add_arguments(p_assembly)
    p_assembly.set_defaults(func=cmd_assembly)

    p_search = sub.add_parser(
        "search",
        help="design-space search over a session (--list/--rivals/--strategy)")
    from harnesscad.agents.exploration import registry as _exploration_registry

    _exploration_registry.add_arguments(p_search)
    p_search.set_defaults(func=cmd_search)

    p_generate = sub.add_parser(
        "generate",
        help="generation strategies driving a session (deterministic stub planner)")
    from harnesscad.agents.generation import registry as _generation_registry

    _generation_registry.add_arguments(p_generate)
    p_generate.set_defaults(func=cmd_generate)

    p_vision = sub.add_parser(
        "vision",
        help="vision surface: trace an image back to CISP ops, calibrate px->mm")
    from harnesscad.domain.vision import registry as _vision_registry

    _vision_registry.add_arguments(p_vision)
    p_vision.set_defaults(func=cmd_vision)

    p_ecosystem = sub.add_parser(
        "ecosystem",
        help="which system, which backend, which bridge, which kernel")
    from harnesscad.io.adapters import registry as _ecosystem_registry

    _ecosystem_registry.add_arguments(p_ecosystem)
    p_ecosystem.set_defaults(func=cmd_ecosystem)

    p_ui = sub.add_parser(
        "ui",
        help="interaction surface: command grammar, prediction, overlays, views")
    from harnesscad.io.surfaces import registry as _surfaces_registry

    _surfaces_registry.add_arguments(p_ui)
    p_ui.set_defaults(func=cmd_ui)

    p_core = sub.add_parser(
        "core",
        help="core guards: op-decoding constraints, routing, feature tree, context")
    from harnesscad.core import registry as _core_registry

    _core_registry.add_arguments(p_core)
    p_core.set_defaults(func=cmd_core)

    p_agent = sub.add_parser(
        "agent",
        help="agent surface: envelopes, gates, approval-gated edits, tool metrics")
    from harnesscad.agents import registry as _agents_registry

    _agents_registry.add_arguments(p_agent)
    p_agent.set_defaults(func=cmd_agent)

    p_bench = sub.add_parser(
        "bench",
        help="metric registry + suite runner (--list/--suites/--suite <name>)")
    from harnesscad.eval.bench import registry as _bench_registry

    _bench_registry.add_arguments(p_bench)
    p_bench.set_defaults(func=cmd_bench)

    p_report = sub.add_parser(
        "report",
        help="quality analysis surface (--list/--rivals/--unadapted, or analyse a model)")
    from harnesscad.eval.quality import registry as _quality_registry

    _quality_registry.add_arguments(p_report)
    p_report.set_defaults(func=cmd_report)

    p_dataset = sub.add_parser(
        "dataset",
        help="data-engine pipeline (--list/--presets/--rivals, or run a named pipeline)")
    from harnesscad.data import pipeline as _data_pipeline

    _data_pipeline.add_arguments(p_dataset)
    p_dataset.set_defaults(func=cmd_dataset)

    p_selftest = sub.add_parser(
        "selftest",
        help="SELF-evaluation: the harness evaluating the harness. Six engines "
             "differentially tested against each other, an analytic golden corpus, "
             "precision/recall PER VERIFIER, and metamorphic laws. Points INWARD: "
             "nothing here scores a model.")
    from harnesscad.eval.selftest import registry as _selftest_registry

    _selftest_registry.add_arguments(p_selftest)
    p_selftest.set_defaults(func=cmd_selftest)

    p_pressure = sub.add_parser(
        "pressure",
        help="pressure test: does a typed diagnostic beat a blind retry? "
             "(runs local ollama models through both loops and scores the geometry)")
    from harnesscad.eval.pressure import cli as _pressure_cli

    _pressure_cli.add_arguments(p_pressure)
    p_pressure.set_defaults(func=cmd_pressure)

    p_gallery = sub.add_parser(
        "gallery",
        help="rendered parts gallery: 16 parts, each exercising a different "
             "capability (--list / --build [--out DIR] [--only NAME])")
    from harnesscad.eval.gallery import render_gallery as _render_gallery

    _render_gallery.add_arguments(p_gallery)
    p_gallery.set_defaults(func=cmd_gallery)

    p_caps = sub.add_parser(
        "capabilities",
        help="discover/dispatch capability modules (--list/--search/--show/--stats)")
    from harnesscad import registry as _registry

    _registry.add_arguments(p_caps)
    p_caps.set_defaults(func=cmd_capabilities)

    p_pdd = sub.add_parser(
        "pdd",
        help="Parts-Driven Development: brief -> MGC -> CISP -> artifact -> a "
             "single measured PASS/FAIL/UNCERTIFIED verdict")
    p_pdd.add_argument(
        "brief",
        help="the part brief: a natural-language string, or a path to a file "
             "holding one")
    p_pdd.add_argument(
        "--ops", required=True,
        help="path to a JSON array of CISP ops -- the plan the model built "
             "(the pipeline never generates it)")
    p_pdd.add_argument("--backend", default="stub", choices=BACKEND_CHOICES,
                       help="geometry backend used to build the part (default: stub)")
    p_pdd.add_argument(
        "--measurement", default=None,
        help="path to a JSON object of contract-keyed measurements the MGC is "
             "checked against (e.g. volume_mm3, bbox_mm, genus); when omitted a "
             "best-effort mapping is adapted from the output gate's measurement")
    p_pdd.add_argument("--part-id", default=None, dest="part_id",
                       help="id to stamp on a contract that has none")
    p_pdd.add_argument("--json", action="store_true",
                       help="emit the verdict as JSON")
    p_pdd.set_defaults(func=cmd_pdd)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
