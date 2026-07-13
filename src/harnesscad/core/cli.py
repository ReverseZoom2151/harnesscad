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

    python cli.py formats [--kind mesh|brep|csg|drawing] [--mode read|write] [--json]
        Print the honest I/O capability matrix: every adapted codec, its
        extensions, kind, and whether it can genuinely read / write / round-trip.

    python cli.py export <out.stl> [--ops ops.json] [--backend stub|cadquery|frep]
        Run an op stream (the built-in demo when --ops is omitted) and write the
        resulting model through the format registry; the extension picks the codec.

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
import sys
from typing import List, Optional

from harnesscad.io.formats import registry as formats
from harnesscad.io.surfaces.server import CISPServer


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


def _run_ops(ops: List[dict], backend: str) -> dict:
    server = CISPServer(backend=backend)
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
    result = _run_ops(ops, args.backend)
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
        try:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(step)
        except OSError as exc:
            print(f"error: could not write STEP to {args.out!r}: {exc}", file=sys.stderr)
            return 2
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


def cmd_export(args: argparse.Namespace) -> int:
    """Run an op stream (or the demo), then write the model to <out>."""
    if args.ops:
        try:
            with open(args.ops, "r", encoding="utf-8") as fh:
                ops = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"error: could not load ops from {args.ops!r}: {exc}", file=sys.stderr)
            return 2
        if not isinstance(ops, list):
            print(f"error: {args.ops!r} must contain a JSON array of ops", file=sys.stderr)
            return 2
    else:
        ops = [dict(op) for op in DEMO_OPS]

    server = CISPServer(backend=args.backend)
    result = server.applyOps(ops)
    _print_result(result)
    if not result["ok"]:
        return 1
    try:
        formats.write(server.session, args.out)
    except formats.FormatError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote:    {args.out}")
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cli.py", description="harnesscad CISP CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_apply = sub.add_parser("apply", help="run a JSON array of ops")
    p_apply.add_argument("ops", help="path to a JSON array of ops")
    p_apply.add_argument("--backend", default="stub", choices=["stub", "cadquery", "frep"])
    p_apply.set_defaults(func=cmd_apply)

    p_demo = sub.add_parser("demo", help="run the built-in constrained-plate sample")
    p_demo.add_argument("--backend", default="stub", choices=["stub", "cadquery", "frep"])
    p_demo.add_argument("--core-only", action="store_true", dest="core_only",
                        help="run only the core verifiers (skip the discovered fleet)")
    p_demo.set_defaults(func=cmd_demo)

    p_build = sub.add_parser(
        "build", help="build a part from a natural-language brief via the LLM planner")
    p_build.add_argument("brief", help="natural-language design brief")
    p_build.add_argument("--backend", default="cadquery", choices=["stub", "cadquery", "frep"])
    p_build.add_argument("--model", default=None, help="model name for the default LLM client")
    p_build.add_argument("--out", default=None, help="write the exported STEP to this path")
    p_build.add_argument("--trace", default=None, help="write JSONL trace events to this path")
    p_build.add_argument("--max-iters", type=int, default=5, dest="max_iters",
                         help="max plan->apply->replan iterations (default 5)")
    p_build.set_defaults(func=cmd_build)

    p_formats = sub.add_parser(
        "formats", help="list the I/O capability matrix (read/write/round-trip)")
    p_formats.add_argument("--kind", default=None,
                           choices=["mesh", "brep", "csg", "drawing"])
    p_formats.add_argument("--mode", default=None, choices=["read", "write"])
    p_formats.add_argument("--json", action="store_true", help="emit the report as JSON")
    p_formats.set_defaults(func=cmd_formats)

    p_export = sub.add_parser(
        "export", help="run ops (or the demo) and write the model to <out>")
    p_export.add_argument("out", help="output path; the extension picks the format")
    p_export.add_argument("--ops", default=None,
                          help="path to a JSON array of ops (default: the built-in demo)")
    p_export.add_argument("--backend", default="stub", choices=["stub", "cadquery", "frep"])
    p_export.set_defaults(func=cmd_export)

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
    p_ingest.add_argument("--backend", default="stub", choices=["stub", "cadquery"])
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

    p_caps = sub.add_parser(
        "capabilities",
        help="discover/dispatch capability modules (--list/--search/--show/--stats)")
    from harnesscad import registry as _registry

    _registry.add_arguments(p_caps)
    p_caps.set_defaults(func=cmd_capabilities)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
