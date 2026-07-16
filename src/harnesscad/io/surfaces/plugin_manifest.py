"""Coding-agent plugin manifest surface, generated from the real CLI.

Mined from earthtojake/text-to-cad (resources/cad_repos/text-to-cad-main),
which packages its CAD workflows for coding agents as a *plugin*: a repo-root
``.claude-plugin/marketplace.json`` listing plugins, each plugin directory
carrying a ``.claude-plugin/plugin.json`` manifest plus skill/command files
with YAML frontmatter (``name``/``description``) and Markdown bodies. A
coding agent (Claude Code, Codex) installs the manifest and the harness's
capabilities become first-class commands in the agent's own surface.

This module gives HarnessCAD the same install surface -- generated from the
REAL CLI, never hand-written. It introspects
``harnesscad.core.cli.build_parser()`` (imported lazily; the hub is read, not
edited) and derives:

* :func:`command_specs` -- one :class:`CommandSpec` per CLI verb, with every
  argument's flags, choices, defaults, and help text taken from argparse.
* :func:`plugin_json` / :func:`marketplace_json` -- ``.claude-plugin`` style
  manifests.
* :func:`command_markdown` -- a per-verb command file (frontmatter + usage).
* :func:`skill_markdown` -- one SKILL.md describing the whole verb surface,
  in the SKILL.md dialect text-to-cad ships (frontmatter, trigger section,
  workflow section).
* :func:`write_plugin_tree` -- materialise the full plugin layout::

      <out>/.claude-plugin/marketplace.json
      <out>/plugins/harnesscad/.claude-plugin/plugin.json
      <out>/plugins/harnesscad/commands/<verb>.md
      <out>/plugins/harnesscad/skills/harnesscad/SKILL.md

Because everything is derived from ``build_parser()``, the manifest can never
drift from the CLI: adding a verb to the CLI adds it here on the next
generation, and a verb that does not exist cannot be advertised.

Stdlib-only, deterministic output (sorted keys, stable ordering), absolute
imports. ``--selfcheck`` introspects the live parser and validates the
generated artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

__all__ = [
    "ArgSpec",
    "CommandSpec",
    "command_specs",
    "plugin_json",
    "marketplace_json",
    "command_markdown",
    "skill_markdown",
    "write_plugin_tree",
    "main",
]

PLUGIN_NAME = "harnesscad"
PLUGIN_DESCRIPTION = (
    "Verifier-first text-to-CAD harness: op streams are verified before the "
    "kernel runs them, geometry is cross-checked across independent backends, "
    "and results are measured, not trusted.")
_REPOSITORY = "https://github.com/ReverseZoom2151/harnesscad"


def _package_version() -> str:
    try:
        from importlib.metadata import version
        return version("harnesscad")
    except Exception:
        return "0.1.0"


@dataclass
class ArgSpec:
    """One CLI argument as argparse reported it."""

    name: str                       # dest
    flags: List[str] = field(default_factory=list)   # e.g. ["--backend"]
    required: bool = False
    positional: bool = False
    default: Optional[str] = None
    choices: List[str] = field(default_factory=list)
    help: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "flags": list(self.flags),
            "required": self.required,
            "positional": self.positional,
            "default": self.default,
            "choices": list(self.choices),
            "help": self.help,
        }

    def usage_token(self) -> str:
        if self.positional:
            return f"<{self.name}>" if self.required else f"[{self.name}]"
        flag = self.flags[0] if self.flags else f"--{self.name}"
        value = "" if self.default in ("False", "True") and not self.choices \
            else f" <{'|'.join(self.choices) if self.choices else self.name}>"
        token = f"{flag}{value}"
        return token if self.required else f"[{token}]"


@dataclass
class CommandSpec:
    """One CLI verb: name, help text, and its arguments."""

    name: str
    help: str = ""
    args: List[ArgSpec] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "help": self.help,
            "args": [a.to_dict() for a in self.args],
        }

    def usage(self, prog: str = "harnesscad") -> str:
        tokens = [prog, self.name]
        tokens.extend(a.usage_token() for a in self.args)
        return " ".join(tokens)


def _subparsers_action(
        parser: argparse.ArgumentParser) -> argparse._SubParsersAction:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action
    raise ValueError("parser has no subcommands")


def _arg_specs(sub: argparse.ArgumentParser) -> List[ArgSpec]:
    out: List[ArgSpec] = []
    for action in sub._actions:
        if isinstance(action, (argparse._HelpAction,
                               argparse._SubParsersAction)):
            continue
        positional = not action.option_strings
        default = None
        if action.default is not None and action.default is not argparse.SUPPRESS:
            default = str(action.default)
        out.append(ArgSpec(
            name=action.dest,
            flags=list(action.option_strings),
            required=bool(action.required) if not positional
            else action.nargs not in ("?", "*"),
            positional=positional,
            default=default,
            choices=[str(c) for c in action.choices] if action.choices else [],
            help=(action.help or "").replace("\n", " "),
        ))
    return out


def command_specs() -> List[CommandSpec]:
    """Introspect the live CLI parser into command specs (read-only)."""
    from harnesscad.core.cli import build_parser  # lazy: hub is read, not edited
    parser = build_parser()
    action = _subparsers_action(parser)
    helps: Dict[str, str] = {
        pseudo.dest: (pseudo.help or "")
        for pseudo in action._choices_actions
    }
    specs: List[CommandSpec] = []
    for name in sorted(action.choices):
        sub = action.choices[name]
        specs.append(CommandSpec(
            name=name,
            help=helps.get(name, "") or (sub.description or ""),
            args=_arg_specs(sub),
        ))
    return specs


# ---------------------------------------------------------------------------
# Manifest and Markdown emitters
# ---------------------------------------------------------------------------
def plugin_json(specs: List[CommandSpec], version: Optional[str] = None) -> dict:
    return {
        "name": PLUGIN_NAME,
        "version": version or _package_version(),
        "description": PLUGIN_DESCRIPTION,
        "repository": _REPOSITORY,
        "license": "MIT",
        "keywords": ["cad", "text-to-cad", "verification", "b-rep", "agent"],
        "commands": "./commands/",
        "skills": "./skills/",
        "metadata": {
            "generated_from": "harnesscad.core.cli.build_parser",
            "command_count": len(specs),
        },
    }


def marketplace_json(specs: List[CommandSpec],
                     version: Optional[str] = None) -> dict:
    v = version or _package_version()
    return {
        "name": PLUGIN_NAME,
        "description": PLUGIN_DESCRIPTION,
        "version": v,
        "owner": {"name": "harnesscad"},
        "plugins": [{
            "name": PLUGIN_NAME,
            "source": f"./plugins/{PLUGIN_NAME}",
            "description": PLUGIN_DESCRIPTION,
            "version": v,
            "license": "MIT",
            "category": "productivity",
            "tags": ["cad", "text-to-cad", "verification"],
        }],
    }


def command_markdown(spec: CommandSpec) -> str:
    lines = [
        "---",
        f"name: {spec.name}",
        f"description: {spec.help or 'harnesscad ' + spec.name}",
        "---",
        "",
        f"# harnesscad {spec.name}",
        "",
        spec.help or f"Run the harness `{spec.name}` verb.",
        "",
        "## Usage",
        "",
        "```bash",
        spec.usage(),
        "```",
    ]
    if spec.args:
        lines += ["", "## Arguments", ""]
        for a in spec.args:
            label = a.name if a.positional else ", ".join(a.flags) or a.name
            detail = a.help or ""
            extras = []
            if a.choices:
                extras.append("choices: " + ", ".join(a.choices))
            if a.default not in (None, "False"):
                extras.append(f"default: {a.default}")
            if a.required:
                extras.append("required")
            suffix = f" ({'; '.join(extras)})" if extras else ""
            lines.append(f"- `{label}`: {detail}{suffix}".rstrip())
    lines += [
        "",
        "This file is generated from the live CLI parser by",
        "`harnesscad.io.surfaces.plugin_manifest`; do not edit by hand.",
        "",
    ]
    return "\n".join(lines)


def skill_markdown(specs: List[CommandSpec],
                   version: Optional[str] = None) -> str:
    verb_list = ", ".join(s.name for s in specs)
    lines = [
        "---",
        f"name: {PLUGIN_NAME}",
        "description: Drive the HarnessCAD verifier-first text-to-CAD harness. "
        "Use when generating, verifying, measuring, exporting, rendering, or "
        "benchmarking parametric CAD parts from op streams or natural-language "
        "briefs with soundness guarantees.",
        "---",
        "",
        "# HarnessCAD",
        "",
        PLUGIN_DESCRIPTION,
        "",
        f"Plugin version {version or _package_version()}; generated from the "
        "live CLI parser. The single entry point is the `harnesscad` "
        "executable (`python -m harnesscad.core.cli` from a source tree).",
        "",
        "## Use this skill when",
        "",
        "Use this skill when the user asks to build CAD geometry from a "
        "natural-language brief, apply or verify a CISP op stream, export or "
        "render a model, ingest an existing model into editable ops, or run "
        "the harness's benchmarks and reliability gates.",
        "",
        "## Workflow",
        "",
        "1. Express the design as a CISP op stream or a natural-language "
        "brief.",
        "2. Run the matching verb below; every verb verifies before the "
        "kernel runs and exits nonzero when the result is not certified.",
        "3. Read the emitted diagnostics; they are named, actionable errors, "
        "not stack traces.",
        "4. Repair the smallest responsible op or parameter and rerun.",
        "",
        "## Verbs",
        "",
        f"Available verbs: {verb_list}.",
        "",
    ]
    for s in specs:
        lines.append(f"- `{s.name}`: {s.help}" if s.help else f"- `{s.name}`")
    lines += [
        "",
        "Each verb has a generated command file under `commands/` with full "
        "usage and argument documentation.",
        "",
    ]
    return "\n".join(lines)


def write_plugin_tree(out_dir: Path, version: Optional[str] = None) -> List[Path]:
    """Materialise the full plugin layout; returns the files written."""
    out_dir = Path(out_dir)
    specs = command_specs()
    written: List[Path] = []

    def emit(rel: str, text: str) -> None:
        path = out_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        written.append(path)

    emit(".claude-plugin/marketplace.json",
         json.dumps(marketplace_json(specs, version), indent=2,
                    sort_keys=True) + "\n")
    plugin_root = f"plugins/{PLUGIN_NAME}"
    emit(f"{plugin_root}/.claude-plugin/plugin.json",
         json.dumps(plugin_json(specs, version), indent=2, sort_keys=True) + "\n")
    for spec in specs:
        emit(f"{plugin_root}/commands/{spec.name}.md", command_markdown(spec))
    emit(f"{plugin_root}/skills/{PLUGIN_NAME}/SKILL.md",
         skill_markdown(specs, version))
    return written


# ---------------------------------------------------------------------------
# selfcheck + CLI
# ---------------------------------------------------------------------------
def _selfcheck() -> int:
    failures: List[str] = []

    def check(cond: bool, message: str) -> None:
        if not cond:
            failures.append(message)

    specs = command_specs()
    names = [s.name for s in specs]
    check(len(specs) >= 10, f"CLI exposes a real verb surface ({len(specs)})")
    for expected in ("apply", "build", "export", "render"):
        check(expected in names, f"core verb '{expected}' present")
    check(names == sorted(names), "verbs are emitted in sorted order")

    apply_spec = next(s for s in specs if s.name == "apply")
    check(any(a.positional and a.name == "ops" for a in apply_spec.args),
          "apply's positional 'ops' argument introspected")
    check(any(a.flags == ["--backend"] and a.choices for a in apply_spec.args),
          "apply's --backend choices introspected")
    check("apply" in apply_spec.usage(), "usage string renders")

    pj = plugin_json(specs)
    check(pj["name"] == PLUGIN_NAME and pj["commands"] == "./commands/",
          "plugin.json shape")
    check(pj["metadata"]["command_count"] == len(specs),
          "plugin.json command count matches the real surface")
    mj = marketplace_json(specs)
    check(mj["plugins"][0]["source"] == f"./plugins/{PLUGIN_NAME}",
          "marketplace.json points at the plugin dir")
    check(json.loads(json.dumps(mj)) == mj, "marketplace JSON serialisable")

    md = command_markdown(apply_spec)
    check(md.startswith("---\nname: apply\n"), "command markdown frontmatter")
    check("## Usage" in md and "## Arguments" in md,
          "command markdown sections present")
    sk = skill_markdown(specs)
    check(sk.startswith(f"---\nname: {PLUGIN_NAME}\n"),
          "skill markdown frontmatter")
    check("## Use this skill when" in sk and "## Workflow" in sk,
          "skill markdown trigger and workflow sections present")
    check(all(f"`{n}`" in sk for n in names), "every verb listed in SKILL.md")

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        written = write_plugin_tree(Path(tmp), version="0.0.0-selfcheck")
        rels = sorted(str(p.relative_to(tmp)).replace("\\", "/")
                      for p in written)
        check(".claude-plugin/marketplace.json" in rels,
              "marketplace manifest written")
        check(f"plugins/{PLUGIN_NAME}/.claude-plugin/plugin.json" in rels,
              "plugin manifest written")
        check(f"plugins/{PLUGIN_NAME}/skills/{PLUGIN_NAME}/SKILL.md" in rels,
              "SKILL.md written")
        check(len([r for r in rels if r.endswith(".md")
                   and "/commands/" in r]) == len(specs),
              "one command file per verb")

    for message in failures:
        print("selfcheck FAIL: " + message)
    print("selfcheck: %s" % ("PASS" if not failures else "FAIL"))
    return 0 if not failures else 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="plugin_manifest",
        description="generate a coding-agent plugin manifest from the real "
                    "harnesscad CLI surface")
    parser.add_argument("--selfcheck", action="store_true",
                        help="run the built-in self-test and exit")
    parser.add_argument("--out", metavar="DIR",
                        help="write the full plugin tree under DIR")
    parser.add_argument("--json", action="store_true",
                        help="print the introspected command specs as JSON")
    parser.add_argument("--version", default=None,
                        help="override the plugin version string")
    args = parser.parse_args(argv)

    if args.selfcheck:
        return _selfcheck()
    if args.json:
        specs = command_specs()
        print(json.dumps({
            "plugin": plugin_json(specs, args.version),
            "commands": [s.to_dict() for s in specs],
        }, indent=2, sort_keys=True))
        return 0
    if args.out:
        written = write_plugin_tree(Path(args.out), version=args.version)
        for path in written:
            print(path)
        print(f"wrote {len(written)} files under {args.out}")
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
